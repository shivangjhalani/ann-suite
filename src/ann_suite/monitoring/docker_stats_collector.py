"""DockerStatsCollector - BaseCollector implementation using Docker stats API.

This collector implements the BaseCollector interface by wrapping the Docker
stats streaming API. It provides a unified interface compatible with other
collectors like CgroupsV2Collector.

Note: For accurate disk I/O operations, prefer CgroupsV2Collector when available.
This collector calculates approximate IOPS from byte deltas and block size.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import docker

from ann_suite.monitoring.base import BaseCollector, CollectorResult, CollectorSample

if TYPE_CHECKING:
    import docker.models.containers

logger = logging.getLogger(__name__)


def get_system_block_size() -> int:
    """Detect the system's physical block size.

    Attempts to read the physical block size from /sys/block/*/queue/physical_block_size.
    Falls back to 4096 if detection fails.

    Returns:
        Block size in bytes (typically 512 or 4096).
    """
    from pathlib import Path

    sys_block = Path("/sys/block")
    if sys_block.exists():
        for device_dir in sys_block.iterdir():
            if device_dir.name.startswith(("loop", "ram", "dm-")):
                continue
            block_size_file = device_dir / "queue" / "physical_block_size"
            if block_size_file.exists():
                try:
                    with open(block_size_file) as f:
                        return int(f.read().strip())
                except (ValueError, OSError):
                    continue

    logger.debug("Could not detect block size, using default 4096")
    return 4096


class DockerStatsCollector(BaseCollector):
    """BaseCollector implementation using Docker stats API.

    This collector streams stats from Docker containers and aggregates them
    into the standard CollectorResult format.

    Note: IOPS values from this collector are approximations calculated from
    byte deltas divided by block size. For true IOPS, use CgroupsV2Collector.

    Example:
        ```python
        collector = DockerStatsCollector(interval_ms=100)
        if collector.is_available():
            collector.start(container.id)
            # ... wait for container work ...
            result = collector.stop()
            print(f"Peak memory: {result.peak_memory_mb} MB")
        ```
    """

    def __init__(self, interval_ms: int = 100) -> None:
        """Initialize the Docker stats collector.

        Args:
            interval_ms: Sampling interval in milliseconds (50-1000)
        """
        self._interval_seconds = max(0.05, min(1.0, interval_ms / 1000))
        self._container: docker.models.containers.Container | None = None
        self._container_id: str | None = None
        self._samples: list[CollectorSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._client: docker.DockerClient | None = None

    @property
    def name(self) -> str:
        return "docker_stats"

    def is_available(self) -> bool:
        """Check if Docker is available."""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def start(self, container_id: str) -> None:
        """Start collecting metrics for a container.

        Args:
            container_id: Docker container ID (short or full)
        """
        if self._running:
            logger.warning("DockerStatsCollector already running")
            return

        try:
            self._client = docker.from_env()
            self._container = self._client.containers.get(container_id)
            self._container_id = container_id
        except Exception as e:
            logger.warning(f"Could not get container {container_id}: {e}")
            return

        self._running = True
        self._start_time = time.monotonic()
        self._samples = []

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug(f"Started Docker stats collection for {container_id[:12]}")

    def stop(self) -> CollectorResult:
        """Stop collecting and return aggregated metrics."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._container = None
        self._container_id = None

        return self._aggregate_samples()

    def _monitor_loop(self) -> None:
        """Background loop that samples container stats using streaming mode."""
        if self._container is None:
            return

        try:
            stats_stream = self._container.stats(stream=True, decode=True)
            for stats in stats_stream:
                if not self._running:
                    break

                sample = self._parse_stats(stats)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)

        except Exception as e:
            if "404" in str(e) or "not running" in str(e).lower():
                logger.debug(f"Container {self._container_id[:12] if self._container_id else '?'} stopped")
            else:
                logger.warning(f"Error in stats stream: {e}")

    def _parse_stats(self, stats: dict[str, Any]) -> CollectorSample | None:
        """Parse Docker stats JSON into a CollectorSample."""
        try:
            now = datetime.now()

            # Memory stats
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)

            # CPU stats
            cpu_percent = self._calculate_cpu_percent(stats)

            # Block I/O stats
            blkio_read, blkio_write = self._parse_blkio_stats(stats)

            return CollectorSample(
                timestamp=now,
                memory_usage_bytes=memory_usage,
                cpu_percent=cpu_percent,
                cpu_time_ns=0,  # Not available from Docker stats
                blkio_read_bytes=blkio_read,
                blkio_write_bytes=blkio_write,
                blkio_read_ops=0,  # Calculated during aggregation
                blkio_write_ops=0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse stats: {e}")
            return None

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu_stats.get(
                "cpu_usage", {}
            ).get("total_usage", 0)

            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
                "system_cpu_usage", 0
            )

            if system_delta > 0 and cpu_delta > 0:
                num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [])) or 1
                return (cpu_delta / system_delta) * num_cpus * 100.0

            return 0.0
        except Exception:
            return 0.0

    def _parse_blkio_stats(self, stats: dict[str, Any]) -> tuple[int, int]:
        """Parse block I/O statistics from Docker stats."""
        blkio_stats = stats.get("blkio_stats", {})
        io_bytes = blkio_stats.get("io_service_bytes_recursive") or []

        read_bytes = 0
        write_bytes = 0

        for entry in io_bytes:
            op = entry.get("op", "").lower()
            value = entry.get("value", 0)
            if op == "read":
                read_bytes += value
            elif op == "write":
                write_bytes += value

        return read_bytes, write_bytes

    def _aggregate_samples(self) -> CollectorResult:
        """Aggregate collected samples into a CollectorResult."""
        with self._lock:
            raw_samples = list(self._samples)

        # Filter out samples where container has stopped
        samples = [s for s in raw_samples if s.memory_usage_bytes > 0]

        if not samples and raw_samples:
            logger.debug("All samples had 0 memory (short run?), using raw samples")
            samples = raw_samples

        if not samples:
            return CollectorResult(sample_count=0)

        # Memory metrics
        memory_values = [s.memory_usage_bytes / (1024 * 1024) for s in samples]
        peak_memory_mb = max(memory_values)
        avg_memory_mb = sum(memory_values) / len(memory_values)

        # CPU metrics
        cpu_values = [s.cpu_percent for s in samples]
        peak_cpu_percent = max(cpu_values)
        avg_cpu_percent = sum(cpu_values) / len(cpu_values)

        # Block I/O totals
        final_sample = samples[-1]
        total_read_bytes = final_sample.blkio_read_bytes
        total_write_bytes = final_sample.blkio_write_bytes

        # Calculate approximate IOPS from byte deltas
        avg_read_iops, avg_write_iops = self._calculate_average_iops(samples)

        # Duration
        if len(samples) >= 2:
            duration = (samples[-1].timestamp - samples[0].timestamp).total_seconds()
        else:
            duration = 0

        return CollectorResult(
            cpu_time_total_seconds=0.0,  # Not available from Docker stats
            avg_cpu_percent=avg_cpu_percent,
            peak_cpu_percent=peak_cpu_percent,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            total_read_ops=0,  # Approximated via IOPS
            total_write_ops=0,
            avg_read_iops=avg_read_iops,
            avg_write_iops=avg_write_iops,
            duration_seconds=duration,
            sample_count=len(samples),
        )

    def _calculate_average_iops(self, samples: list[CollectorSample]) -> tuple[float, float]:
        """Calculate approximate IOPS from sample byte deltas.

        Note: This is an approximation. For true IOPS, use CgroupsV2Collector.
        """
        if len(samples) < 2:
            return 0.0, 0.0

        total_read_ops = 0.0
        total_write_ops = 0.0
        total_time = 0.0
        block_size = get_system_block_size()

        for i in range(1, len(samples)):
            prev = samples[i - 1]
            curr = samples[i]

            time_delta = (curr.timestamp - prev.timestamp).total_seconds()
            if time_delta <= 0:
                continue

            read_delta = curr.blkio_read_bytes - prev.blkio_read_bytes
            write_delta = curr.blkio_write_bytes - prev.blkio_write_bytes

            if read_delta > 0:
                total_read_ops += read_delta / block_size
            if write_delta > 0:
                total_write_ops += write_delta / block_size

            total_time += time_delta

        if total_time > 0:
            return total_read_ops / total_time, total_write_ops / total_time

        return 0.0, 0.0
