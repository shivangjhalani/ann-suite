"""Resource monitoring for Docker containers.

This module implements a background monitor that collects resource usage metrics
from running Docker containers, including:
- Memory usage (peak and average)
- Block I/O statistics (read/write bytes and IOPS)
- CPU utilization

The monitor runs in a background thread and samples at configurable intervals.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ann_suite.core.schemas import ResourceSample, ResourceSummary

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

    # Try to detect from first available block device
    sys_block = Path("/sys/block")
    if sys_block.exists():
        for device_dir in sys_block.iterdir():
            # Skip virtual devices like loop*, ram*, etc.
            if device_dir.name.startswith(("loop", "ram", "dm-")):
                continue
            block_size_file = device_dir / "queue" / "physical_block_size"
            if block_size_file.exists():
                try:
                    with open(block_size_file) as f:
                        return int(f.read().strip())
                except (ValueError, OSError):
                    continue

    # Fallback to 4KB (common default)
    logger.debug("Could not detect block size, using default 4096")
    return 4096


class ResourceMonitor:
    """Background resource monitor for Docker containers.

    Collects resource metrics at regular intervals while a container is running.
    Designed to capture metrics critical for evaluating disk-based ANN algorithms:
    - Block I/O throughput and IOPS
    - Peak and average memory usage

    Example:
        ```python
        monitor = ResourceMonitor(container, interval_ms=100)
        monitor.start()
        # ... wait for container to complete work ...
        summary = monitor.stop()
        print(f"Peak memory: {summary.peak_memory_mb} MB")
        ```
    """

    def __init__(
        self,
        container: docker.models.containers.Container,
        interval_ms: int = 100,
    ) -> None:
        """Initialize the resource monitor.

        Args:
            container: Docker container object to monitor
            interval_ms: Sampling interval in milliseconds (50-1000)
        """
        self._container = container
        self._interval_seconds = max(0.05, min(1.0, interval_ms / 1000))
        self._samples: list[ResourceSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._prev_blkio_read: int = 0
        self._prev_blkio_write: int = 0
        self._prev_sample_time: float | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._start_time = time.monotonic()
        self._samples = []
        self._prev_blkio_read = 0
        self._prev_blkio_write = 0
        self._prev_sample_time = None

        # Start monitoring thread immediately - it uses streaming mode for fast sample delivery
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug(f"Started monitoring container {self._container.short_id}")

    def stop(self) -> ResourceSummary:
        """Stop monitoring and return aggregated metrics.

        Returns:
            ResourceSummary with peak/avg memory, IOPS, etc.
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        return self._aggregate_samples()

    def _monitor_loop(self) -> None:
        """Background loop that samples container stats using streaming mode.

        Uses stream=True for faster sample delivery (~100ms vs ~1s for stream=False).
        This is critical for capturing stats from fast-running containers.
        """
        try:
            # Use streaming mode for faster sample delivery
            stats_stream = self._container.stats(stream=True, decode=True)
            for stats in stats_stream:
                if not self._running:
                    break

                sample = self._parse_stats(stats)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)

        except Exception as e:
            # Container may have stopped - this is expected
            if "404" in str(e) or "not running" in str(e).lower():
                logger.debug(f"Container {self._container.short_id} stopped")
            else:
                logger.warning(f"Error in stats stream: {e}")

    def _collect_sample(self) -> None:
        """Collect a single sample from Docker stats API."""
        try:
            stats = self._container.stats(stream=False)
        except Exception:
            # Container may have stopped
            return

        sample = self._parse_stats(stats)
        if sample is not None:
            with self._lock:
                self._samples.append(sample)

    def _parse_stats(self, stats: dict[str, Any]) -> ResourceSample | None:
        """Parse Docker stats JSON into a ResourceSample.

        This method carefully extracts metrics from the Docker stats API,
        handling the nested structure and potential missing fields.

        Args:
            stats: Raw stats dict from container.stats()

        Returns:
            ResourceSample or None if parsing fails
        """
        try:
            now = datetime.now()
            current_time = time.monotonic()

            # Memory stats
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0

            # CPU stats - calculate percentage
            cpu_percent = self._calculate_cpu_percent(stats)

            # Block I/O stats
            blkio_read, blkio_write = self._parse_blkio_stats(stats)

            # PIDs
            pids = stats.get("pids_stats", {}).get("current", 1)

            # Track previous values for IOPS calculation
            self._prev_blkio_read = blkio_read
            self._prev_blkio_write = blkio_write
            self._prev_sample_time = current_time

            return ResourceSample(
                timestamp=now,
                memory_usage_bytes=memory_usage,
                memory_limit_bytes=memory_limit,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                blkio_read_bytes=blkio_read,
                blkio_write_bytes=blkio_write,
                pids=pids,
            )
        except Exception as e:
            logger.warning(f"Failed to parse stats: {e}")
            return None

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        """Calculate CPU percentage from Docker stats.

        Uses the delta between current and previous CPU usage readings.
        """
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
        """Parse block I/O statistics from Docker stats.

        Extracts read and write bytes from io_service_bytes_recursive.
        This is critical for evaluating disk-based ANN algorithms.

        Returns:
            Tuple of (read_bytes, write_bytes)
        """
        blkio_stats = stats.get("blkio_stats", {})

        # io_service_bytes_recursive contains an array of operations
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

    def _aggregate_samples(self) -> ResourceSummary:
        """Aggregate collected samples into a summary.

        Calculates peak/average values and IOPS from the raw samples.
        Filters out samples with zero memory usage (indicating container stopped).
        """
        with self._lock:
            raw_samples = list(self._samples)

        # Filter out samples where container has stopped (memory_usage_bytes == 0)
        # Docker stats returns zeros after container exits
        samples = [s for s in raw_samples if s.memory_usage_bytes > 0]

        # If we filtered everything (container too fast), fall back to raw samples
        if not samples and raw_samples:
           logger.debug("All samples had 0 memory (short run?), using raw samples")
           samples = raw_samples

        if not samples:
            return ResourceSummary(
                peak_memory_mb=0,
                avg_memory_mb=0,
                avg_cpu_percent=0,
                peak_cpu_percent=0,
                total_blkio_read_mb=0,
                total_blkio_write_mb=0,
                avg_read_iops=0,
                avg_write_iops=0,
                sample_count=0,
                duration_seconds=0,
                samples=raw_samples,  # Keep all samples for debugging
            )

        # Calculate memory metrics
        memory_values = [s.memory_usage_bytes / (1024 * 1024) for s in samples]
        peak_memory_mb = max(memory_values)
        avg_memory_mb = sum(memory_values) / len(memory_values)

        # Calculate CPU metrics
        cpu_values = [s.cpu_percent for s in samples]
        peak_cpu_percent = max(cpu_values)
        avg_cpu_percent = sum(cpu_values) / len(cpu_values)

        # Block I/O totals (use last valid sample values as cumulative)
        final_sample = samples[-1]
        total_blkio_read_mb = final_sample.blkio_read_bytes / (1024 * 1024)
        total_blkio_write_mb = final_sample.blkio_write_bytes / (1024 * 1024)

        # Calculate IOPS from sample deltas
        avg_read_iops, avg_write_iops = self._calculate_average_iops(samples)



        # Duration
        if len(samples) >= 2:
            duration = (samples[-1].timestamp - samples[0].timestamp).total_seconds()
        else:
            duration = 0

        return ResourceSummary(
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            peak_cpu_percent=peak_cpu_percent,
            total_blkio_read_mb=total_blkio_read_mb,
            total_blkio_write_mb=total_blkio_write_mb,
            avg_read_iops=avg_read_iops,
            avg_write_iops=avg_write_iops,
            sample_count=len(samples),
            duration_seconds=duration,
            samples=raw_samples,  # Keep all samples for debugging
        )

    def _calculate_average_iops(self, samples: list[ResourceSample]) -> tuple[float, float]:
        """Calculate average IOPS from sample deltas.

        IOPS is calculated as the change in bytes between samples divided
        by the time delta and the system block size (detected or 4KB default).
        """
        if len(samples) < 2:
            return 0.0, 0.0

        total_read_ops = 0.0
        total_write_ops = 0.0
        total_time = 0.0
        block_size = get_system_block_size()  # Detect from system

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
