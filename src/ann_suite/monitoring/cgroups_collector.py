"""cgroups v2 collector for direct kernel metrics.

This collector reads metrics directly from the cgroups v2 filesystem,
providing the most accurate container resource metrics without Docker API overhead.

Metrics sourced:
- io.stat: rbytes, wbytes, rios, wios (I/O bytes and operations)
- cpu.stat: usage_usec (CPU time)
- memory.current: current memory usage
- memory.peak: peak memory usage (if available)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from ann_suite.monitoring.base import BaseCollector, CollectorResult, CollectorSample

logger = logging.getLogger(__name__)


class CgroupsV2Collector(BaseCollector):
    """Collector that reads metrics directly from cgroups v2 filesystem.

    This provides the most accurate I/O metrics (IOPS, bytes) for containers
    by reading directly from /sys/fs/cgroup/.

    Requires:
    - cgroups v2 mounted at /sys/fs/cgroup (unified hierarchy)
    - Container cgroup path discoverable (Docker/Podman)
    """

    # Common cgroup paths for Docker containers
    DOCKER_CGROUP_PATTERNS = [
        "/sys/fs/cgroup/system.slice/docker-{container_id}.scope",
        "/sys/fs/cgroup/docker/{container_id}",
    ]

    def __init__(self, interval_ms: int = 100) -> None:
        """Initialize the cgroups v2 collector.

        Args:
            interval_ms: Sampling interval in milliseconds
        """
        self._interval_seconds = max(0.05, min(1.0, interval_ms / 1000))
        self._container_id: str | None = None
        self._cgroup_path: Path | None = None
        self._samples: list[CollectorSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None

    @property
    def name(self) -> str:
        return "cgroups_v2"

    def is_available(self) -> bool:
        """Check if cgroups v2 is available on this system."""
        cgroup_path = Path("/sys/fs/cgroup")
        if not cgroup_path.exists():
            return False

        # Check for cgroups v2 (unified hierarchy) by looking for cgroup.controllers
        controllers_file = cgroup_path / "cgroup.controllers"
        return controllers_file.exists()

    def start(self, container_id: str) -> None:
        """Start collecting metrics for a container.

        Args:
            container_id: Docker container ID (short or full)
        """
        if self._running:
            logger.warning("CgroupsV2Collector already running")
            return

        self._container_id = container_id
        self._cgroup_path = self._find_cgroup_path(container_id)

        if self._cgroup_path is None:
            logger.warning(f"Could not find cgroup path for container {container_id}")
            # Fall back to not collecting - metrics will be zero
            return

        logger.debug(f"Found cgroup path: {self._cgroup_path}")

        self._running = True
        self._start_time = time.monotonic()
        self._samples = []

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> CollectorResult:
        """Stop collecting and return aggregated metrics."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        return self._aggregate_samples()

    def _find_cgroup_path(self, container_id: str) -> Path | None:
        """Find the cgroup path for a Docker container.

        Args:
            container_id: Short or full container ID

        Returns:
            Path to container's cgroup, or None if not found
        """
        # Try known patterns
        for pattern in self.DOCKER_CGROUP_PATTERNS:
            path = Path(pattern.format(container_id=container_id))
            logger.debug(f"Checking cgroup pattern: {path}")
            if path.exists():
                return path

        # Try to find by searching (more expensive but catches edge cases)
        cgroup_base = Path("/sys/fs/cgroup")
        logger.debug(f"Searching for cgroup in {cgroup_base}")

        # Search in system.slice for docker-*.scope
        system_slice = cgroup_base / "system.slice"
        if system_slice.exists():
            for child in system_slice.iterdir():
                if container_id[:12] in child.name and child.is_dir():
                    return child

        logger.warning(f"Failed to find cgroup path for {container_id}")
        return None

    def _monitor_loop(self) -> None:
        """Background loop that samples cgroup stats."""
        while self._running:
            try:
                sample = self._collect_sample()
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
            except Exception as e:
                logger.debug(f"Error collecting cgroup sample: {e}")

            time.sleep(self._interval_seconds)

    def _collect_sample(self) -> CollectorSample | None:
        """Collect a single sample from cgroup files."""
        if self._cgroup_path is None:
            return None

        now = datetime.now()

        # Read memory
        memory_current = self._read_single_value(self._cgroup_path / "memory.current")

        # Read CPU time (in microseconds)
        cpu_stat = self._read_cpu_stat()
        cpu_time_ns = cpu_stat.get("usage_usec", 0) * 1000  # Convert to nanoseconds

        # Read I/O stats
        io_stat = self._read_io_stat()

        return CollectorSample(
            timestamp=now,
            memory_usage_bytes=memory_current,
            cpu_percent=0.0,  # Calculated during aggregation
            cpu_time_ns=cpu_time_ns,
            blkio_read_bytes=io_stat.get("rbytes", 0),
            blkio_write_bytes=io_stat.get("wbytes", 0),
            blkio_read_ops=io_stat.get("rios", 0),
            blkio_write_ops=io_stat.get("wios", 0),
        )

    def _read_single_value(self, path: Path) -> int:
        """Read a single integer value from a cgroup file."""
        try:
            return int(path.read_text().strip())
        except (FileNotFoundError, ValueError, PermissionError):
            return 0

    def _read_cpu_stat(self) -> dict[str, int]:
        """Read cpu.stat file.

        Format:
            usage_usec 123456
            user_usec 100000
            system_usec 23456
        """
        result: dict[str, int] = {}
        cpu_stat_path = self._cgroup_path / "cpu.stat"  # type: ignore

        try:
            content = cpu_stat_path.read_text()
            for line in content.strip().split("\n"):
                parts = line.split()
                if len(parts) == 2:
                    result[parts[0]] = int(parts[1])
        except (FileNotFoundError, PermissionError):
            pass

        return result

    def _read_io_stat(self) -> dict[str, int]:
        """Read io.stat file.

        Format (per device):
            8:0 rbytes=12345 wbytes=67890 rios=100 wios=50 ...

        Returns aggregated values across all devices.
        """
        result: dict[str, int] = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
        io_stat_path = self._cgroup_path / "io.stat"  # type: ignore

        try:
            content = io_stat_path.read_text()
            for line in content.strip().split("\n"):
                if not line:
                    continue
                # Parse key=value pairs
                for match in re.finditer(r"(rbytes|wbytes|rios|wios)=(\d+)", line):
                    key, value = match.groups()
                    result[key] += int(value)
        except (FileNotFoundError, PermissionError):
            pass

        return result

    def _aggregate_samples(self) -> CollectorResult:
        """Aggregate collected samples into a result."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return CollectorResult()

        # Filter out samples with zero memory (container may have stopped)
        valid_samples = [s for s in samples if s.memory_usage_bytes > 0]
        if not valid_samples:
            return CollectorResult(sample_count=len(samples))

        # Memory metrics
        memory_values = [s.memory_usage_bytes / (1024 * 1024) for s in valid_samples]
        peak_memory_mb = max(memory_values)
        avg_memory_mb = sum(memory_values) / len(memory_values)

        # CPU metrics
        # Calculate total CPU time from first to last sample
        first_sample = valid_samples[0]
        last_sample = valid_samples[-1]
        cpu_time_delta_ns = last_sample.cpu_time_ns - first_sample.cpu_time_ns
        cpu_time_total_seconds = cpu_time_delta_ns / 1e9

        # Duration
        duration = (last_sample.timestamp - first_sample.timestamp).total_seconds()

        # Calculate average CPU percent
        avg_cpu_percent = (cpu_time_total_seconds / duration * 100) if duration > 0 else 0.0

        # Calculate peak CPU percent by checking each interval
        peak_cpu_percent = 0.0
        for i in range(1, len(valid_samples)):
            s1 = valid_samples[i - 1]
            s2 = valid_samples[i]

            interval_ns = s2.cpu_time_ns - s1.cpu_time_ns
            interval_duration = (s2.timestamp - s1.timestamp).total_seconds()

            if interval_duration > 0:
                interval_cpu = (interval_ns / 1e9) / interval_duration * 100
                peak_cpu_percent = max(peak_cpu_percent, interval_cpu)

        # Calculate I/O metrics
        read_ops_delta = last_sample.blkio_read_ops - first_sample.blkio_read_ops
        write_ops_delta = last_sample.blkio_write_ops - first_sample.blkio_write_ops
        avg_read_iops = (read_ops_delta / duration) if duration > 0 else 0.0
        avg_write_iops = (write_ops_delta / duration) if duration > 0 else 0.0

        return CollectorResult(
            cpu_time_total_seconds=cpu_time_total_seconds,
            avg_cpu_percent=avg_cpu_percent,
            peak_cpu_percent=peak_cpu_percent,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            total_read_bytes=last_sample.blkio_read_bytes,
            total_write_bytes=last_sample.blkio_write_bytes,
            total_read_ops=last_sample.blkio_read_ops,
            total_write_ops=last_sample.blkio_write_ops,
            avg_read_iops=avg_read_iops,
            avg_write_iops=avg_write_iops,
            duration_seconds=duration,
            sample_count=len(valid_samples),
        )
