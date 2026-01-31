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
from datetime import UTC, datetime
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
        return CgroupsV2Collector.check_available()

    @staticmethod
    def check_available() -> bool:
        """Check if cgroups v2 is available on this system.

        This static method can be called without an instance to verify
        cgroups v2 availability before creating a collector.

        Returns:
            True if cgroups v2 is available and readable
        """
        cgroup_path = Path("/sys/fs/cgroup")
        if not cgroup_path.exists():
            return False

        # Check for cgroups v2 (unified hierarchy) by looking for cgroup.controllers
        controllers_file = cgroup_path / "cgroup.controllers"
        if not controllers_file.exists():
            return False

        # Verify we can actually read the controllers file
        try:
            controllers_file.read_text()
            return True
        except (PermissionError, OSError):
            return False

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
            raise RuntimeError(
                f"Could not find cgroup path for container {container_id}. "
                "Ensure the container runtime is using cgroups v2."
            )

        logger.debug(f"Found cgroup path: {self._cgroup_path}")

        self._running = True
        self._start_time = time.monotonic()
        self._samples = []

        # Collect initial sample synchronously to capture fast containers
        initial_sample = self._collect_sample()
        if initial_sample is not None:
            self._samples.append(initial_sample)
            logger.debug("Collected initial cgroups sample")

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> CollectorResult:
        """Stop collecting and return aggregated metrics."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Collect final sample synchronously for accurate deltas
        if self._cgroup_path is not None:
            final_sample = self._collect_sample()
            if final_sample is not None:
                with self._lock:
                    self._samples.append(final_sample)
                logger.debug("Collected final cgroups sample")

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

        now = datetime.now(UTC)
        mono_now = time.monotonic()

        # Read memory
        memory_current = self._read_single_value(self._cgroup_path / "memory.current")

        # Read CPU time (in microseconds)
        cpu_stat = self._read_cpu_stat()
        cpu_time_ns = cpu_stat.get("usage_usec", 0) * 1000  # Convert to nanoseconds

        # Read I/O stats
        io_stat = self._read_io_stat()

        return CollectorSample(
            timestamp=now,
            monotonic_time=mono_now,
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

    def get_summary(
        self,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
    ) -> CollectorResult:
        """Get aggregated metrics, optionally filtered by time window.

        Args:
            start_timestamp: Start of the window (inclusive)
            end_timestamp: End of the window (inclusive)

        Returns:
            Aggregated metrics for the specified window
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return CollectorResult()

        if start_timestamp or end_timestamp:
            filtered_samples = []
            for s in samples:
                if start_timestamp and s.timestamp < start_timestamp:
                    continue
                if end_timestamp and s.timestamp > end_timestamp:
                    continue
                filtered_samples.append(s)

            if len(filtered_samples) < 2 and len(samples) >= 2:
                logger.debug(
                    f"Time window filtering left {len(filtered_samples)} samples "
                    f"(original: {len(samples)}). Using nearest bracketing samples."
                )
                bracketed_samples = self._find_bracketing_samples(
                    samples, start_timestamp, end_timestamp
                )
                if len(bracketed_samples) >= 2:
                    samples = bracketed_samples
                else:
                    logger.warning("Could not find valid bracketing samples, using all samples")
            else:
                samples = filtered_samples

        # Filtering preserves order so samples remain sorted
        return self._aggregate_samples(samples, already_sorted=True)

    def _find_bracketing_samples(
        self,
        samples: list[CollectorSample],
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
    ) -> list[CollectorSample]:
        """Find samples that bracket the requested time window.

        Returns the sample closest to (but before or at) start_timestamp and
        the sample closest to (but after or at) end_timestamp.
        """
        before_sample: CollectorSample | None = None
        after_sample: CollectorSample | None = None

        for s in samples:
            if start_timestamp and s.timestamp <= start_timestamp:
                if before_sample is None or s.timestamp > before_sample.timestamp:
                    before_sample = s
            if end_timestamp and s.timestamp >= end_timestamp:
                if after_sample is None or s.timestamp < after_sample.timestamp:
                    after_sample = s

        if before_sample is None and samples:
            before_sample = samples[0]
        if after_sample is None and samples:
            after_sample = samples[-1]

        if before_sample and after_sample and before_sample is not after_sample:
            return [before_sample, after_sample]
        return []

    def _aggregate_samples(
        self, samples: list[CollectorSample] | None = None, *, already_sorted: bool = False
    ) -> CollectorResult:
        """Aggregate collected samples into a result.

        Args:
            samples: Optional list of samples to aggregate. Uses self._samples if None.
            already_sorted: If True, skip sorting (samples collected sequentially are
                            already in monotonic order).
        """
        if samples is None:
            with self._lock:
                samples = list(self._samples)
            # Samples from the internal list are always collected in order
            already_sorted = True

        if not samples:
            logger.warning("No cgroups samples collected - metrics will be zero")
            return CollectorResult()

        # Only sort if samples may be out of order (e.g., after filtering/merging)
        if not already_sorted:
            samples = sorted(
                samples,
                key=lambda s: s.monotonic_time if s.monotonic_time > 0 else s.timestamp.timestamp(),
            )

        def _duration_seconds(s1: CollectorSample, s2: CollectorSample) -> float:
            """Compute duration between two samples using monotonic clock when available."""
            if (
                s1.monotonic_time > 0.0
                and s2.monotonic_time > 0.0
                and s2.monotonic_time >= s1.monotonic_time
            ):
                return s2.monotonic_time - s1.monotonic_time
            return (s2.timestamp - s1.timestamp).total_seconds()

        first_sample = samples[0]
        last_sample = samples[-1]
        duration = _duration_seconds(first_sample, last_sample)

        # Log sample count for adequacy assessment
        if len(samples) < 5:
            logger.debug(f"Low sample count ({len(samples)}) - metrics may have higher variance")

        # Filter samples with valid memory only for memory metrics
        memory_samples = [s for s in samples if s.memory_usage_bytes > 0]
        if memory_samples:
            memory_values = [s.memory_usage_bytes / (1024 * 1024) for s in memory_samples]
            peak_memory_mb = max(memory_values)
            avg_memory_mb = sum(memory_values) / len(memory_values)
        else:
            peak_memory_mb = 0.0
            avg_memory_mb = 0.0

        # CPU metrics - use all samples for accurate duration
        # Clamp deltas to non-negative to handle counter edge cases (reset, wraparound)
        cpu_time_delta_ns = max(0, last_sample.cpu_time_ns - first_sample.cpu_time_ns)
        cpu_time_total_seconds = cpu_time_delta_ns / 1e9

        # Calculate average CPU percent
        avg_cpu_percent = (cpu_time_total_seconds / duration * 100) if duration > 0 else 0.0

        # Calculate peak CPU percent by checking each interval
        # NOTE: We ignore very short intervals (<< configured sampling interval) because
        # they can occur around start/stop and produce exaggerated peak estimates.
        min_interval_seconds = max(0.01, self._interval_seconds * 0.5)
        peak_cpu_percent = 0.0
        for i in range(1, len(samples)):
            s1 = samples[i - 1]
            s2 = samples[i]

            interval_ns = s2.cpu_time_ns - s1.cpu_time_ns
            interval_duration = _duration_seconds(s1, s2)

            if interval_duration >= min_interval_seconds:
                interval_cpu = (interval_ns / 1e9) / interval_duration * 100
                peak_cpu_percent = max(peak_cpu_percent, interval_cpu)

        # If the window is too short or filtered such that no interval met the threshold,
        # fall back to average CPU as a conservative peak estimate.
        if peak_cpu_percent == 0.0 and duration > 0:
            peak_cpu_percent = avg_cpu_percent

        # Calculate I/O metrics (use all samples for accurate deltas)
        # Clamp deltas to non-negative to handle counter edge cases (reset, wraparound)
        read_bytes_delta = max(0, last_sample.blkio_read_bytes - first_sample.blkio_read_bytes)
        write_bytes_delta = max(0, last_sample.blkio_write_bytes - first_sample.blkio_write_bytes)
        read_ops_delta = max(0, last_sample.blkio_read_ops - first_sample.blkio_read_ops)
        write_ops_delta = max(0, last_sample.blkio_write_ops - first_sample.blkio_write_ops)
        avg_read_iops = (read_ops_delta / duration) if duration > 0 else 0.0
        avg_write_iops = (write_ops_delta / duration) if duration > 0 else 0.0

        return CollectorResult(
            cpu_time_total_seconds=cpu_time_total_seconds,
            avg_cpu_percent=avg_cpu_percent,
            peak_cpu_percent=peak_cpu_percent,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            total_read_bytes=read_bytes_delta,
            total_write_bytes=write_bytes_delta,
            total_read_ops=read_ops_delta,
            total_write_ops=write_ops_delta,
            avg_read_iops=avg_read_iops,
            avg_write_iops=avg_write_iops,
            duration_seconds=duration,
            sample_count=len(samples),
        )
