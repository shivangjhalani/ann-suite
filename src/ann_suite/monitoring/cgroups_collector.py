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
import math
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ann_suite.monitoring.base import (
    BaseCollector,
    CollectorResult,
    CollectorSample,
    DeviceIOStat,
    FilteredSamplesMeta,
    TopDeviceSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample filtering predicates
#
# Explicit predicates for filtering samples during aggregation. Each predicate
# returns True if the sample is VALID for that metric type. Filtering is minimal:
# only invalid/missing timestamps, zero/uninitialized counters at boundaries,
# and negative deltas are filtered.
# ---------------------------------------------------------------------------


def is_valid_cpu_sample(sample: CollectorSample) -> bool:
    """Check if sample has valid CPU data.

    Filters out samples where cpu_time_ns is zero, which indicates:
    - Cgroup was destroyed before the sample was taken
    - CPU controller not enabled
    - Uninitialized counter at collection boundary
    """
    return sample.cpu_time_ns > 0


def is_valid_memory_sample(sample: CollectorSample) -> bool:
    """Check if sample has valid memory data.

    Filters out samples where memory_usage_bytes is zero, which indicates:
    - Cgroup was destroyed before the sample was taken
    - Uninitialized counter at collection boundary
    """
    return sample.memory_usage_bytes > 0


def is_valid_io_sample(sample: CollectorSample) -> bool:
    """Check if sample has valid I/O data.

    Filters out samples where all I/O counters are zero, which indicates:
    - Cgroup was destroyed before the sample was taken
    - No I/O activity recorded yet (uninitialized at boundary)

    Note: A sample with any non-zero I/O field is considered valid, as
    some workloads may only read or only write.
    """
    return (
        sample.blkio_read_bytes > 0
        or sample.blkio_write_bytes > 0
        or sample.blkio_read_ops > 0
        or sample.blkio_write_ops > 0
    )


def filter_samples(
    samples: list[CollectorSample],
    predicate: Callable[[CollectorSample], bool],
) -> tuple[list[CollectorSample], int]:
    """Filter samples using a predicate and return valid samples with count of filtered.

    Args:
        samples: List of samples to filter
        predicate: Function returning True for valid samples

    Returns:
        Tuple of (valid_samples, filtered_count)
    """
    valid = [s for s in samples if predicate(s)]
    return valid, len(samples) - len(valid)


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

        # Fail fast if CPU controller is not enabled (Issue #1)
        self._validate_cpu_controller_enabled()

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
        # NOTE: We check if cgroup path still exists because Docker may have
        # already cleaned up the cgroup when the container exited.
        if self._cgroup_path is not None and self._cgroup_path.exists():
            final_sample = self._collect_sample()
            if final_sample is not None:
                with self._lock:
                    self._samples.append(final_sample)
                logger.debug("Collected final cgroups sample")
        elif self._cgroup_path is not None:
            logger.debug(
                f"Skipping final sample - cgroup path no longer exists: {self._cgroup_path}"
            )

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

    def _validate_cpu_controller_enabled(self) -> None:
        """Validate that the CPU controller is enabled for this container's cgroup.

        The cgroup v2 CPU controller must be enabled in the parent's subtree_control
        for cpu.stat to report accurate usage_usec values. Without it, CPU time
        metrics will always be zero.

        Raises:
            RuntimeError: If CPU controller is not enabled with remediation guidance.
        """
        if self._cgroup_path is None:
            return

        # Check the container's cgroup parent for subtree_control
        # Docker places containers in e.g., /sys/fs/cgroup/system.slice/docker-<id>.scope
        # The parent (system.slice) must have 'cpu' in its cgroup.subtree_control
        parent_path = self._cgroup_path.parent
        subtree_control_path = parent_path / "cgroup.subtree_control"

        if not subtree_control_path.exists():
            # Also check at the container level (less common but possible)
            subtree_control_path = self._cgroup_path / "cgroup.subtree_control"
            if not subtree_control_path.exists():
                raise RuntimeError(
                    "cgroup.subtree_control not found for container cgroup hierarchy. "
                    "Cannot validate CPU controller. Ensure cgroups v2 is properly mounted and "
                    "the container runtime uses a delegated cgroup slice with CPU enabled."
                )

        try:
            subtree_control = subtree_control_path.read_text().strip()
            controllers = subtree_control.split()
            logger.debug(f"cgroup.subtree_control at {subtree_control_path.parent}: {controllers}")

            if "cpu" not in controllers:
                # CPU controller not enabled - fail fast with clear remediation
                container_short_id = self._container_id[:12] if self._container_id else "unknown"
                raise RuntimeError(
                    f"CPU controller not enabled for container {container_short_id}. "
                    f"The cgroup at '{parent_path}' has subtree_control='{subtree_control}' "
                    "which does not include 'cpu'. CPU time metrics will be unreliable.\n\n"
                    "Remediation:\n"
                    "1. Enable the CPU controller in the parent cgroup:\n"
                    f"   echo '+cpu' | sudo tee {subtree_control_path}\n"
                    "2. Or configure Docker/systemd to use a cgroup slice with CPU enabled.\n"
                    "3. For systemd-managed Docker, ensure docker.service has:\n"
                    "   Delegate=cpu cpuset io memory pids\n"
                    "4. See docs/METRICS.md for detailed setup instructions."
                )

        except PermissionError as e:
            raise RuntimeError(
                f"Permission denied reading {subtree_control_path}: {e}. "
                "Cannot validate CPU controller for container metrics."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Error reading {subtree_control_path}: {e}. "
                "Cannot validate CPU controller for container metrics."
            ) from e

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

        # Read CPU time (in microseconds) and throttling
        cpu_stat = self._read_cpu_stat()
        cpu_time_ns = cpu_stat.get("usage_usec", 0) * 1000  # Convert to nanoseconds

        # Read I/O stats (aggregated + per-device)
        io_stat, per_device_io = self._read_io_stat()

        # Read I/O pressure (PSI)
        io_pressure = self._read_io_pressure()

        # Read memory.stat
        memory_stat = self._read_memory_stat()

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
            blkio_read_usec=io_stat.get("rusec", 0),
            blkio_write_usec=io_stat.get("wusec", 0),
            per_device_io=per_device_io if per_device_io else None,
            io_pressure_some_total_usec=io_pressure.get("some_total", 0),
            io_pressure_full_total_usec=io_pressure.get("full_total", 0),
            pgmajfault=memory_stat.get("pgmajfault", 0),
            pgfault=memory_stat.get("pgfault", 0),
            file_bytes=memory_stat.get("file", 0),
            file_mapped_bytes=memory_stat.get("file_mapped", 0),
            active_file_bytes=memory_stat.get("active_file", 0),
            inactive_file_bytes=memory_stat.get("inactive_file", 0),
            nr_throttled=cpu_stat.get("nr_throttled", 0),
            throttled_usec=cpu_stat.get("throttled_usec", 0),
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
        container_label = self._container_id[:12] if self._container_id else "unknown"

        try:
            content = cpu_stat_path.read_text()
            logger.debug(f"cpu.stat content for {container_label}: {content.strip()!r}")
            for line in content.strip().split("\n"):
                parts = line.split()
                if len(parts) == 2:
                    result[parts[0]] = int(parts[1])
            logger.debug(f"Parsed cpu.stat for {container_label}: {result}")
        except FileNotFoundError:
            # cpu.stat missing is unusual after passing CPU controller validation.
            # This may indicate the cgroup was destroyed mid-collection.
            # If we have collected samples, this is likely just the container exiting race.
            if len(self._samples) > 0:
                logger.debug(
                    f"cpu.stat not found for container {container_label} (final sample). "
                    "Container likely exited normally."
                )
            else:
                logger.warning(
                    f"cpu.stat not found for container {container_label} at {cpu_stat_path}. "
                    "The cgroup may have been destroyed (container exited) before any samples were collected."
                )
        except PermissionError:
            logger.warning(f"Permission denied reading cpu.stat for container {container_label}")

        return result

    def _read_io_stat(self) -> tuple[dict[str, int], list[DeviceIOStat]]:
        """Read io.stat file.

        Format (per device):
            8:0 rbytes=12345 wbytes=67890 rios=100 wios=50 rusec=1000 wusec=2000 ...

        Returns:
            Tuple of (aggregated stats dict, list of per-device stats)
        """
        result: dict[str, int] = {
            "rbytes": 0,
            "wbytes": 0,
            "rios": 0,
            "wios": 0,
            "rusec": 0,
            "wusec": 0,
        }
        per_device: list[DeviceIOStat] = []
        io_stat_path = self._cgroup_path / "io.stat"  # type: ignore

        try:
            content = io_stat_path.read_text()
            for line in content.strip().split("\n"):
                if not line:
                    continue
                # First token is device (e.g., "8:0")
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                device = parts[0]
                rest = parts[1]

                # Parse key=value pairs for this device
                device_stat = DeviceIOStat(device=device)
                for match in re.finditer(r"(rbytes|wbytes|rios|wios|rusec|wusec)=(\d+)", rest):
                    key, value = match.groups()
                    val = int(value)
                    result[key] += val
                    setattr(device_stat, key, val)
                per_device.append(device_stat)
        except (FileNotFoundError, PermissionError):
            pass

        return result, per_device

    def _read_io_pressure(self) -> dict[str, int]:
        """Read io.pressure file for PSI (Pressure Stall Information).

        Format:
            some avg10=0.00 avg60=0.00 avg300=0.00 total=12345
            full avg10=0.00 avg60=0.00 avg300=0.00 total=67890

        Returns dict with some_total and full_total in microseconds.
        """
        result: dict[str, int] = {"some_total": 0, "full_total": 0}
        io_pressure_path = self._cgroup_path / "io.pressure"  # type: ignore

        try:
            content = io_pressure_path.read_text()
            for line in content.strip().split("\n"):
                if not line:
                    continue
                if line.startswith("some "):
                    match = re.search(r"total=(\d+)", line)
                    if match:
                        result["some_total"] = int(match.group(1))
                elif line.startswith("full "):
                    match = re.search(r"total=(\d+)", line)
                    if match:
                        result["full_total"] = int(match.group(1))
        except (FileNotFoundError, PermissionError):
            pass

        return result

    def _read_memory_stat(self) -> dict[str, int]:
        """Read memory.stat file.

        Format:
            anon 12345
            file 67890
            pgfault 1000
            pgmajfault 10
            file_mapped 5000
            active_file 3000
            inactive_file 4000
            ...

        Returns dict with requested fields.
        """
        result: dict[str, int] = {
            "pgfault": 0,
            "pgmajfault": 0,
            "file": 0,
            "file_mapped": 0,
            "active_file": 0,
            "inactive_file": 0,
        }
        memory_stat_path = self._cgroup_path / "memory.stat"  # type: ignore

        try:
            content = memory_stat_path.read_text()
            for line in content.strip().split("\n"):
                parts = line.split()
                if len(parts) == 2 and parts[0] in result:
                    result[parts[0]] = int(parts[1])
        except (FileNotFoundError, PermissionError, ValueError):
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
            if (
                start_timestamp
                and s.timestamp <= start_timestamp
                and (before_sample is None or s.timestamp > before_sample.timestamp)
            ):
                before_sample = s
            if (
                end_timestamp
                and s.timestamp >= end_timestamp
                and (after_sample is None or s.timestamp < after_sample.timestamp)
            ):
                after_sample = s

        if before_sample is None and samples:
            before_sample = samples[0]
        if after_sample is None and samples:
            after_sample = samples[-1]

        if before_sample and after_sample and before_sample is not after_sample:
            return [before_sample, after_sample]
        return []

    def _compute_top_read_device(
        self, first_sample: CollectorSample, last_sample: CollectorSample
    ) -> TopDeviceSummary | None:
        """Compute the top device by read bytes delta between samples.

        Matches devices between first and last samples by device ID,
        computes delta, and returns the device with the highest read bytes delta.
        """
        if not first_sample.per_device_io or not last_sample.per_device_io:
            return None

        # Build lookup from first sample
        first_by_device: dict[str, DeviceIOStat] = {d.device: d for d in first_sample.per_device_io}

        # Compute deltas per device
        device_deltas: list[tuple[str, int, int, int, int]] = []
        for last_dev in last_sample.per_device_io:
            first_dev = first_by_device.get(last_dev.device)
            if first_dev:
                read_delta = max(0, last_dev.rbytes - first_dev.rbytes)
                write_delta = max(0, last_dev.wbytes - first_dev.wbytes)
                read_ops_delta = max(0, last_dev.rios - first_dev.rios)
                write_ops_delta = max(0, last_dev.wios - first_dev.wios)
            else:
                # New device appeared, treat all as delta
                read_delta = last_dev.rbytes
                write_delta = last_dev.wbytes
                read_ops_delta = last_dev.rios
                write_ops_delta = last_dev.wios
            device_deltas.append(
                (last_dev.device, read_delta, write_delta, read_ops_delta, write_ops_delta)
            )

        if not device_deltas:
            return None

        # Find top by read bytes
        top = max(device_deltas, key=lambda x: x[1])
        if top[1] == 0 and top[2] == 0:
            return None

        return TopDeviceSummary(
            device=top[0],
            total_read_bytes=top[1],
            total_write_bytes=top[2],
            total_read_ops=top[3],
            total_write_ops=top[4],
        )

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

        filtered_meta = FilteredSamplesMeta(total_samples=len(samples))

        # Filter samples with valid memory only for memory metrics
        memory_samples, filtered_memory = filter_samples(samples, is_valid_memory_sample)
        filtered_meta.memory_filtered_count = filtered_memory
        filtered_meta.memory_filter_reason = "memory_usage_bytes <= 0"
        if memory_samples:
            memory_values = [s.memory_usage_bytes / (1024 * 1024) for s in memory_samples]
            peak_memory_mb = max(memory_values)
            avg_memory_mb = sum(memory_values) / len(memory_values)
        else:
            peak_memory_mb = 0.0
            avg_memory_mb = 0.0

        # CPU metrics - use all samples for accurate duration
        # Clamp deltas to non-negative to handle counter edge cases (reset, wraparound)
        # Filter out samples with zero cpu_time_ns (can happen if cgroup was destroyed mid-collection)
        valid_cpu_samples, filtered_cpu = filter_samples(samples, is_valid_cpu_sample)
        filtered_meta.cpu_filtered_count = filtered_cpu
        filtered_meta.cpu_filter_reason = "cpu_time_ns <= 0"

        if len(valid_cpu_samples) >= 2:
            first_cpu_sample = valid_cpu_samples[0]
            last_cpu_sample = valid_cpu_samples[-1]
            cpu_time_delta_ns = max(0, last_cpu_sample.cpu_time_ns - first_cpu_sample.cpu_time_ns)
        else:
            # Fall back to using all samples if we don't have enough valid ones
            cpu_time_delta_ns = max(0, last_sample.cpu_time_ns - first_sample.cpu_time_ns)

        cpu_time_total_seconds = cpu_time_delta_ns / 1e9

        # Log warning if CPU time is zero but we have samples (indicates cgroup issue)
        if cpu_time_total_seconds == 0.0 and len(samples) > 1:
            # Check if cgroup path still exists to diagnose container cleanup timing
            cgroup_still_exists = self._cgroup_path.exists() if self._cgroup_path else False
            logger.warning(
                f"CPU time is zero despite {len(samples)} samples over {duration:.2f}s. "
                f"First cpu_time_ns={first_sample.cpu_time_ns}, "
                f"Last cpu_time_ns={last_sample.cpu_time_ns}. "
                f"Valid CPU samples: {len(valid_cpu_samples)}. "
                f"Cgroup path exists: {cgroup_still_exists}. "
                "The container was likely destroyed before the final sample was collected."
            )

        # Calculate average CPU percent
        avg_cpu_percent = (cpu_time_total_seconds / duration * 100) if duration > 0 else 0.0

        # Calculate peak CPU percent by checking each interval
        # NOTE: We ignore very short intervals (<< configured sampling interval) because
        # they can occur around start/stop and produce exaggerated peak estimates.
        # Also skip intervals where cpu_time_ns is 0 (cgroup was destroyed)
        min_interval_seconds = max(0.01, self._interval_seconds * 0.5)
        peak_cpu_percent = 0.0
        for i in range(1, len(samples)):
            s1 = samples[i - 1]
            s2 = samples[i]

            # Skip intervals where either sample has zero CPU time (cgroup destroyed)
            if s1.cpu_time_ns == 0 or s2.cpu_time_ns == 0:
                continue

            interval_ns = s2.cpu_time_ns - s1.cpu_time_ns
            interval_duration = _duration_seconds(s1, s2)

            if interval_duration >= min_interval_seconds and interval_ns > 0:
                interval_cpu = (interval_ns / 1e9) / interval_duration * 100
                peak_cpu_percent = max(peak_cpu_percent, interval_cpu)

        # If the window is too short or filtered such that no interval met the threshold,
        # fall back to average CPU as a conservative peak estimate.
        if peak_cpu_percent == 0.0 and duration > 0:
            peak_cpu_percent = avg_cpu_percent

        # Calculate I/O metrics (use valid samples with non-zero I/O data)
        # Filter out samples that likely have incomplete data (cgroup destroyed)
        valid_io_samples, filtered_io = filter_samples(samples, is_valid_io_sample)
        filtered_meta.io_filtered_count = filtered_io
        filtered_meta.io_filter_reason = "blkio counters all zero"

        if len(valid_io_samples) >= 2:
            first_io_sample = valid_io_samples[0]
            last_io_sample = valid_io_samples[-1]
        else:
            # Fall back to all samples if no valid I/O samples found
            first_io_sample = first_sample
            last_io_sample = last_sample

        # Clamp deltas to non-negative to handle counter edge cases (reset, wraparound)
        read_bytes_delta = max(
            0, last_io_sample.blkio_read_bytes - first_io_sample.blkio_read_bytes
        )
        write_bytes_delta = max(
            0, last_io_sample.blkio_write_bytes - first_io_sample.blkio_write_bytes
        )
        read_ops_delta = max(0, last_io_sample.blkio_read_ops - first_io_sample.blkio_read_ops)
        write_ops_delta = max(0, last_io_sample.blkio_write_ops - first_io_sample.blkio_write_ops)
        avg_read_iops = (read_ops_delta / duration) if duration > 0 else 0.0
        avg_write_iops = (write_ops_delta / duration) if duration > 0 else 0.0

        # I/O latency deltas (rusec/wusec)
        read_usec_delta = max(0, last_io_sample.blkio_read_usec - first_io_sample.blkio_read_usec)
        write_usec_delta = max(
            0, last_io_sample.blkio_write_usec - first_io_sample.blkio_write_usec
        )

        # I/O pressure deltas (PSI totals)
        io_pressure_some_delta = max(
            0,
            last_sample.io_pressure_some_total_usec - first_sample.io_pressure_some_total_usec,
        )
        io_pressure_full_delta = max(
            0,
            last_sample.io_pressure_full_total_usec - first_sample.io_pressure_full_total_usec,
        )

        # Memory stat deltas (page faults are counters, file stats are gauges)
        pgmajfault_delta = max(0, last_sample.pgmajfault - first_sample.pgmajfault)
        pgfault_delta = max(0, last_sample.pgfault - first_sample.pgfault)

        # File cache stats (gauges)
        file_bytes_values = [s.file_bytes for s in samples]
        file_mapped_values = [s.file_mapped_bytes for s in samples]
        active_file_values = [s.active_file_bytes for s in samples]
        inactive_file_values = [s.inactive_file_bytes for s in samples]

        avg_file_bytes = sum(file_bytes_values) / len(file_bytes_values) if file_bytes_values else 0
        avg_file_mapped_bytes = (
            sum(file_mapped_values) / len(file_mapped_values) if file_mapped_values else 0
        )
        avg_active_file_bytes = (
            sum(active_file_values) / len(active_file_values) if active_file_values else 0
        )
        avg_inactive_file_bytes = (
            sum(inactive_file_values) / len(inactive_file_values) if inactive_file_values else 0
        )

        peak_file_bytes = max(file_bytes_values) if file_bytes_values else 0
        peak_file_mapped_bytes = max(file_mapped_values) if file_mapped_values else 0
        peak_active_file_bytes = max(active_file_values) if active_file_values else 0
        peak_inactive_file_bytes = max(inactive_file_values) if inactive_file_values else 0

        # CPU throttling deltas
        nr_throttled_delta = max(0, last_sample.nr_throttled - first_sample.nr_throttled)
        throttled_usec_delta = max(0, last_sample.throttled_usec - first_sample.throttled_usec)

        # Per-device I/O: compute top device by read bytes delta
        top_read_device = self._compute_top_read_device(first_sample, last_sample)

        def _percentile(values: list[float], percentile: float) -> float | None:
            if not values:
                return None
            values_sorted = sorted(values)
            rank = math.ceil((percentile / 100) * len(values_sorted)) - 1
            index = max(0, min(rank, len(values_sorted) - 1))
            return values_sorted[index]

        interval_read_iops: list[float] = []
        interval_read_mbps: list[float] = []
        interval_read_service_time_ms: list[float] = []

        for i in range(1, len(samples)):
            s1 = samples[i - 1]
            s2 = samples[i]
            interval_duration = _duration_seconds(s1, s2)
            if interval_duration < min_interval_seconds:
                continue

            read_ops_delta = max(0, s2.blkio_read_ops - s1.blkio_read_ops)
            read_bytes_delta = max(0, s2.blkio_read_bytes - s1.blkio_read_bytes)
            read_usec_delta = max(0, s2.blkio_read_usec - s1.blkio_read_usec)

            interval_read_iops.append(read_ops_delta / interval_duration)
            interval_read_mbps.append((read_bytes_delta / (1024 * 1024)) / interval_duration)

            if read_ops_delta > 0 and read_usec_delta > 0:
                interval_read_service_time_ms.append((read_usec_delta / read_ops_delta) / 1000.0)

        p95_read_iops = _percentile(interval_read_iops, 95)
        max_read_iops = max(interval_read_iops) if interval_read_iops else None
        p95_read_mbps = _percentile(interval_read_mbps, 95)
        max_read_mbps = max(interval_read_mbps) if interval_read_mbps else None
        p95_read_service_time_ms = _percentile(interval_read_service_time_ms, 95)
        max_read_service_time_ms = (
            max(interval_read_service_time_ms) if interval_read_service_time_ms else None
        )

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
            total_read_usec=read_usec_delta,
            total_write_usec=write_usec_delta,
            io_pressure_some_total_usec=io_pressure_some_delta,
            io_pressure_full_total_usec=io_pressure_full_delta,
            pgmajfault_delta=pgmajfault_delta,
            pgfault_delta=pgfault_delta,
            avg_file_bytes=avg_file_bytes,
            peak_file_bytes=peak_file_bytes,
            avg_file_mapped_bytes=avg_file_mapped_bytes,
            peak_file_mapped_bytes=peak_file_mapped_bytes,
            avg_active_file_bytes=avg_active_file_bytes,
            peak_active_file_bytes=peak_active_file_bytes,
            avg_inactive_file_bytes=avg_inactive_file_bytes,
            peak_inactive_file_bytes=peak_inactive_file_bytes,
            nr_throttled_delta=nr_throttled_delta,
            throttled_usec_delta=throttled_usec_delta,
            top_read_device=top_read_device,
            p95_read_iops=p95_read_iops,
            max_read_iops=max_read_iops,
            p95_read_mbps=p95_read_mbps,
            max_read_mbps=max_read_mbps,
            p95_read_service_time_ms=p95_read_service_time_ms,
            max_read_service_time_ms=max_read_service_time_ms,
            duration_seconds=duration,
            sample_count=len(samples),
            filtered_samples_meta=filtered_meta,
        )
