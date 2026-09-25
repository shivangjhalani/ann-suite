"""Base collector abstract class for metrics collection.

The suite collects container metrics directly from cgroups v2.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeviceIOStat:
    """Per-device I/O statistics from cgroups v2 io.stat."""

    device: str  # Device identifier (e.g., "8:0")
    rbytes: int = 0
    wbytes: int = 0
    rios: int = 0
    wios: int = 0
    rusec: int = 0  # Read latency microseconds (if available)
    wusec: int = 0  # Write latency microseconds (if available)


@dataclass
class CollectorSample:
    """A single sample from any collector.

    Unified sample format that all collectors produce.
    """

    timestamp: datetime
    # Monotonic timestamp (seconds) for accurate duration/interval calculation.
    # This avoids wall-clock adjustments (e.g., NTP) affecting rate metrics.
    # When unavailable (e.g., tests constructing samples manually), leave as 0.0 and
    # aggregation will fall back to `timestamp`.
    monotonic_time: float = 0.0
    # Memory (bytes)
    memory_usage_bytes: int = 0
    # CPU
    cpu_percent: float = 0.0
    cpu_time_ns: int = 0  # Nanoseconds of CPU time
    # Block I/O (from cgroups v2 io.stat) - aggregated across devices
    blkio_read_bytes: int = 0
    blkio_write_bytes: int = 0
    blkio_read_ops: int = 0  # rios from cgroups
    blkio_write_ops: int = 0  # wios from cgroups
    # I/O latency (from io.stat rusec/wusec if present)
    blkio_read_usec: int = 0
    blkio_write_usec: int = 0
    # Per-device I/O stats (optional detailed breakdown)
    per_device_io: list[DeviceIOStat] | None = None
    # I/O pressure (PSI from io.pressure)
    io_pressure_some_total_usec: int = 0
    io_pressure_full_total_usec: int = 0
    # Memory stats (from memory.stat)
    pgmajfault: int = 0
    pgfault: int = 0
    file_bytes: int = 0  # Page cache file bytes
    file_mapped_bytes: int = 0
    active_file_bytes: int = 0
    inactive_file_bytes: int = 0
    # CPU throttling (from cpu.stat)
    nr_throttled: int = 0
    throttled_usec: int = 0
    # Queue depth (in-flight I/O ops summed over physical block devices, from /sys/block)
    queue_depth: int = 0
    # Device-level read counters (from /sys/block/<dev>/stat), summed across physical
    # devices. System-wide (not cgroup-scoped), but a reliable fallback for read
    # service time when the cgroup's io.stat has no rusec/wusec (see
    # read_device_io_totals()).
    device_reads_completed: int = 0
    device_read_ticks_ms: int = 0
    # System-wide CPU busy/total jiffies (from /proc/stat), for machine-level
    # utilization during the search window (distinct from the per-cgroup cpu_percent).
    machine_cpu_busy_ticks: int = 0
    machine_cpu_total_ticks: int = 0


@dataclass
class TopDeviceSummary:
    """Summary of the top I/O device by read bytes."""

    device: str
    total_read_bytes: int = 0
    total_write_bytes: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0

    def to_dict(self) -> dict[str, int | str]:
        """Convert to dictionary for ResourceSummary serialization."""
        return {
            "device": self.device,
            "total_read_bytes": self.total_read_bytes,
            "total_write_bytes": self.total_write_bytes,
            "total_read_ops": self.total_read_ops,
            "total_write_ops": self.total_write_ops,
        }


@dataclass
class FilteredSamplesMeta:
    """Metadata about samples filtered during aggregation.

    Records counts and reasons for filtered samples to avoid silent bias
    and provide transparency into data quality.
    """

    # Total samples before any filtering
    total_samples: int = 0

    # CPU filtering: zero/uninitialized counters at boundaries
    cpu_filtered_count: int = 0
    cpu_filter_reason: str = ""

    # Memory filtering: zero/uninitialized values
    memory_filtered_count: int = 0
    memory_filter_reason: str = ""

    # I/O filtering: zero/uninitialized counters at boundaries
    io_filtered_count: int = 0
    io_filter_reason: str = ""

    def to_dict(self) -> dict[str, int | str]:
        """Convert to dictionary for serialization."""
        return {
            "total_samples": self.total_samples,
            "cpu_filtered_count": self.cpu_filtered_count,
            "cpu_filter_reason": self.cpu_filter_reason,
            "memory_filtered_count": self.memory_filtered_count,
            "memory_filter_reason": self.memory_filter_reason,
            "io_filtered_count": self.io_filtered_count,
            "io_filter_reason": self.io_filter_reason,
        }


@dataclass
class CollectorResult:
    """Aggregated result from a collector run."""

    # CPU metrics
    cpu_time_total_seconds: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    # Disk I/O metrics (CRITICAL)
    total_read_bytes: int = 0
    total_write_bytes: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    avg_read_iops: float = 0.0
    avg_write_iops: float = 0.0
    # I/O latency totals (delta of rusec/wusec)
    total_read_usec: int = 0
    total_write_usec: int = 0
    # I/O pressure (PSI deltas)
    io_pressure_some_total_usec: int = 0
    io_pressure_full_total_usec: int = 0
    # Whether PSI counters were observed as non-zero in any sample in this window.
    # Distinguishes "measured zero stall" (True) from "PSI unavailable" (False).
    psi_available: bool = False
    # Memory stats deltas
    pgmajfault_delta: int = 0
    pgfault_delta: int = 0
    avg_file_bytes: float = 0.0
    peak_file_bytes: int = 0
    avg_file_mapped_bytes: float = 0.0
    peak_file_mapped_bytes: int = 0
    avg_active_file_bytes: float = 0.0
    peak_active_file_bytes: int = 0
    avg_inactive_file_bytes: float = 0.0
    peak_inactive_file_bytes: int = 0
    # CPU throttling deltas
    nr_throttled_delta: int = 0
    throttled_usec_delta: int = 0
    # Per-device summary (top device by read bytes)
    top_read_device: TopDeviceSummary | None = None
    # Tail metrics from per-interval deltas
    p95_read_iops: float | None = None
    max_read_iops: float | None = None
    p95_read_mbps: float | None = None
    max_read_mbps: float | None = None
    p95_read_service_time_ms: float | None = None
    max_read_service_time_ms: float | None = None
    # Queue depth (in-flight I/O ops, sampled from /sys/block)
    avg_queue_depth: float = 0.0
    max_queue_depth: int = 0
    p95_queue_depth: float | None = None
    # Device-level (system-wide) read IOPS and mean read service time, from
    # /sys/block/<dev>/stat deltas across the window. Fallback/complement for
    # avg_read_service_time_ms when the cgroup's io.stat rusec is unavailable.
    device_read_iops: float | None = None
    device_avg_read_service_time_ms: float | None = None
    # System-wide CPU utilization (0-1) during the window, from /proc/stat deltas.
    machine_cpu_util: float | None = None
    # Meta
    duration_seconds: float = 0.0
    sample_count: int = 0
    samples: list[CollectorSample] | None = None
    # Filtering metadata: tracks samples filtered during aggregation
    filtered_samples_meta: FilteredSamplesMeta | None = None


class BaseCollector(ABC):
    """Abstract base class for metrics collectors.

    Implementations:
    - CgroupsV2Collector: Direct cgroups v2 filesystem access
    """

    @abstractmethod
    def start(self, container_id: str) -> None:
        """Start collecting metrics for a container.

        Args:
            container_id: Docker container ID (short or full)
        """
        pass

    @abstractmethod
    def stop(self) -> CollectorResult:
        """Stop collecting and return aggregated metrics.

        Returns:
            CollectorResult with all collected metrics
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this collector can run on the current system.

        Returns:
            True if the collector's prerequisites are met
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this collector."""
        pass


def read_system_queue_depth() -> int:
    """Read the current system-wide block I/O queue depth.

    Sums in-flight read and write operations across all physical block devices
    via /sys/block/<dev>/inflight. This is a system-level gauge (not per-cgroup),
    so it is most accurate when the benchmark host is otherwise idle.

    Returns:
        Total in-flight I/O operations; 0 if unreadable or no activity.
    """
    sys_block = Path("/sys/block")
    if not sys_block.exists():
        return 0

    total = 0
    try:
        for device_dir in sys_block.iterdir():
            # Skip virtual/loop devices (same filter as get_system_block_size)
            if device_dir.name.startswith(("loop", "ram", "dm-", "sr", "fd")):
                continue

            inflight_path = device_dir / "inflight"
            if not inflight_path.exists():
                continue
            try:
                # Format: one line "<in-flight reads> <in-flight writes>"
                parts = inflight_path.read_text().split()
                total += sum(int(p) for p in parts)
            except (ValueError, PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        return 0

    return total


def read_device_io_totals() -> tuple[int, int]:
    """Read cumulative read completions and read ticks across physical block devices.

    Sources /sys/block/<dev>/stat, whose fields are (per Documentation/ABI/stable
    /sysfs-block-device, and the same fields the PipeANN reference open-loop driver
    reads): field 0 = reads completed, field 3 = milliseconds spent reading. These
    are monotonic counters maintained by the kernel block layer for every device,
    independent of cgroups, so they are available even when a cgroup's io.stat
    lacks rusec/wusec (not all kernels/controllers populate those fields, which is
    why `avg_read_service_time_ms` could read as 0/None despite real disk I/O).

    Returns:
        (reads_completed, read_ticks_ms) summed over all physical devices; (0, 0)
        if /sys/block is unreadable.
    """
    sys_block = Path("/sys/block")
    if not sys_block.exists():
        return 0, 0

    total_reads = 0
    total_read_ticks_ms = 0
    try:
        for device_dir in sys_block.iterdir():
            if device_dir.name.startswith(("loop", "ram", "dm-", "sr", "fd")):
                continue
            stat_path = device_dir / "stat"
            if not stat_path.exists():
                continue
            try:
                fields = stat_path.read_text().split()
                # fields[0] = reads completed successfully, fields[3] = ms spent reading
                if len(fields) >= 4:
                    total_reads += int(fields[0])
                    total_read_ticks_ms += int(fields[3])
            except (ValueError, PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        return 0, 0

    return total_reads, total_read_ticks_ms


def read_machine_cpu_ticks() -> tuple[int, int]:
    """Read system-wide CPU busy/total jiffies from /proc/stat's aggregate 'cpu' line.

    Mirrors the PipeANN reference open-loop driver's read_cpu(): total is the sum of
    the first 8 fields (user, nice, system, idle, iowait, irq, softirq, steal), busy
    is total minus idle and iowait. This is a machine-wide gauge, not scoped to the
    benchmarked container/cgroup, so it is most meaningful when the host is otherwise
    quiet (the same caveat that applies to read_system_queue_depth()).

    Returns:
        (busy_ticks, total_ticks); (0, 0) if /proc/stat is unreadable.
    """
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu" or len(parts) < 9:
            return 0, 0
        values = [int(x) for x in parts[1:9]]
        total = sum(values)
        idle_and_iowait = values[3] + values[4]
        busy = total - idle_and_iowait
        return busy, total
    except (ValueError, PermissionError, OSError, IndexError):
        return 0, 0


def get_system_block_size() -> int:
    """Detect the system block size from block device attributes.

    Checks /sys/block/*/queue/physical_block_size for common devices
    (sda, nvme0n1, etc.) and returns the detected block size.

    Returns:
        Block size in bytes (defaults to 4096 if detection fails)
    """
    default_block_size = 4096

    # Priority list of devices to check (skip loop, ram, dm-* devices)
    block_device_prefixes = ["nvme", "sd", "vd", "hd", "xvd"]

    sys_block = Path("/sys/block")
    if not sys_block.exists():
        logger.debug("No /sys/block directory found, using default block size")
        return default_block_size

    try:
        for device_dir in sys_block.iterdir():
            device_name = device_dir.name

            # Skip virtual/loop devices
            if device_name.startswith(("loop", "ram", "dm-", "sr", "fd")):
                continue

            # Check if it's a real block device we care about
            is_real_device = any(device_name.startswith(p) for p in block_device_prefixes)
            if not is_real_device:
                continue

            block_size_path = device_dir / "queue" / "physical_block_size"
            if block_size_path.exists():
                try:
                    block_size = int(block_size_path.read_text().strip())
                    if 512 <= block_size <= 65536:  # Sanity check
                        logger.debug(f"Detected block size {block_size} from {device_name}")
                        return block_size
                except (ValueError, PermissionError, OSError):
                    continue

    except (PermissionError, OSError) as e:
        logger.debug(f"Error detecting block size: {e}")

    logger.debug(f"Using default block size: {default_block_size}")
    return default_block_size
