"""Base collector abstract class for metrics collection.

All collectors implement this interface to enable modular, swappable
metrics collection strategies (Docker stats, cgroups v2, eBPF, etc.).
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
    - EBPFCollector: eBPF-based deep I/O tracing (future)
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
