"""Base collector abstract class for metrics collection.

All collectors implement this interface to enable modular, swappable
metrics collection strategies (Docker stats, cgroups v2, eBPF, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CollectorSample:
    """A single sample from any collector.

    Unified sample format that all collectors produce.
    """

    timestamp: datetime
    # Memory (bytes)
    memory_usage_bytes: int = 0
    # CPU
    cpu_percent: float = 0.0
    cpu_time_ns: int = 0  # Nanoseconds of CPU time
    # Block I/O (from cgroups v2 io.stat)
    blkio_read_bytes: int = 0
    blkio_write_bytes: int = 0
    blkio_read_ops: int = 0  # rios from cgroups
    blkio_write_ops: int = 0  # wios from cgroups


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
    # Meta
    duration_seconds: float = 0.0
    sample_count: int = 0
    samples: list[CollectorSample] | None = None


class BaseCollector(ABC):
    """Abstract base class for metrics collectors.

    Implementations:
    - DockerStatsCollector: Uses Docker stats API (existing, to be refactored)
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
