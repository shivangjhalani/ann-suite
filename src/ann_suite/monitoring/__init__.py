"""Monitoring module - Resource monitoring for Docker containers.

Provides multiple collector implementations:
- CgroupsV2Collector: Direct cgroups v2 access

Shared utilities:
- io_utils: Common I/O metric calculation helpers
"""

from __future__ import annotations

from ann_suite.core.constants import STANDARD_PAGE_SIZE
from ann_suite.monitoring.base import (
    BaseCollector,
    CollectorResult,
    CollectorSample,
    DeviceIOStat,
    TopDeviceSummary,
    get_system_block_size,
)
from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector
from ann_suite.monitoring.io_utils import (
    bytes_to_mb,
    bytes_to_pages,
    compute_avg_bytes_per_op,
    compute_avg_service_time_ms,
    compute_io_stall_percent,
    compute_iops,
    compute_throughput_mbps,
)

__all__ = [
    "BaseCollector",
    "CgroupsV2Collector",
    "CollectorResult",
    "CollectorSample",
    "DeviceIOStat",
    "STANDARD_PAGE_SIZE",
    "TopDeviceSummary",
    "bytes_to_mb",
    "bytes_to_pages",
    "compute_avg_bytes_per_op",
    "compute_avg_service_time_ms",
    "compute_io_stall_percent",
    "compute_iops",
    "compute_throughput_mbps",
    "get_system_block_size",
]
