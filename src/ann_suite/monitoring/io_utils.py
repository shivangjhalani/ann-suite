"""Shared I/O utilities for metrics aggregation.

This module provides common utilities for disk I/O metric calculations
used by both the cgroups collector (raw sample aggregation) and the
evaluator (benchmark-level derivations).

Constants:
    STANDARD_PAGE_SIZE: 4KB page size for cross-system comparability

Functions:
    bytes_to_mb: Convert bytes to megabytes
    bytes_to_pages: Convert bytes to standardized 4KB pages
    compute_iops: Calculate IOPS from operations delta and duration
    compute_throughput_mbps: Calculate throughput in MB/s
    compute_io_stall_percent: Calculate I/O stall percentage from PSI data
    compute_avg_service_time_ms: Calculate average service time in milliseconds
"""

from __future__ import annotations

from ann_suite.core.constants import STANDARD_PAGE_SIZE


def bytes_to_mb(byte_count: int | float) -> float:
    """Convert bytes to megabytes.

    Args:
        byte_count: Number of bytes

    Returns:
        Megabytes (float)
    """
    return byte_count / (1024 * 1024)


def bytes_to_pages(byte_count: int | float, page_size: int = STANDARD_PAGE_SIZE) -> int:
    """Convert bytes to page count using standardized page size.

    Args:
        byte_count: Number of bytes
        page_size: Page size in bytes (default: 4096)

    Returns:
        Number of pages (integer)
    """
    return int(byte_count / page_size)


def compute_iops(ops_delta: int, duration_seconds: float) -> float:
    """Compute IOPS from operations delta and duration.

    Args:
        ops_delta: Number of I/O operations in the window
        duration_seconds: Duration of the window in seconds

    Returns:
        IOPS (operations per second), 0.0 if duration is zero
    """
    if duration_seconds <= 0:
        return 0.0
    return ops_delta / duration_seconds


def compute_throughput_mbps(bytes_delta: int | float, duration_seconds: float) -> float:
    """Compute throughput in MB/s from bytes delta and duration.

    Args:
        bytes_delta: Number of bytes transferred in the window
        duration_seconds: Duration of the window in seconds

    Returns:
        Throughput in MB/s, 0.0 if duration is zero
    """
    if duration_seconds <= 0:
        return 0.0
    return bytes_to_mb(bytes_delta) / duration_seconds


def compute_io_stall_percent(psi_total_usec: int, duration_seconds: float) -> float | None:
    """Compute I/O stall percentage from PSI pressure data.

    Converts PSI microseconds to a percentage of wall-clock time.

    Args:
        psi_total_usec: PSI total stall time in microseconds
        duration_seconds: Duration of the window in seconds

    Returns:
        Stall percentage (0-100+), None if no valid data
    """
    if duration_seconds <= 0 or psi_total_usec <= 0:
        return None
    return (psi_total_usec / (duration_seconds * 1_000_000)) * 100.0


def compute_avg_service_time_ms(total_usec: int, total_ops: int) -> float | None:
    """Compute average I/O service time in milliseconds.

    Args:
        total_usec: Total service time in microseconds
        total_ops: Total number of I/O operations

    Returns:
        Average service time in milliseconds, None if no operations
    """
    if total_ops <= 0:
        return None
    if total_usec <= 0:
        return None
    return (total_usec / total_ops) / 1000.0


def compute_avg_bytes_per_op(total_bytes: int | float, total_ops: int) -> float | None:
    """Compute average bytes per I/O operation.

    Useful as a proxy for I/O efficiency (larger reads = fewer IOPS needed).

    Args:
        total_bytes: Total bytes transferred
        total_ops: Total number of I/O operations

    Returns:
        Average bytes per operation, None if no operations
    """
    if total_ops <= 0:
        return None
    return total_bytes / total_ops
