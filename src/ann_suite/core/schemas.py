"""Pydantic schemas for ANN Benchmarking Suite.

This module defines all data contracts used throughout the benchmarking suite,
including algorithm configurations, dataset definitions, and result structures.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


def _flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = "_") -> dict[str, Any]:
    """Recursively flatten a nested dict into dot/sep-prefixed scalar keys.

    Nested mappings become prefixed keys (e.g. {"search": {"ef": 100}} ->
    {"search_ef": 100}). Lists and scalar values are kept as-is under their key so
    they remain available for inspection without being mangled.
    """
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key, sep=sep))
        else:
            flat[full_key] = value
    return flat


# =============================================================================
# Output TypedDicts for type-safe JSON serialization
# =============================================================================


class QualityMetricsDict(TypedDict):
    """Quality metrics output structure."""

    recall: float | None
    qps: float | None


class BuildPhaseDict(TypedDict):
    """Build phase metrics output structure."""

    duration_seconds: float | None
    index_size_bytes: int | None
    cpu_time_seconds: float
    peak_cpu_percent: float
    peak_rss_mb: float
    error: str | None


class WarmupPhaseDict(TypedDict):
    """Warmup phase metrics output structure."""

    duration_seconds: float | None
    cpu_time_seconds: float
    peak_cpu_percent: float
    peak_rss_mb: float
    read_mb: float
    write_mb: float
    io_stall_percent: float | None
    major_faults_per_second: float | None
    file_cache_avg_mb: float | None
    file_cache_peak_mb: float | None


class DiskIODict(TypedDict):
    """Disk I/O metrics output structure."""

    avg_read_iops: float
    avg_write_iops: float
    avg_read_throughput_mbps: float
    avg_write_throughput_mbps: float
    total_read_mb: float
    total_write_mb: float
    total_pages_read: int
    total_pages_written: int
    pages_per_query: float | None
    reads_per_query: float | None
    bytes_read_per_query: float | None
    # Queue depth (system-wide in-flight I/O, sampled per interval)
    avg_queue_depth: float | None
    max_queue_depth: int | None
    p95_queue_depth: float | None
    # Service time proxy / efficiency metrics
    avg_bytes_per_read_op: float | None
    avg_bytes_per_write_op: float | None
    avg_read_service_time_ms: float | None
    avg_write_service_time_ms: float | None
    # Tail metrics (p95/max IOPS)
    p95_read_iops: float | None
    max_read_iops: float | None
    p95_read_mbps: float | None
    max_read_mbps: float | None
    p95_read_service_time_ms: float | None
    max_read_service_time_ms: float | None
    # PSI stall metrics
    io_stall_percent: float | None
    io_full_stall_percent: float | None


class SearchPhaseDict(TypedDict):
    """Search phase metrics output structure."""

    duration_seconds: float | None
    cpu_time_seconds: float
    cpu_time_per_query_ms: float | None
    avg_cpu_percent: float
    peak_cpu_percent: float
    peak_rss_mb: float
    avg_rss_mb: float
    disk_io: DiskIODict
    major_faults_per_query: float | None
    major_faults_per_second: float | None
    page_cache_hit_rate: float | None
    file_cache_avg_mb: float | None
    file_cache_peak_mb: float | None
    cpu_nr_throttled: int
    cpu_throttled_usec: int
    error: str | None


class LatencyDict(TypedDict):
    """Latency metrics output structure."""

    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None


class MetadataDict(TypedDict):
    """Metadata output structure."""

    run_id: str
    physical_block_size: int
    sample_count: int
    query_start_timestamp: str | None
    query_end_timestamp: str | None


class DebugArtifactsDict(TypedDict):
    """Debug artifacts pointer for traceability without bloating results.json.

    Raw metric samples are stored in a separate JSONL file to enable:
    - Post-hoc verification of metric calculations
    - Diagnosis of anomalous readings
    - Custom time-window analysis on historical data

    The pointer includes path and sample count for quick reference.
    """

    samples_path: str | None
    samples_count: int


class BenchmarkSummaryDict(TypedDict):
    """Complete phase-structured benchmark result output.

    This is the primary JSON output format for results.json, organized by phase:
    - quality: Primary metrics (recall, QPS)
    - build: Index construction phase
    - warmup: Index loading/cache warming phase
    - search: Query execution phase (primary benchmark data)
    - latency: Query latency distribution
    - metadata: Run metadata and timestamps
    - debug_artifacts: Pointers to separate debug files (samples.jsonl)
    """

    algorithm: str
    dataset: str
    timestamp: str
    quality: QualityMetricsDict
    build: BuildPhaseDict
    warmup: WarmupPhaseDict
    search: SearchPhaseDict
    latency: LatencyDict
    algorithm_stats: dict[str, Any] | None
    metadata: MetadataDict
    debug_artifacts: DebugArtifactsDict
    hyperparameters: dict[str, Any]


class DistanceMetric(str, Enum):
    """Supported distance metrics for ANN algorithms."""

    L2 = "L2"  # Euclidean distance
    IP = "IP"  # Inner product (cosine similarity when normalized)
    COSINE = "cosine"  # Cosine similarity
    HAMMING = "hamming"  # Hamming distance for binary vectors


class AlgorithmType(str, Enum):
    """Algorithm storage type classification."""

    MEMORY = "memory"  # In-memory algorithms (HNSW, Annoy, etc.)
    DISK = "disk"  # Disk-based algorithms (DiskANN, SPANN, etc.)
    HYBRID = "hybrid"  # Hybrid algorithms with both memory and disk components


class BuildConfig(BaseModel):
    """Configuration for the index building phase."""

    timeout_seconds: int = Field(default=3600, ge=60, description="Build phase timeout")
    prebuilt_path: Path | None = Field(
        default=None,
        description=(
            "Optional existing index directory, relative to index_dir or an absolute path. "
            "When set, the evaluator validates and mounts this index and skips building."
        ),
    )
    args: dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific build args")
    reuse_index: bool = Field(
        default=True,
        description=(
            "Reuse a built index across search-phase sweep points within this run instead of "
            "rebuilding it for every point. Set false to rebuild for each benchmark point "
            "(legacy behavior; needed when build cost itself is under study)."
        ),
    )

    model_config = {"extra": "allow"}


class WarmupConfig(BaseModel):
    """Configuration for the warmup/cache-warming phase before benchmarking.

    This controls how the index is prepared before timed query execution:
    - collect_metrics: Whether to collect and report warmup phase metrics
    - cache_warmup_queries: Number of untimed queries to warm OS/algorithm caches
    - drop_caches_before: Drop the OS page cache before each search phase so index
      reads hit physical disk (cold-start benchmarking). Requires root or sudo;
      a password may be supplied via the ANN_SUITE_SUDO_PASSWORD environment
      variable. On failure the run continues with a warm cache and logs a warning.

    Example scenarios:
    - Cold start benchmark: drop_caches_before=True, cache_warmup_queries=0
    - Warm cache benchmark: cache_warmup_queries=1000
    - Default (realistic): both disabled
    """

    collect_metrics: bool = Field(
        default=True,
        description="Collect and report warmup phase metrics (index loading time/resources)",
    )
    cache_warmup_queries: int = Field(
        default=0,
        ge=0,
        description="Number of random queries to run before timed benchmark to warm caches",
    )
    drop_caches_before: bool = Field(
        default=False,
        description=(
            "Drop the OS page cache before each search phase (cold-start benchmarking). "
            "Requires root or sudo; set ANN_SUITE_SUDO_PASSWORD for passworded sudo. "
            "If the drop fails, the search runs with a warm cache and a warning is logged."
        ),
    )


class SearchConfig(BaseModel):
    """Configuration for the search/query phase."""

    timeout_seconds: int = Field(default=600, ge=10, description="Search phase timeout")
    k: int = Field(default=10, ge=1, le=1000, description="Number of neighbors to retrieve")
    query_rounds: int = Field(
        default=1,
        ge=1,
        description="Number of timed passes over the full query set",
    )
    args: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific search args (can be list for sweeps)"
    )
    sweep: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional explicit search parameter points. When set, each mapping is run once "
            "instead of expanding list-valued args as a Cartesian product."
        ),
    )
    batch_mode: bool = Field(
        default=False,
        description=(
            "False (default, research-oriented): run each query serially and time it "
            "individually, yielding real latency percentiles (p50/p95/p99). True: run the "
            "whole query batch in a single native call for maximum QPS, but only mean "
            "latency is available (percentiles are None/N/A)."
        ),
    )
    warmup: WarmupConfig = Field(
        default_factory=WarmupConfig, description="Warmup/cache-warming configuration"
    )

    model_config = {"extra": "allow"}


class AlgorithmConfig(BaseModel):
    """Complete algorithm configuration for benchmarking.

    Attributes:
        name: Human-readable algorithm identifier
        docker_image: Docker image tag (e.g., 'ann-suite/hnsw:latest')
        algorithm_type: Whether algorithm is memory, disk, or hybrid based
        build: Build phase configuration
        search: Search phase configuration
        disabled: If True, skip this algorithm during benchmarking
        env_vars: Environment variables to pass to the container
    """

    name: str = Field(..., min_length=1, description="Algorithm identifier")
    docker_image: str = Field(..., min_length=1, description="Docker image tag")
    algorithm_type: AlgorithmType = Field(default=AlgorithmType.MEMORY)
    build: BuildConfig = Field(default_factory=BuildConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    disabled: bool = Field(default=False)
    env_vars: dict[str, str] = Field(default_factory=dict)
    cpu_affinity: str | None = Field(
        default=None,
        description=(
            "CPU core affinity/cpuset (e.g., '0-3', '0,2,4,6'); pins the container to these "
            "cores (Docker cpuset_cpus). When set, the matching NUMA node(s) are also pinned "
            "(cpuset_mems) so CPU and memory placement are local, removing cross-NUMA traffic "
            "noise from benchmarks. Prefer pinning all cores of a single NUMA node for "
            "deterministic, low-noise measurements."
        ),
    )
    cpu_limit: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Hard CPU utilization cap in cores (e.g., 8.0); maps to Docker's nano_cpus "
            "(CFS CPU quota). ⚠️ CAVEAT: the CFS scheduler enforces this by throttling the "
            "container in ~100ms windows, which can inject periodic stalls that inflate "
            "p95/p99 latency for steady-state query workloads. For latency-sensitive ANN "
            "benchmarks prefer cpu_affinity (NUMA-pinned cores, no throttling) over a "
            "cpu_limit. If you must cap usage, combine affinity + limit and watch the "
            "CPUThrottlingMetrics (nr_throttled / throttled_percent) in results. This cap is "
            "independent of cpu_affinity: you may set cpu_affinity to pin placement (e.g., "
            "to a NUMA socket) AND cpu_limit to cap concurrent core usage together; a "
            "combination pins to the socket while also limiting aggregate throughput."
        ),
    )
    memory_limit: str | None = Field(
        default=None,
        description="Hard container memory limit (e.g., '8g'); swap is capped to the same value",
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Dataset names to run on. Empty = all datasets",
    )

    @field_validator("docker_image")
    @classmethod
    def validate_docker_image(cls, v: str) -> str:
        """Ensure docker image has valid format."""
        if ":" not in v:
            return f"{v}:latest"
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration for benchmarking.

    Attributes:
        name: Human-readable dataset identifier
        base_path: Path to base vectors (HDF5 or NumPy format)
        query_path: Path to query vectors (optional, defaults to base_path)
        ground_truth_path: Path to ground truth neighbors (optional)
        distance_metric: Distance metric to use
        dimension: Vector dimension (validated against actual data)
        point_type: Data type of vectors (float32, uint8, etc.)
    """

    name: str = Field(..., min_length=1)
    base_path: Path = Field(..., description="Path to base vectors")
    query_path: Path | None = Field(default=None, description="Path to query vectors")
    ground_truth_path: Path | None = Field(default=None, description="Path to ground truth")
    distance_metric: DistanceMetric = Field(default=DistanceMetric.L2)
    dimension: int = Field(..., ge=1, le=65536, description="Vector dimension")
    point_type: str = Field(default="float32")
    base_count: int | None = Field(default=None, description="Number of base vectors (for info)")
    query_count: int | None = Field(default=None, description="Number of query vectors (for info)")

    @model_validator(mode="after")
    def set_query_path_default(self) -> DatasetConfig:
        """Default query_path to base_path if not specified."""
        if self.query_path is None:
            self.query_path = self.base_path
        return self


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration.

    This is the main configuration loaded from YAML/JSON files.
    """

    name: str = Field(default="ANN Benchmark", description="Benchmark run name")
    description: str = Field(default="", description="Optional description")
    data_dir: Path = Field(default=Path("./data"), description="Base directory for datasets")
    results_dir: Path = Field(default=Path("./results"), description="Directory for results")
    index_dir: Path = Field(default=Path("./indices"), description="Directory for built indices")
    algorithms: list[AlgorithmConfig] = Field(default_factory=list)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    monitor_interval_ms: int = Field(
        default=100, ge=50, le=1000, description="Resource monitor interval"
    )
    include_raw_samples: bool = Field(
        default=False,
        description="Also include raw samples in results_detailed.json (debug JSONL always stored)",
    )

    @property
    def enabled_algorithms(self) -> list[AlgorithmConfig]:
        """Return only enabled algorithms."""
        return [a for a in self.algorithms if not a.disabled]


class ResourceSample(BaseModel):
    """Single resource monitoring sample from Docker stats."""

    timestamp: datetime
    memory_usage_bytes: int = Field(ge=0)
    memory_limit_bytes: int | None = Field(default=None, ge=0)
    memory_percent: float | None = Field(default=None, ge=0, le=100)
    cpu_percent: float = Field(ge=0)
    blkio_read_bytes: int = Field(default=0, ge=0)
    blkio_write_bytes: int = Field(default=0, ge=0)
    pids: int = Field(default=1, ge=0)


class ResourceSummary(BaseModel):
    """Aggregated resource metrics from monitoring.

    These metrics are critical for evaluating disk-based algorithms
    where IOPS indicate disk access patterns.
    """

    peak_memory_mb: float = Field(ge=0, description="Peak RSS in megabytes")
    avg_memory_mb: float = Field(ge=0, description="Average RSS in megabytes")
    cpu_time_total_seconds: float = Field(
        default=0.0, ge=0, description="Total CPU time from cgroups"
    )
    avg_cpu_percent: float = Field(ge=0, description="Average CPU utilization")
    peak_cpu_percent: float = Field(ge=0, description="Peak CPU utilization")
    total_blkio_read_mb: float = Field(ge=0, description="Total bytes read from block devices")
    total_blkio_write_mb: float = Field(ge=0, description="Total bytes written to block devices")
    total_read_ops: int = Field(default=0, ge=0, description="Total read I/O operations")
    total_write_ops: int = Field(default=0, ge=0, description="Total write I/O operations")
    avg_read_iops: float = Field(ge=0, description="Average read IOPS")
    avg_write_iops: float = Field(ge=0, description="Average write IOPS")
    # Issue #5 fix: None when kernel does not expose rusec/wusec in io.stat
    total_read_usec: int = Field(
        default=0, ge=0, description="Total read service time from io.stat (microseconds)"
    )
    total_write_usec: int = Field(
        default=0, ge=0, description="Total write service time from io.stat (microseconds)"
    )
    io_pressure_some_total_usec: int = Field(
        default=0, ge=0, description="Total PSI io.some time (microseconds)"
    )
    io_pressure_full_total_usec: int = Field(
        default=0, ge=0, description="Total PSI io.full time (microseconds)"
    )
    psi_available: bool = Field(
        default=False,
        description="Whether PSI counters were observed as non-zero in any sample; "
        "distinguishes a measured zero-stall window from PSI being unavailable",
    )
    pgmajfault_delta: int = Field(
        default=0, ge=0, description="Major page faults during window (delta)"
    )
    pgfault_delta: int = Field(default=0, ge=0, description="Total page faults (delta)")
    avg_file_bytes: float = Field(default=0.0, ge=0, description="Avg file cache bytes")
    peak_file_bytes: int = Field(default=0, ge=0, description="Peak file cache bytes")
    avg_file_mapped_bytes: float = Field(default=0.0, ge=0, description="Avg mapped file bytes")
    peak_file_mapped_bytes: int = Field(default=0, ge=0, description="Peak mapped file bytes")
    avg_active_file_bytes: float = Field(
        default=0.0, ge=0, description="Avg active file cache bytes"
    )
    peak_active_file_bytes: int = Field(default=0, ge=0, description="Peak active file cache bytes")
    avg_inactive_file_bytes: float = Field(
        default=0.0, ge=0, description="Avg inactive file cache bytes"
    )
    peak_inactive_file_bytes: int = Field(
        default=0, ge=0, description="Peak inactive file cache bytes"
    )
    nr_throttled_delta: int = Field(default=0, ge=0, description="CPU throttle count (delta)")
    throttled_usec_delta: int = Field(
        default=0, ge=0, description="CPU throttled time (microseconds)"
    )
    top_read_device: dict[str, int | str] | None = Field(
        default=None,
        description="Top device by read bytes {device, total_read_bytes, total_write_bytes, total_read_ops, total_write_ops}",
    )
    p95_read_iops: float | None = Field(
        default=None, ge=0, description="95th percentile read IOPS (interval-based)"
    )
    max_read_iops: float | None = Field(
        default=None, ge=0, description="Max read IOPS (interval-based)"
    )
    p95_read_mbps: float | None = Field(
        default=None, ge=0, description="95th percentile read MB/s (interval-based)"
    )
    max_read_mbps: float | None = Field(
        default=None, ge=0, description="Max read MB/s (interval-based)"
    )
    p95_read_service_time_ms: float | None = Field(
        default=None, ge=0, description="95th percentile read service time (ms/op)"
    )
    max_read_service_time_ms: float | None = Field(
        default=None, ge=0, description="Max read service time (ms/op)"
    )
    # Queue depth (system-wide in-flight I/O, sampled per interval)
    avg_queue_depth: float | None = Field(default=None, ge=0)
    max_queue_depth: int | None = Field(default=None, ge=0)
    p95_queue_depth: float | None = Field(default=None, ge=0)
    sample_count: int = Field(ge=0, description="Number of samples collected")
    duration_seconds: float = Field(
        ge=0,
        description="Sample span duration (first to last sample). See TimeBases for explicit time bases.",
    )
    samples: list[ResourceSample] = Field(default_factory=list, description="Raw samples")
    block_size: int = Field(default=4096, ge=512, description="Detected system block size in bytes")
    debug_samples_path: Path | None = Field(
        default=None,
        description="Path to JSONL file containing raw samples for debugging",
    )
    filtered_samples_meta: dict[str, int | str] | None = Field(
        default=None,
        description="Counts and reasons for samples filtered during aggregation",
    )


# =============================================================================
# STRUCTURED METRICS (CRITICAL + HIGH Priority Only)
# =============================================================================


class CPUMetrics(BaseModel):
    """Structured CPU metrics (HIGH priority).

    Metrics are separated by phase for accurate analysis:
    - BUILD: Index construction (can take hours for large datasets)
    - WARMUP: Index loading into memory/cache (search container startup)
    - SEARCH: Actual query execution (primary benchmark metric)

    CPU time is deterministic and unaffected by other processes on the system.
    """

    # BUILD phase CPU metrics (index construction)
    build_cpu_time_seconds: float = Field(
        default=0.0, ge=0, description="CPU time during index build phase in seconds"
    )
    build_peak_cpu_percent: float = Field(
        default=0.0, ge=0, description="Peak CPU utilization during build phase %"
    )

    # WARMUP phase CPU metrics (index loading before queries)
    warmup_cpu_time_seconds: float = Field(
        default=0.0, ge=0, description="CPU time during index warmup/load phase in seconds"
    )
    warmup_peak_cpu_percent: float = Field(
        default=0.0, ge=0, description="Peak CPU utilization during warmup phase %"
    )

    # SEARCH phase CPU metrics (primary focus for benchmarking)
    search_cpu_time_seconds: float = Field(
        default=0.0, ge=0, description="CPU time during search phase in seconds"
    )
    search_avg_cpu_percent: float = Field(
        default=0.0, ge=0, description="Average CPU utilization during search phase %"
    )
    search_peak_cpu_percent: float = Field(
        default=0.0, ge=0, description="Peak CPU utilization during search phase %"
    )

    # CPU time per query (stable comparison metric)
    search_cpu_time_per_query_ms: float = Field(
        default=0.0,
        ge=0,
        description="Aggregate CPU time per query in milliseconds (total CPU time across all "
        "threads divided by query count; may exceed wall-clock latency on multi-threaded runs)",
    )

    # CPU throttling metrics (from cgroups cpu.stat)
    search_nr_throttled: int = Field(
        default=0,
        ge=0,
        description="Number of times the CPU was throttled during search phase",
    )
    search_throttled_usec: int = Field(
        default=0,
        ge=0,
        description="Total microseconds the CPU was throttled during search phase",
    )


class MemoryMetrics(BaseModel):
    """Structured memory metrics (HIGH priority).

    Focuses on RSS (Resident Set Size) which represents actual physical memory used.
    Metrics are separated by phase for accurate analysis:
    - BUILD: Memory used during index construction
    - WARMUP: Memory used while loading index into memory/cache
    - SEARCH: Memory used during query execution (primary metric for disk-based algorithms)
    """

    # BUILD phase memory
    build_peak_rss_mb: float = Field(
        default=0.0, ge=0, description="Peak RSS during index build phase in MB"
    )

    # WARMUP phase memory (index loading)
    warmup_peak_rss_mb: float = Field(
        default=0.0, ge=0, description="Peak RSS during index warmup/load phase in MB"
    )

    # SEARCH phase memory (primary metric)
    search_peak_rss_mb: float = Field(
        default=0.0, ge=0, description="Peak RSS during search phase in MB"
    )
    search_avg_rss_mb: float = Field(
        default=0.0, ge=0, description="Average RSS during search phase in MB"
    )

    # Cache/fault statistics (optional - requires /proc or memory.stat access).
    # FAULT-BASED only: these reflect mmap/page-fault traffic, NOT direct I/O.
    # Algorithms that read via O_DIRECT/pread (e.g. DiskANN) produce ~no faults and
    # therefore report None here; use disk_io.search_pages_per_query /
    # bytes_read_per_query / IOPS as the authoritative disk metrics for them.
    search_major_faults: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Major page faults during search (disk-backed pages). Fault-based only; "
            "zero/None for O_DIRECT workloads that bypass the page cache entirely."
        ),
    )
    search_page_cache_hit_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Fraction of page-fault accesses served without disk I/O, computed as "
            "1 - (pgmajfault / pgfault) from cgroup memory.stat. This is the OS "
            "page-cache hit rate for FAULTED pages only; it is None when there are "
            "no faults to judge (e.g. O_DIRECT readers). Algorithm-internal caches "
            "that avoid syscalls/mmap faults are not visible here (see algorithm_stats)."
        ),
    )


class DiskIOMetrics(BaseModel):
    """Structured disk I/O metrics (CRITICAL + HIGH priority).

    Primary focus for evaluating disk-based ANN algorithms like DiskANN and SPANN.
    All metrics sourced from cgroups v2 io.stat for accuracy.

    IMPORTANT: Page metrics use STANDARD_PAGE_SIZE (4KB) for cross-system comparability,
    regardless of the actual physical block size of the underlying storage device.
    This ensures consistent metrics when comparing results across different hardware.

    Phases:
    - WARMUP: I/O during index loading (may include sequential reads, mmap faults)
    - SEARCH: I/O during query execution (primary metric for disk-based algorithms)
    """

    # WARMUP phase I/O metrics (index loading)
    warmup_read_mb: float = Field(
        default=0.0, ge=0, description="Total MB read during warmup/index load phase"
    )
    warmup_write_mb: float = Field(
        default=0.0, ge=0, description="Total MB written during warmup phase"
    )
    warmup_io_stall_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of time stalled on I/O during warmup (from PSI)",
    )
    warmup_major_faults_per_second: float | None = Field(
        default=None, ge=0, description="Major faults per second during warmup"
    )
    warmup_file_cache_avg_mb: float | None = Field(
        default=None, ge=0, description="Average file cache size during warmup (MB)"
    )
    warmup_file_cache_peak_mb: float | None = Field(
        default=None, ge=0, description="Peak file cache size during warmup (MB)"
    )

    # SEARCH phase IOPS metrics (operations per second)
    search_avg_read_iops: float = Field(
        default=0.0, ge=0, description="Average read IOPS during search phase"
    )
    search_avg_write_iops: float = Field(
        default=0.0, ge=0, description="Average write IOPS during search phase"
    )

    # SEARCH phase throughput metrics
    search_avg_read_throughput_mbps: float = Field(
        default=0.0, ge=0, description="Average read throughput in MB/s during search phase"
    )
    search_avg_write_throughput_mbps: float = Field(
        default=0.0, ge=0, description="Average write throughput in MB/s during search phase"
    )

    # SEARCH phase total volume metrics
    search_total_read_mb: float = Field(
        default=0.0, ge=0, description="Total MB read during search phase"
    )
    search_total_write_mb: float = Field(
        default=0.0, ge=0, description="Total MB written during search phase"
    )
    search_total_pages_read: int = Field(
        default=0, ge=0, description="Total 4KB pages read during search phase"
    )
    search_total_pages_written: int = Field(
        default=0, ge=0, description="Total 4KB pages written during search phase"
    )
    search_pages_per_query: float | None = Field(
        default=None, ge=0, description="Average 4KB pages read per query (search phase only)"
    )
    search_reads_per_query: float | None = Field(
        default=None,
        ge=0,
        description="Average block-device read operations per query (search phase only)",
    )
    search_bytes_read_per_query: float | None = Field(
        default=None,
        ge=0,
        description="Average bytes read from block devices per query (search phase only)",
    )

    # Queue depth (system-wide in-flight I/O, sampled per interval)
    search_avg_queue_depth: float | None = Field(
        default=None,
        ge=0,
        description="Average in-flight I/O operations during search (from /sys/block/*/inflight)",
    )
    search_p95_queue_depth: float | None = Field(
        default=None, ge=0, description="95th percentile in-flight I/O ops across samples"
    )
    search_max_queue_depth: int | None = Field(
        default=None, ge=0, description="Maximum in-flight I/O ops observed in any sample"
    )

    # Service time proxy metrics (bytes per operation - indicates I/O pattern efficiency)
    search_avg_bytes_per_read_op: float | None = Field(
        default=None,
        ge=0,
        description="Average bytes per read operation (proxy for I/O efficiency)",
    )
    search_avg_bytes_per_write_op: float | None = Field(
        default=None, ge=0, description="Average bytes per write operation"
    )
    search_avg_read_service_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Average read service time per op (ms) from rusec/rios",
    )
    search_avg_write_service_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Average write service time per op (ms) from wusec/wios",
    )

    # Tail metrics for IOPS (p95/max)
    search_p95_read_iops: float | None = Field(
        default=None, ge=0, description="95th percentile read IOPS (from per-interval samples)"
    )
    search_max_read_iops: float | None = Field(
        default=None, ge=0, description="Maximum read IOPS observed in any sampling interval"
    )
    search_p95_read_mbps: float | None = Field(
        default=None, ge=0, description="95th percentile read MB/s (from per-interval samples)"
    )
    search_max_read_mbps: float | None = Field(
        default=None, ge=0, description="Maximum read MB/s observed in any sampling interval"
    )
    search_p95_read_service_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="95th percentile read service time per op (ms) from interval deltas",
    )
    search_max_read_service_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Maximum read service time per op (ms) from interval deltas",
    )

    # PSI (Pressure Stall Information) metrics - Linux 4.20+
    search_io_stall_percent: float | None = Field(
        default=None, ge=0, le=100, description="Percentage of time stalled on I/O (from PSI some)"
    )
    search_io_full_stall_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of time ALL tasks stalled on I/O (from PSI full). "
        "More severe than io_stall_percent; indicates complete I/O blockage.",
    )
    search_major_faults_per_query: float | None = Field(
        default=None, ge=0, description="Major page faults per query during search"
    )
    search_major_faults_per_second: float | None = Field(
        default=None, ge=0, description="Major page faults per second during search"
    )
    search_file_cache_avg_mb: float | None = Field(
        default=None, ge=0, description="Average file cache size during search (MB)"
    )
    search_file_cache_peak_mb: float | None = Field(
        default=None, ge=0, description="Peak file cache size during search (MB)"
    )
    # File cache breakdown: mapped (page cache backed), active, inactive
    search_file_mapped_avg_mb: float | None = Field(
        default=None, ge=0, description="Average mapped file bytes during search (MB)"
    )
    search_file_active_avg_mb: float | None = Field(
        default=None, ge=0, description="Average active file cache during search (MB)"
    )
    search_file_inactive_avg_mb: float | None = Field(
        default=None, ge=0, description="Average inactive file cache during search (MB)"
    )

    # Metadata for transparency
    physical_block_size: int = Field(
        default=4096, ge=512, description="Detected physical block size of storage device (bytes)"
    )
    sample_count: int = Field(
        default=0, ge=0, description="Number of samples collected for I/O metrics"
    )


class LatencyMetrics(BaseModel):
    """Latency distribution metrics (HIGH priority).

    Standard percentiles for understanding query latency characteristics.
    """

    mean_ms: float = Field(default=0.0, ge=0, description="Mean query latency in ms")
    p50_ms: float | None = Field(default=None, ge=0, description="Median (p50) latency in ms")
    p95_ms: float | None = Field(default=None, ge=0, description="95th percentile latency in ms")
    p99_ms: float | None = Field(default=None, ge=0, description="99th percentile latency in ms")
    max_ms: float | None = Field(default=None, ge=0, description="Maximum query latency in ms")


class AlgorithmStats(BaseModel):
    """Algorithm-reported search statistics (optional container protocol extension).

    Algorithms report raw totals for the whole timed run via the ``stats`` key of
    ``ContainerProtocol.SearchOutput``. Well-known keys (all optional):

    - ``distance_computations``: vector distance evaluations performed
    - ``hops``: graph edges traversed during search
    - ``candidates_explored``: candidate nodes expanded/visited
    - ``cache_hits`` / ``cache_misses``: algorithm-internal cache accesses
    - ``io_reads`` / ``io_bytes_read``: algorithm-level storage reads

    Unknown keys are preserved in ``extra`` so any custom counter survives the
    round trip. Per-query normalizations are computed by the evaluator.
    """

    # Raw totals reported by the algorithm (whole timed run)
    distance_computations: int | None = Field(
        default=None, ge=0, description="Total distance computations during search"
    )
    hops: int | None = Field(
        default=None, ge=0, description="Total graph hops/edges traversed during search"
    )
    candidates_explored: int | None = Field(
        default=None, ge=0, description="Total candidate nodes explored during search"
    )
    cache_hits: int | None = Field(
        default=None, ge=0, description="Algorithm-level cache hits during search"
    )
    cache_misses: int | None = Field(
        default=None, ge=0, description="Algorithm-level cache misses during search"
    )
    io_reads: int | None = Field(
        default=None,
        ge=0,
        description="Algorithm-level storage read operations (e.g., SSD node reads)",
    )
    io_bytes_read: int | None = Field(
        default=None, ge=0, description="Algorithm-level bytes read from storage"
    )

    # Any additional counters emitted by the algorithm
    extra: dict[str, float] = Field(
        default_factory=dict, description="Additional algorithm-reported counters"
    )

    # Per-query normalizations (computed by the evaluator, not reported)
    distance_computations_per_query: float | None = Field(
        default=None, ge=0, description="Distance computations per query"
    )
    hops_per_query: float | None = Field(default=None, ge=0, description="Graph hops per query")
    candidates_explored_per_query: float | None = Field(
        default=None, ge=0, description="Candidates explored per query"
    )

    @classmethod
    def from_output(cls, stats: dict[str, Any]) -> AlgorithmStats:
        """Build AlgorithmStats from a container-reported stats dict.

        Known keys map to typed fields; unknown numeric keys land in ``extra``.
        Non-numeric or negative values are ignored with a warning.
        """
        known = {
            "distance_computations",
            "hops",
            "candidates_explored",
            "cache_hits",
            "cache_misses",
            "io_reads",
            "io_bytes_read",
        }
        fields: dict[str, Any] = {"extra": {}}
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                logger.warning(f"Ignoring non-numeric/negative algorithm stat: {key}={value!r}")
                continue
            if key in known:
                fields[key] = int(value)
            else:
                fields["extra"][key] = float(value)
        return cls(**fields)

    def with_per_query(self, num_queries: int) -> AlgorithmStats:
        """Return a copy with per-query normalizations of the well-known totals.

        Args:
            num_queries: Total queries executed in the timed run (0 disables).
        """
        if num_queries <= 0:
            return self
        normalized = self.model_copy()
        if normalized.distance_computations is not None:
            normalized.distance_computations_per_query = (
                normalized.distance_computations / num_queries
            )
        if normalized.hops is not None:
            normalized.hops_per_query = normalized.hops / num_queries
        if normalized.candidates_explored is not None:
            normalized.candidates_explored_per_query = (
                normalized.candidates_explored / num_queries
            )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for results output (None fields omitted)."""
        values: dict[str, Any] = {
            "distance_computations": self.distance_computations,
            "hops": self.hops,
            "candidates_explored": self.candidates_explored,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "io_reads": self.io_reads,
            "io_bytes_read": self.io_bytes_read,
            **self.extra,
        }
        return {k: v for k, v in values.items() if v is not None}


class TimeBases(BaseModel):
    """Explicit time bases for rate metric calculation.

    Research-grade benchmarking requires unambiguous time denominators.
    Query-window metrics are primary for algorithm comparison;
    container-lifetime metrics are kept for transparency.

    Phases:
    - container_duration: Total wall time of the container (includes startup overhead)
    - warmup_duration: Time spent loading index into memory/cache
    - query_duration: Time spent executing actual queries (PRIMARY time base for search metrics)
    """

    container_duration_seconds: float = Field(
        default=0.0, ge=0, description="Total container wall time from start to exit"
    )
    sample_span_seconds: float = Field(
        default=0.0, ge=0, description="Time span between first and last resource samples"
    )
    warmup_duration_seconds: float | None = Field(
        default=None, ge=0, description="Time to load/warm up the index (before queries)"
    )
    query_duration_seconds: float | None = Field(
        default=None, ge=0, description="Time spent executing queries only (PRIMARY time base)"
    )
    query_start_timestamp: str | None = Field(
        default=None, description="ISO-8601 timestamp when query execution started"
    )
    query_end_timestamp: str | None = Field(
        default=None, description="ISO-8601 timestamp when query execution ended"
    )


class PhaseResult(BaseModel):
    """Result from a single benchmark phase (build or search).

    For search phase, resources are split into:
    - warmup_resources: Metrics from index loading/warmup (before queries)
    - resources: Metrics from query execution only (primary benchmark data)
    """

    phase: str = Field(..., description="Phase name: 'build' or 'search'")
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)
    duration_seconds: float = Field(ge=0)
    stdout_path: Path | None = Field(default=None, description="Path to stdout log file")
    stderr_path: Path | None = Field(default=None, description="Path to stderr log file")
    resources: ResourceSummary
    warmup_resources: ResourceSummary | None = Field(
        default=None, description="Resources used during warmup/index load sub-phase"
    )
    output: dict[str, Any] = Field(default_factory=dict, description="Algorithm output")
    time_bases: TimeBases | None = Field(
        default=None, description="Explicit time bases for this phase"
    )


class BenchmarkResult(BaseModel):
    """Complete result from benchmarking a single algorithm on a dataset.

    Uses structured metrics schemas for organized, type-safe metric access.
    This is the primary output format for benchmark runs.
    """

    # Identification
    run_id: UUID = Field(
        default_factory=uuid4, description="Unique run identifier for log correlation"
    )
    algorithm: str
    dataset: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Phase results (contain raw ResourceSummary for backward compatibility)
    build_result: PhaseResult | None = Field(default=None)
    search_result: PhaseResult | None = Field(default=None)

    # Structured metrics (CRITICAL + HIGH priority)
    cpu: CPUMetrics = Field(default_factory=CPUMetrics, description="CPU metrics")
    memory: MemoryMetrics = Field(default_factory=MemoryMetrics, description="Memory metrics")
    disk_io: DiskIOMetrics = Field(
        default_factory=DiskIOMetrics, description="Disk I/O metrics (primary focus)"
    )
    latency: LatencyMetrics = Field(
        default_factory=LatencyMetrics, description="Query latency distribution"
    )

    # Optional algorithm-reported statistics (distance comps, hops, candidates, ...)
    algorithm_stats: AlgorithmStats | None = Field(
        default=None, description="Algorithm-reported search statistics (if provided)"
    )

    # Quality metrics
    recall: float | None = Field(default=None, ge=0, le=1, description="Recall@k")
    qps: float | None = Field(default=None, ge=0, description="Queries per second")

    # Build phase summary
    total_build_time_seconds: float | None = Field(default=None, ge=0)
    index_size_bytes: int | None = Field(default=None, ge=0)

    # Configuration (hyperparameters used for this run)
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Combined build and search hyperparameters"
    )

    def _get_time_base(self, field: str) -> Any:
        """Safely extract a field from search_result.time_bases."""
        if self.search_result and self.search_result.time_bases:
            return getattr(self.search_result.time_bases, field, None)
        return None

    def to_summary_dict(self) -> BenchmarkSummaryDict:
        """Convert to phase-structured dictionary for human-readable JSON output.

        Returns a BenchmarkSummaryDict organized by benchmark phase:
        - quality: Primary metrics (recall, QPS)
        - build: Index construction phase
        - warmup: Index loading/cache warming phase
        - search: Query execution phase (primary benchmark data)
        - latency: Query latency distribution
        - metadata: Run metadata and timestamps

        This is the format used for results.json output.
        """
        return BenchmarkSummaryDict(
            algorithm=self.algorithm,
            dataset=self.dataset,
            timestamp=self.timestamp.isoformat(),
            quality=self._build_quality_dict(),
            build=self._build_build_phase_dict(),
            warmup=self._build_warmup_phase_dict(),
            search=self._build_search_phase_dict(),
            latency=self._build_latency_dict(),
            algorithm_stats=self.algorithm_stats.to_dict() if self.algorithm_stats else None,
            metadata=self._build_metadata_dict(),
            debug_artifacts=self._build_debug_artifacts_dict(),
            hyperparameters=self.hyperparameters,
        )

    def _build_quality_dict(self) -> QualityMetricsDict:
        return QualityMetricsDict(
            recall=round(self.recall, 6) if self.recall is not None else None,
            qps=self.qps,
        )

    def _build_build_phase_dict(self) -> BuildPhaseDict:
        return BuildPhaseDict(
            duration_seconds=self.total_build_time_seconds,
            index_size_bytes=self.index_size_bytes,
            cpu_time_seconds=self.cpu.build_cpu_time_seconds,
            peak_cpu_percent=self.cpu.build_peak_cpu_percent,
            peak_rss_mb=self.memory.build_peak_rss_mb,
            error=self.build_result.error_message if self.build_result else None,
        )

    def _build_warmup_phase_dict(self) -> WarmupPhaseDict:
        return WarmupPhaseDict(
            duration_seconds=self._get_time_base("warmup_duration_seconds"),
            cpu_time_seconds=self.cpu.warmup_cpu_time_seconds,
            peak_cpu_percent=self.cpu.warmup_peak_cpu_percent,
            peak_rss_mb=self.memory.warmup_peak_rss_mb,
            read_mb=self.disk_io.warmup_read_mb,
            write_mb=self.disk_io.warmup_write_mb,
            io_stall_percent=self.disk_io.warmup_io_stall_percent,
            major_faults_per_second=self.disk_io.warmup_major_faults_per_second,
            file_cache_avg_mb=self.disk_io.warmup_file_cache_avg_mb,
            file_cache_peak_mb=self.disk_io.warmup_file_cache_peak_mb,
        )

    def _build_search_phase_dict(self) -> SearchPhaseDict:
        return SearchPhaseDict(
            duration_seconds=self._get_time_base("query_duration_seconds"),
            cpu_time_seconds=self.cpu.search_cpu_time_seconds,
            cpu_time_per_query_ms=self.cpu.search_cpu_time_per_query_ms,
            avg_cpu_percent=self.cpu.search_avg_cpu_percent,
            peak_cpu_percent=self.cpu.search_peak_cpu_percent,
            peak_rss_mb=self.memory.search_peak_rss_mb,
            avg_rss_mb=self.memory.search_avg_rss_mb,
            disk_io=self._build_disk_io_dict(),
            major_faults_per_query=self.disk_io.search_major_faults_per_query,
            major_faults_per_second=self.disk_io.search_major_faults_per_second,
            page_cache_hit_rate=self.memory.search_page_cache_hit_rate,
            file_cache_avg_mb=self.disk_io.search_file_cache_avg_mb,
            file_cache_peak_mb=self.disk_io.search_file_cache_peak_mb,
            cpu_nr_throttled=self.cpu.search_nr_throttled,
            cpu_throttled_usec=self.cpu.search_throttled_usec,
            error=self.search_result.error_message if self.search_result else None,
        )

    def _build_disk_io_dict(self) -> DiskIODict:
        return DiskIODict(
            avg_read_iops=self.disk_io.search_avg_read_iops,
            avg_write_iops=self.disk_io.search_avg_write_iops,
            avg_read_throughput_mbps=self.disk_io.search_avg_read_throughput_mbps,
            avg_write_throughput_mbps=self.disk_io.search_avg_write_throughput_mbps,
            total_read_mb=self.disk_io.search_total_read_mb,
            total_write_mb=self.disk_io.search_total_write_mb,
            total_pages_read=self.disk_io.search_total_pages_read,
            total_pages_written=self.disk_io.search_total_pages_written,
            pages_per_query=self.disk_io.search_pages_per_query,
            reads_per_query=self.disk_io.search_reads_per_query,
            bytes_read_per_query=self.disk_io.search_bytes_read_per_query,
            avg_queue_depth=self.disk_io.search_avg_queue_depth,
            max_queue_depth=self.disk_io.search_max_queue_depth,
            p95_queue_depth=self.disk_io.search_p95_queue_depth,
            # Service time proxy / efficiency metrics
            avg_bytes_per_read_op=self.disk_io.search_avg_bytes_per_read_op,
            avg_bytes_per_write_op=self.disk_io.search_avg_bytes_per_write_op,
            avg_read_service_time_ms=self.disk_io.search_avg_read_service_time_ms,
            avg_write_service_time_ms=self.disk_io.search_avg_write_service_time_ms,
            # Tail metrics (p95/max IOPS)
            p95_read_iops=self.disk_io.search_p95_read_iops,
            max_read_iops=self.disk_io.search_max_read_iops,
            p95_read_mbps=self.disk_io.search_p95_read_mbps,
            max_read_mbps=self.disk_io.search_max_read_mbps,
            p95_read_service_time_ms=self.disk_io.search_p95_read_service_time_ms,
            max_read_service_time_ms=self.disk_io.search_max_read_service_time_ms,
            # PSI stall metrics
            io_stall_percent=self.disk_io.search_io_stall_percent,
            io_full_stall_percent=self.disk_io.search_io_full_stall_percent,
        )

    def _build_latency_dict(self) -> LatencyDict:
        return LatencyDict(
            mean_ms=self.latency.mean_ms,
            p50_ms=self.latency.p50_ms,
            p95_ms=self.latency.p95_ms,
            p99_ms=self.latency.p99_ms,
            max_ms=self.latency.max_ms,
        )

    def _build_metadata_dict(self) -> MetadataDict:
        return MetadataDict(
            run_id=str(self.run_id),
            physical_block_size=self.disk_io.physical_block_size,
            sample_count=self.disk_io.sample_count,
            query_start_timestamp=self._get_time_base("query_start_timestamp"),
            query_end_timestamp=self._get_time_base("query_end_timestamp"),
        )

    def _build_debug_artifacts_dict(self) -> DebugArtifactsDict:
        return DebugArtifactsDict(
            samples_path=self._get_debug_samples_path(),
            samples_count=self.disk_io.sample_count,
        )

    def _get_debug_samples_path(self) -> str | None:
        """Resolve debug samples path from phase results if available."""
        for phase in [self.search_result, self.build_result]:
            if phase and phase.resources.debug_samples_path:
                return str(phase.resources.debug_samples_path)
        return None

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame/CSV creation.

        Flattens nested metric objects into prefixed column names:
        - cpu.build_cpu_time_seconds -> cpu_build_cpu_time_seconds
        - disk_io.search_avg_read_iops -> disk_io_search_avg_read_iops

        Metrics are organized by phase: build, warmup (index load), search.
        """
        # Round recall to 6 decimal places to avoid floating-point artifacts
        recall_rounded = round(self.recall, 6) if self.recall is not None else None

        data: dict[str, Any] = {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "timestamp": self.timestamp.isoformat(),
            "recall": recall_rounded,
            "qps": self.qps,
            "total_build_time_seconds": self.total_build_time_seconds,
            "index_size_bytes": self.index_size_bytes,
            "build_error": self.build_result.error_message if self.build_result else None,
            "search_error": self.search_result.error_message if self.search_result else None,
        }

        # Flatten nested metrics with prefix
        for prefix, metrics in [
            ("cpu", self.cpu),
            ("memory", self.memory),
            ("disk_io", self.disk_io),
            ("latency", self.latency),
        ]:
            for key, value in metrics.model_dump().items():
                data[f"{prefix}_{key}"] = value

        # Flatten time bases if available
        if self.search_result and self.search_result.time_bases:
            for key, value in self.search_result.time_bases.model_dump().items():
                data[f"time_{key}"] = value

        # Flatten hyperparameters into prefixed hp_* columns so sweeps become
        # stable, sortable CSV columns (e.g. hp_search_ef -> 100).
        for key, value in _flatten_dict(self.hyperparameters).items():
            data[f"hp_{key}"] = value

        # Flatten algorithm-reported stats into stats_* columns for the dashboard.
        if self.algorithm_stats is not None:
            for key, value in _flatten_dict(self.algorithm_stats.to_dict()).items():
                data[f"stats_{key}"] = value

        return data


class ContainerProtocol(BaseModel):
    """JSON protocol for communication with algorithm containers.

    This defines the interface that all algorithm containers must implement.
    """

    class BuildInput(BaseModel):
        """Input for build phase."""

        mode: str = Field(default="build", frozen=True)
        dataset_path: str = Field(..., description="Path to base vectors inside container")
        index_path: str = Field(..., description="Path to write index")
        dimension: int = Field(..., ge=1)
        metric: str = Field(..., description="Distance metric")
        build_args: dict[str, Any] = Field(default_factory=dict)

    class BuildOutput(BaseModel):
        """Expected output from build phase."""

        status: str = Field(..., description="'success' or 'error'")
        build_time_seconds: float = Field(ge=0)
        index_size_bytes: int = Field(ge=0)
        error_message: str | None = Field(default=None)

    class SearchInput(BaseModel):
        """Input for search phase."""

        mode: str = Field(default="search", frozen=True)
        index_path: str = Field(..., description="Path to index directory")
        queries_path: str = Field(..., description="Path to query vectors")
        ground_truth_path: str | None = Field(default=None)
        k: int = Field(default=10, ge=1)
        query_rounds: int = Field(default=1, ge=1)
        batch_mode: bool = Field(default=False)
        search_args: dict[str, Any] = Field(default_factory=dict)
        # Warmup configuration
        cache_warmup_queries: int = Field(
            default=0, ge=0, description="Number of untimed queries to warm caches before benchmark"
        )

    class SearchOutput(BaseModel):
        """Expected output from search phase.

        Timing fields enable precise phase separation:
        - warmup: Index loading/initialization (warmup_start -> warmup_end)
        - query: Actual search execution (query_start -> query_end)
        """

        status: str
        # Optional algorithm-reported counters; see AlgorithmStats for well-known keys.
        stats: dict[str, float] | None = Field(default=None)
        total_queries: int = Field(ge=0)
        total_time_seconds: float = Field(ge=0)
        qps: float = Field(ge=0)
        recall: float | None = Field(default=None, ge=0, le=1)
        mean_latency_ms: float = Field(ge=0)
        p50_latency_ms: float | None = Field(default=None, ge=0)
        p95_latency_ms: float | None = Field(default=None, ge=0)
        p99_latency_ms: float | None = Field(default=None, ge=0)
        max_latency_ms: float | None = Field(default=None, ge=0)
        error_message: str | None = Field(default=None)
        # Phase timing for resource window filtering
        warmup_duration_seconds: float | None = Field(default=None, ge=0)
        warmup_start_timestamp: str | None = Field(default=None)
        warmup_end_timestamp: str | None = Field(default=None)
        # Backward-compatible legacy "load_*" fields (older containers).
        load_duration_seconds: float | None = Field(default=None, ge=0)
        load_start_timestamp: str | None = Field(default=None)
        load_end_timestamp: str | None = Field(default=None)
        # Optional cache warmup (untimed) metadata for research reproducibility.
        cache_warmup_queries_requested: int | None = Field(default=None, ge=0)
        cache_warmup_queries_executed: int | None = Field(default=None, ge=0)
        cache_warmup_duration_seconds: float | None = Field(default=None, ge=0)
        query_start_timestamp: str | None = Field(default=None)
        query_end_timestamp: str | None = Field(default=None)
