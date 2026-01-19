"""Pydantic schemas for ANN Benchmarking Suite.

This module defines all data contracts used throughout the benchmarking suite,
including algorithm configurations, dataset definitions, and result structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


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
    args: dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific build args")

    model_config = {"extra": "allow"}


class SearchConfig(BaseModel):
    """Configuration for the search/query phase."""

    timeout_seconds: int = Field(default=600, ge=10, description="Search phase timeout")
    k: int = Field(default=10, ge=1, le=1000, description="Number of neighbors to retrieve")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific search args (can be list for sweeps)"
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
    cpu_limit: str | None = Field(default=None, description="CPU limit (e.g., '4' or '0-3')")
    memory_limit: str | None = Field(default=None, description="Memory limit (e.g., '8g')")
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

    @property
    def enabled_algorithms(self) -> list[AlgorithmConfig]:
        """Return only enabled algorithms."""
        return [a for a in self.algorithms if not a.disabled]


class ResourceSample(BaseModel):
    """Single resource monitoring sample from Docker stats."""

    timestamp: datetime
    memory_usage_bytes: int = Field(ge=0)
    memory_limit_bytes: int = Field(ge=0)
    memory_percent: float = Field(ge=0, le=100)
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
    cpu_time_total_seconds: float = Field(default=0.0, ge=0, description="Total CPU time from cgroups")
    avg_cpu_percent: float = Field(ge=0, description="Average CPU utilization")
    peak_cpu_percent: float = Field(ge=0, description="Peak CPU utilization")
    total_blkio_read_mb: float = Field(ge=0, description="Total bytes read from block devices")
    total_blkio_write_mb: float = Field(ge=0, description="Total bytes written to block devices")
    avg_read_iops: float = Field(ge=0, description="Average read IOPS")
    avg_write_iops: float = Field(ge=0, description="Average write IOPS")
    sample_count: int = Field(ge=0, description="Number of samples collected")
    duration_seconds: float = Field(ge=0, description="Monitoring duration")
    samples: list[ResourceSample] = Field(default_factory=list, description="Raw samples")


# =============================================================================
# STRUCTURED METRICS (CRITICAL + HIGH Priority Only)
# =============================================================================


class CPUMetrics(BaseModel):
    """Structured CPU metrics (HIGH priority).

    Primary metric is cpu_time_total_seconds which is deterministic
    and unaffected by other processes running on the system.
    """

    cpu_time_total_seconds: float = Field(
        default=0.0, ge=0, description="Total CPU time (user + system) from cgroups"
    )
    avg_cpu_percent: float = Field(default=0.0, ge=0, description="Average CPU utilization %")
    peak_cpu_percent: float = Field(default=0.0, ge=0, description="Peak CPU utilization %")


class MemoryMetrics(BaseModel):
    """Structured memory metrics (HIGH priority).

    Focuses on RSS (Resident Set Size) which represents actual physical memory used.
    """

    peak_rss_mb: float = Field(default=0.0, ge=0, description="Peak RSS in megabytes")
    avg_rss_mb: float = Field(default=0.0, ge=0, description="Average RSS in megabytes")


class DiskIOMetrics(BaseModel):
    """Structured disk I/O metrics (CRITICAL + HIGH priority).

    Primary focus for evaluating disk-based ANN algorithms like DiskANN and SPANN.
    All metrics sourced from cgroups v2 io.stat for accuracy.
    """

    # CRITICAL: IOPS metrics (operations per second)
    avg_read_iops: float = Field(default=0.0, ge=0, description="Average read IOPS from cgroups")
    avg_write_iops: float = Field(default=0.0, ge=0, description="Average write IOPS from cgroups")

    # CRITICAL: Throughput metrics
    avg_read_throughput_mbps: float = Field(
        default=0.0, ge=0, description="Average read throughput in MB/s"
    )
    avg_write_throughput_mbps: float = Field(
        default=0.0, ge=0, description="Average write throughput in MB/s"
    )

    # HIGH: Page-level metrics (4KB pages)
    total_pages_read: int = Field(default=0, ge=0, description="Total 4KB pages read")
    total_pages_written: int = Field(default=0, ge=0, description="Total 4KB pages written")
    pages_per_query: float | None = Field(
        default=None, ge=0, description="Average pages read per query (search phase)"
    )


class LatencyMetrics(BaseModel):
    """Latency distribution metrics (HIGH priority).

    Standard percentiles for understanding query latency characteristics.
    """

    mean_ms: float = Field(default=0.0, ge=0, description="Mean query latency in ms")
    p50_ms: float = Field(default=0.0, ge=0, description="Median (p50) latency in ms")
    p95_ms: float = Field(default=0.0, ge=0, description="95th percentile latency in ms")
    p99_ms: float = Field(default=0.0, ge=0, description="99th percentile latency in ms")


class PhaseResult(BaseModel):
    """Result from a single benchmark phase (build or search)."""

    phase: str = Field(..., description="Phase name: 'build' or 'search'")
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)
    duration_seconds: float = Field(ge=0)
    resources: ResourceSummary
    output: dict[str, Any] = Field(default_factory=dict, description="Algorithm output")


class BenchmarkResult(BaseModel):
    """Complete result from benchmarking a single algorithm on a dataset.

    Uses structured metrics schemas for organized, type-safe metric access.
    This is the primary output format for benchmark runs.
    """

    # Identification
    run_id: UUID = Field(default_factory=uuid4, description="Unique run identifier for log correlation")
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

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation.

        Flattens nested metric objects into prefixed column names:
        - cpu.avg_cpu_percent -> cpu_avg_cpu_percent
        - disk_io.avg_read_iops -> disk_io_avg_read_iops
        """
        data: dict[str, Any] = {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "timestamp": self.timestamp.isoformat(),
            "recall": self.recall,
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

        # Add hyperparameters as JSON string for CSV compatibility
        data["hyperparameters"] = self.hyperparameters

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
        search_args: dict[str, Any] = Field(default_factory=dict)

    class SearchOutput(BaseModel):
        """Expected output from search phase."""

        status: str
        total_queries: int = Field(ge=0)
        total_time_seconds: float = Field(ge=0)
        qps: float = Field(ge=0)
        recall: float | None = Field(default=None, ge=0, le=1)
        mean_latency_ms: float = Field(ge=0)
        p50_latency_ms: float | None = Field(default=None, ge=0)
        p95_latency_ms: float | None = Field(default=None, ge=0)
        p99_latency_ms: float | None = Field(default=None, ge=0)
        error_message: str | None = Field(default=None)
