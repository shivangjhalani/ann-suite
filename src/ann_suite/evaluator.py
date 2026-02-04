"""Core benchmark evaluator orchestrating the entire benchmark pipeline.

This module is the heart of the ANN benchmarking suite, coordinating:
- Dataset loading and preparation
- Algorithm container execution
- Resource monitoring
- Result collection and aggregation

The evaluator ensures fair comparison by:
- Running each algorithm in isolated containers
- Consistent volume mounting for disk-based algorithms
- Comprehensive resource monitoring during execution
"""

from __future__ import annotations

import itertools
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ann_suite.core.constants import STANDARD_PAGE_SIZE
from ann_suite.core.schemas import (
    AlgorithmConfig,
    BenchmarkConfig,
    BenchmarkResult,
    CPUMetrics,
    DatasetConfig,
    DiskIOMetrics,
    LatencyMetrics,
    MemoryMetrics,
    PhaseResult,
    TimeBases,
)
from ann_suite.datasets.loader import DatasetLoader
from ann_suite.results.storage import ResultsStorage
from ann_suite.runners.container_runner import ContainerRunner

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def expand_sweep_params(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued parameters into all combinations.

    Supports parameter sweeps by expanding list values into cartesian product.

    Example:
        >>> expand_sweep_params({"ef": [50, 100], "num_threads": 4})
        [{"ef": 50, "num_threads": 4}, {"ef": 100, "num_threads": 4}]

    Args:
        args: Dictionary of parameters, where some values may be lists

    Returns:
        List of dictionaries with all combinations of list values
    """
    if not args:
        return [{}]

    # Separate list-valued params from scalar params
    list_keys = []
    list_values = []
    scalar_params = {}

    for key, value in args.items():
        if isinstance(value, list) and len(value) > 0:
            list_keys.append(key)
            list_values.append(value)
        else:
            scalar_params[key] = value

    # If no list params, return single dict
    if not list_keys:
        return [args.copy()]

    # Generate cartesian product of all list values
    combinations = []
    for combo in itertools.product(*list_values):
        params = scalar_params.copy()
        for key, val in zip(list_keys, combo, strict=True):
            params[key] = val
        combinations.append(params)

    return combinations


class BenchmarkEvaluator:
    """Main evaluator class for running ANN benchmarks.

    This class orchestrates the complete benchmarking pipeline:
    1. Load and prepare datasets
    2. For each algorithm + dataset pair:
       - Pull Docker image
       - Run build phase with monitoring
       - Run search phase with monitoring
       - Collect and aggregate results
    3. Store results in multiple formats

    Example:
        ```python
        config = load_config("config.yaml")
        evaluator = BenchmarkEvaluator(config)
        results = evaluator.run()
        ```
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the evaluator with benchmark configuration.

        Args:
            config: Complete benchmark configuration
        """
        self.config = config
        self.data_dir = Path(config.data_dir).resolve()
        self.index_dir = Path(config.index_dir).resolve()
        self.results_dir = Path(config.results_dir).resolve()
        self._run_id: str = ""  # Set properly when run() is called

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset_loader = DatasetLoader(self.data_dir)
        self.container_runner = ContainerRunner(
            data_dir=self.data_dir,
            index_dir=self.index_dir,
            results_dir=self.results_dir,
            monitor_interval_ms=config.monitor_interval_ms,
            include_raw_samples=config.include_raw_samples,
        )
        self.results_storage = ResultsStorage(self.results_dir)

    def run(self) -> list[BenchmarkResult]:
        """Run the complete benchmark suite.

        Returns:
            List of BenchmarkResult objects for all algorithm/dataset pairs
        """
        results: list[BenchmarkResult] = []
        algorithms = self.config.enabled_algorithms

        # Generate run_id for log correlation with stored results
        # Store as instance variable so it can be accessed by helper methods
        self._run_id = str(uuid.uuid4())[:8]

        logger.info(
            f"[{self._run_id}] Starting benchmark: {len(algorithms)} algorithms, "
            f"{len(self.config.datasets)} datasets"
        )

        # Cache loaded datasets to avoid reloading
        dataset_cache: dict[
            str, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32] | None]
        ] = {}

        for algo_config in algorithms:
            # Filter datasets for this algorithm
            if algo_config.datasets:
                algo_datasets = [d for d in self.config.datasets if d.name in algo_config.datasets]
                if not algo_datasets:
                    logger.warning(
                        f"Algorithm {algo_config.name} specifies datasets {algo_config.datasets} "
                        f"but none match configured datasets"
                    )
                    continue
            else:
                algo_datasets = self.config.datasets  # All datasets

            for dataset_config in algo_datasets:
                logger.info(
                    f"[{self._run_id}] Benchmarking: {algo_config.name} on {dataset_config.name}"
                )

                # Load dataset from cache or prepare it
                if dataset_config.name not in dataset_cache:
                    try:
                        dataset_cache[dataset_config.name] = self._prepare_dataset(dataset_config)
                    except Exception as e:
                        logger.error(
                            f"[{self._run_id}] Failed to load dataset {dataset_config.name}: {e}"
                        )
                        continue

                base_vectors, query_vectors, ground_truth = dataset_cache[dataset_config.name]

                # Expand parameter sweeps (e.g., ef: [50, 100, 200] -> 3 runs)
                search_param_combos = expand_sweep_params(algo_config.search.args)
                n_combos = len(search_param_combos)
                if n_combos > 1:
                    logger.info(
                        f"[{self._run_id}]   Running {n_combos} parameter combinations for sweep"
                    )

                for param_idx, search_params in enumerate(search_param_combos):
                    # Create modified config with expanded params
                    if n_combos > 1:
                        sweep_info = ", ".join(f"{k}={v}" for k, v in search_params.items())
                        logger.info(f"[{self._run_id}]   [{param_idx + 1}/{n_combos}] {sweep_info}")

                    try:
                        result = self._run_single_benchmark(
                            algo_config,
                            dataset_config,
                            base_vectors,
                            query_vectors,
                            ground_truth,
                            search_params_override=search_params,
                        )
                        results.append(result)
                        logger.info(
                            f"[{self._run_id}] Completed: {algo_config.name} - "
                            f"recall={result.recall:.4f}, qps={result.qps:.1f}"
                            if result.recall and result.qps
                            else f"[{self._run_id}] Completed: {algo_config.name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[{self._run_id}] Benchmark failed for {algo_config.name}: {e}"
                        )
                        # Create a failed result
                        results.append(
                            BenchmarkResult(
                                algorithm=algo_config.name,
                                dataset=dataset_config.name,
                                hyperparameters={
                                    "build": algo_config.build.args,
                                    "search": search_params,
                                },
                            )
                        )

        # Store results
        if results:
            self.results_storage.save(results, run_name=self.config.name)

        return results

    def _prepare_dataset(
        self, config: DatasetConfig
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32] | None]:
        """Prepare dataset for benchmarking.

        Loads vectors and optionally saves them in container-accessible paths.

        If cached .npy files exist in the dataset directory (base.npy, queries.npy,
        ground_truth.npy), load them directly via mmap to avoid reloading the
        original formats and reduce memory usage.
        """
        dataset_dir = self.data_dir / config.name
        cached_base = dataset_dir / "base.npy"
        cached_queries = dataset_dir / "queries.npy"
        cached_gt = dataset_dir / "ground_truth.npy"

        # If cached .npy files exist, load via mmap to avoid reloading original formats
        if cached_base.exists() and cached_queries.exists():
            logger.info(f"Loading cached dataset from {dataset_dir}")
            base_vectors: NDArray[np.float32] = np.load(cached_base, mmap_mode="r")
            query_vectors: NDArray[np.float32] = np.load(cached_queries, mmap_mode="r")
            ground_truth: NDArray[np.int32] | None = None
            if cached_gt.exists():
                ground_truth = np.load(cached_gt, mmap_mode="r")
            logger.info(
                f"Loaded {base_vectors.shape[0]} base vectors, "
                f"{query_vectors.shape[0]} queries from cache (mmap)"
            )
            return base_vectors, query_vectors, ground_truth

        # Fall back to loading from original source
        # Check if dataset exists
        base_path = (
            config.base_path if config.base_path.is_absolute() else self.data_dir / config.base_path
        )

        if not base_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {base_path}\n"
                f"Please download the dataset first:\n"
                f"  uv run ann-suite download --dataset {config.name} "
                f"--output {self.data_dir}"
            )

        return self.dataset_loader.load(config)

    def _run_single_benchmark(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        base_vectors: NDArray[np.float32],
        query_vectors: NDArray[np.float32],
        ground_truth: NDArray[np.int32] | None,
        search_params_override: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run a single algorithm/dataset benchmark.

        Args:
            search_params_override: Optional dict to override search.args for parameter sweeps

        Executes build and search phases, collecting metrics throughout.
        """
        # Ensure image is available
        if not self.container_runner.pull_image(algo_config.docker_image):
            raise RuntimeError(f"Failed to pull image: {algo_config.docker_image}")

        # Prepare paths (relative to container's /data mount)
        dataset_dir = self.data_dir / dataset_config.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save vectors if needed (for container access)
        base_path = dataset_dir / "base.npy"
        queries_path = dataset_dir / "queries.npy"
        gt_path = dataset_dir / "ground_truth.npy"

        if not base_path.exists():
            np.save(base_path, base_vectors)
        if not queries_path.exists():
            np.save(queries_path, query_vectors)
        if ground_truth is not None and not gt_path.exists():
            np.save(gt_path, ground_truth)

        # Create index directory for this algorithm
        algo_index_dir = self.index_dir / algo_config.name / dataset_config.name
        algo_index_dir.mkdir(parents=True, exist_ok=True)

        # Build phase
        build_result = self._run_build_phase(
            algo_config,
            dataset_config,
            base_path,
            algo_index_dir,
        )

        # Check if build succeeded before attempting search
        if not build_result.success:
            logger.warning(
                f"Build phase failed for {algo_config.name}, skipping search phase: "
                f"{build_result.error_message}"
            )
            # Return result with only build phase (no search attempted)
            return BenchmarkResult(
                algorithm=algo_config.name,
                dataset=dataset_config.name,
                build_result=build_result,
                hyperparameters={
                    "build": algo_config.build.args,
                    "search": search_params_override
                    if search_params_override
                    else algo_config.search.args,
                },
            )

        # Search phase (only if build succeeded)
        search_result = self._run_search_phase(
            algo_config,
            dataset_config,
            algo_index_dir,
            queries_path,
            gt_path if ground_truth is not None else None,
            search_params_override=search_params_override,
        )

        # Aggregate results
        return self._aggregate_results(
            algo_config,
            dataset_config,
            build_result,
            search_result,
            search_params_override=search_params_override,
        )

    def _run_build_phase(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        base_path: Path,
        index_dir: Path,
    ) -> PhaseResult:
        """Run the index building phase."""
        # Build config for container
        build_config = {
            "dataset_path": f"/data/{dataset_config.name}/base.npy",
            "index_path": f"/data/index/{algo_config.name}/{dataset_config.name}",
            "dimension": dataset_config.dimension,
            "metric": dataset_config.distance_metric.value,
            "build_args": algo_config.build.args,
        }

        container_result, resources = self.container_runner.run_phase(
            algorithm=algo_config,
            mode="build",
            config=build_config,
            timeout_seconds=algo_config.build.timeout_seconds,
            run_id=self._run_id,
        )

        # Create time bases for build phase
        time_bases = TimeBases(
            container_duration_seconds=container_result.duration_seconds,
            sample_span_seconds=resources.duration_seconds,
        )

        return PhaseResult(
            phase="build",
            success=container_result.success,
            error_message=container_result.error_message,
            duration_seconds=container_result.duration_seconds,
            resources=resources,
            output=container_result.output,
            time_bases=time_bases,
            stdout_path=container_result.stdout_path,
            stderr_path=container_result.stderr_path,
        )

    def _run_search_phase(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        index_dir: Path,
        queries_path: Path,
        gt_path: Path | None,
        search_params_override: dict[str, Any] | None = None,
    ) -> PhaseResult:
        """Run the search/query phase.

        Args:
            search_params_override: Optional dict to override search.args for parameter sweeps
        """
        # Use override if provided (for parameter sweeps), otherwise use config
        search_args = search_params_override if search_params_override else algo_config.search.args

        # Get warmup configuration
        warmup_config = algo_config.search.warmup

        search_config: dict[str, Any] = {
            "index_path": f"/data/index/{algo_config.name}/{dataset_config.name}",
            "queries_path": f"/data/{dataset_config.name}/queries.npy",
            "k": algo_config.search.k,
            "search_args": search_args,
            "dimension": dataset_config.dimension,
            "metric": dataset_config.distance_metric.value,
            "batch_mode": algo_config.search.batch_mode,
            # Warmup configuration
            "cache_warmup_queries": warmup_config.cache_warmup_queries,
        }

        if gt_path is not None:
            search_config["ground_truth_path"] = f"/data/{dataset_config.name}/ground_truth.npy"

        # Log warmup configuration if non-default
        if warmup_config.cache_warmup_queries > 0:
            logger.info(
                f"[{self._run_id}] Cache warming enabled: {warmup_config.cache_warmup_queries} "
                "untimed queries before benchmark"
            )

        container_result, resources = self.container_runner.run_phase(
            algorithm=algo_config,
            mode="search",
            config=search_config,
            timeout_seconds=algo_config.search.timeout_seconds,
            run_id=self._run_id,
        )

        # Create time bases from container result and algorithm output
        # Support both old "load_" and new "warmup_" field names for backward compatibility
        warmup_duration = container_result.output.get(
            "warmup_duration_seconds", container_result.output.get("load_duration_seconds")
        )
        time_bases = TimeBases(
            container_duration_seconds=container_result.duration_seconds,
            sample_span_seconds=resources.duration_seconds,
            warmup_duration_seconds=warmup_duration,
            query_duration_seconds=container_result.output.get("total_time_seconds"),
            query_start_timestamp=container_result.output.get("query_start_timestamp"),
            query_end_timestamp=container_result.output.get("query_end_timestamp"),
        )

        return PhaseResult(
            phase="search",
            success=container_result.success,
            error_message=container_result.error_message,
            duration_seconds=container_result.duration_seconds,
            resources=resources,
            warmup_resources=container_result.warmup_resources,
            output=container_result.output,
            time_bases=time_bases,
            stdout_path=container_result.stdout_path,
            stderr_path=container_result.stderr_path,
        )

    def _aggregate_results(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        build_result: PhaseResult,
        search_result: PhaseResult,
        search_params_override: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Aggregate build and search results into a single BenchmarkResult.

        Populates structured metrics from both phase resources and container output.
        Metrics are clearly separated into three phases:
        - BUILD: Index construction
        - WARMUP: Index loading/initialization (before queries)
        - SEARCH: Query execution (primary benchmark metric)

        If the search phase failed, quality metrics (recall, qps) are set to None,
        and latency/resource metrics are zeroed to avoid emitting invalid data.
        """
        build_output = build_result.output
        build_res = build_result.resources

        # Handle search failure: return result with build info but no search metrics
        if not search_result.success:
            logger.warning(
                f"[{self._run_id}] Search phase failed for {algo_config.name}: "
                f"{search_result.error_message}. Quality metrics will be None."
            )

            # Combine hyperparameters
            hyperparameters = {
                "build": algo_config.build.args,
                "search": search_params_override
                if search_params_override
                else algo_config.search.args,
                "k": algo_config.search.k,
            }

            # Return result with build metrics but empty/None search metrics
            return BenchmarkResult(
                algorithm=algo_config.name,
                dataset=dataset_config.name,
                timestamp=datetime.now(),
                build_result=build_result,
                search_result=search_result,
                # Build-only CPU metrics
                cpu=CPUMetrics(
                    build_cpu_time_seconds=build_res.cpu_time_total_seconds,
                    build_peak_cpu_percent=build_res.peak_cpu_percent,
                    # Search metrics zeroed due to failure
                    warmup_cpu_time_seconds=0.0,
                    warmup_peak_cpu_percent=0.0,
                    search_cpu_time_seconds=0.0,
                    search_avg_cpu_percent=0.0,
                    search_peak_cpu_percent=0.0,
                    search_cpu_time_per_query_ms=0.0,
                ),
                # Build-only Memory metrics
                memory=MemoryMetrics(
                    build_peak_rss_mb=build_res.peak_memory_mb,
                    warmup_peak_rss_mb=0.0,
                    search_peak_rss_mb=0.0,
                    search_avg_rss_mb=0.0,
                ),
                # Empty Disk I/O metrics (use defaults for metadata fields)
                disk_io=DiskIOMetrics(
                    warmup_read_mb=0.0,
                    warmup_write_mb=0.0,
                    search_avg_read_iops=0.0,
                    search_avg_write_iops=0.0,
                    search_avg_read_throughput_mbps=0.0,
                    search_avg_write_throughput_mbps=0.0,
                    search_total_read_mb=0.0,
                    search_total_pages_read=0,
                    search_total_pages_written=0,
                    search_pages_per_query=None,
                    # Use schema defaults for metadata (physical_block_size defaults to 4096)
                    sample_count=0,
                ),
                # Empty Latency metrics
                latency=LatencyMetrics(
                    mean_ms=0.0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                ),
                # Quality metrics are None for failed search
                recall=None,
                qps=None,
                # Build summary
                total_build_time_seconds=build_result.duration_seconds,
                index_size_bytes=build_output.get("index_size_bytes"),
                # Configuration
                hyperparameters=hyperparameters,
            )

        # Search succeeded - proceed with normal metric aggregation
        search_output = search_result.output
        search_res = search_result.resources

        # Get number of queries and query duration for per-query metrics
        num_queries = search_output.get("total_queries", 0)
        query_duration = search_output.get("total_time_seconds", 0.0)

        # Calculate CPU time per query
        cpu_time_per_query_ms = 0.0
        if num_queries > 0 and search_res.cpu_time_total_seconds > 0:
            cpu_time_per_query_ms = (search_res.cpu_time_total_seconds * 1000.0) / num_queries

        # Log sample adequacy warning
        if search_res.sample_count < 10:
            logger.warning(
                f"Only {search_res.sample_count} samples collected during search - "
                "metrics may be unreliable. Consider increasing run duration."
            )

        # Get warmup phase resources (index loading before queries)
        # Only include if warmup metrics collection is enabled
        collect_warmup = algo_config.search.warmup.collect_metrics
        warmup_res = search_result.warmup_resources if collect_warmup else None

        # Aggregate CPU metrics (separated by phase: BUILD, WARMUP, SEARCH)
        cpu = CPUMetrics(
            # BUILD phase metrics
            build_cpu_time_seconds=build_res.cpu_time_total_seconds,
            build_peak_cpu_percent=build_res.peak_cpu_percent,
            # WARMUP phase metrics (index loading)
            warmup_cpu_time_seconds=warmup_res.cpu_time_total_seconds if warmup_res else 0.0,
            warmup_peak_cpu_percent=warmup_res.peak_cpu_percent if warmup_res else 0.0,
            # SEARCH phase metrics (primary benchmark focus)
            search_cpu_time_seconds=search_res.cpu_time_total_seconds,
            search_avg_cpu_percent=search_res.avg_cpu_percent,
            search_peak_cpu_percent=search_res.peak_cpu_percent,
            search_cpu_time_per_query_ms=cpu_time_per_query_ms,
        )

        # Calculate page cache hit ratio from page fault counters
        # pgfault = total page faults (major + minor)
        # pgmajfault = major page faults (disk reads)
        # hit_ratio = 1 - (major_faults / total_faults)
        page_cache_hit_ratio = None
        if search_res.pgfault_delta > 0:
            page_cache_hit_ratio = 1.0 - (search_res.pgmajfault_delta / search_res.pgfault_delta)
            # Clamp to valid range [0, 1] to handle edge cases
            page_cache_hit_ratio = max(0.0, min(1.0, page_cache_hit_ratio))

        # Aggregate Memory metrics (separated by phase: BUILD, WARMUP, SEARCH)
        memory = MemoryMetrics(
            build_peak_rss_mb=build_res.peak_memory_mb,
            warmup_peak_rss_mb=warmup_res.peak_memory_mb if warmup_res else 0.0,
            search_peak_rss_mb=search_res.peak_memory_mb,
            search_avg_rss_mb=search_res.avg_memory_mb,
            search_major_faults=search_res.pgmajfault_delta,
            search_page_cache_hit_ratio=page_cache_hit_ratio,
        )

        # Aggregate Disk I/O metrics (CRITICAL for disk-based algorithms)
        # Use query_duration as the CONSISTENT time base for all rate metrics
        io_time_base = query_duration if query_duration > 0 else search_res.duration_seconds

        # Warn if we can't compute throughput
        if io_time_base <= 0:
            logger.warning(
                "No valid time base for throughput calculation (io_time_base=0). "
                "Container may have exited too quickly for metrics collection. "
                "Throughput metrics will be reported as 0."
            )

        search_total_read_mb = search_res.total_blkio_read_mb
        search_total_write_mb = search_res.total_blkio_write_mb
        search_total_read_bytes = search_total_read_mb * 1024 * 1024
        search_total_write_bytes = search_total_write_mb * 1024 * 1024

        # Calculate pages using STANDARD 4KB page size (not physical block size)
        search_total_pages_read = int(search_total_read_bytes / STANDARD_PAGE_SIZE)
        search_total_pages_written = int(search_total_write_bytes / STANDARD_PAGE_SIZE)

        # Calculate warmup phase I/O
        warmup_read_mb = warmup_res.total_blkio_read_mb if warmup_res else 0.0
        warmup_write_mb = warmup_res.total_blkio_write_mb if warmup_res else 0.0

        # Use raw deltas from collector for accurate IOPS (avoids lossy reconstruction)
        if io_time_base > 0:
            search_avg_read_iops = search_res.total_read_ops / io_time_base
            search_avg_write_iops = search_res.total_write_ops / io_time_base
        else:
            search_avg_read_iops = 0.0
            search_avg_write_iops = 0.0

        # Compute service time proxy metrics (bytes per operation + service time)
        search_avg_bytes_per_read_op: float | None = None
        search_avg_bytes_per_write_op: float | None = None
        search_avg_read_service_time_ms: float | None = None
        search_avg_write_service_time_ms: float | None = None
        if search_res.total_read_ops > 0:
            search_avg_bytes_per_read_op = search_total_read_bytes / search_res.total_read_ops
            if search_res.total_read_usec > 0:
                search_avg_read_service_time_ms = (
                    search_res.total_read_usec / search_res.total_read_ops
                ) / 1000.0
        if search_res.total_write_ops > 0:
            search_avg_bytes_per_write_op = search_total_write_bytes / search_res.total_write_ops
            if search_res.total_write_usec > 0:
                search_avg_write_service_time_ms = (
                    search_res.total_write_usec / search_res.total_write_ops
                ) / 1000.0

        # Tail metrics from per-interval samples (computed by collector)
        search_p95_read_iops = search_res.p95_read_iops
        search_max_read_iops = search_res.max_read_iops
        search_p95_read_mbps = search_res.p95_read_mbps
        search_max_read_mbps = search_res.max_read_mbps
        search_p95_read_service_time_ms = search_res.p95_read_service_time_ms
        search_max_read_service_time_ms = search_res.max_read_service_time_ms

        # PSI stall metrics
        search_io_stall_percent: float | None = None
        if io_time_base > 0 and search_res.io_pressure_some_total_usec > 0:
            search_io_stall_percent = (
                search_res.io_pressure_some_total_usec / (io_time_base * 1_000_000)
            ) * 100.0

        # Per-device summary (top device only)
        per_device_summary: list[dict[str, Any]] | None = None
        if search_res.top_read_device:
            device_name = str(search_res.top_read_device.get("device", ""))
            read_bytes = float(search_res.top_read_device.get("total_read_bytes", 0))
            write_bytes = float(search_res.top_read_device.get("total_write_bytes", 0))
            read_ops = int(search_res.top_read_device.get("total_read_ops", 0))
            write_ops = int(search_res.top_read_device.get("total_write_ops", 0))
            if device_name:
                per_device_summary = [
                    {
                        "device": device_name,
                        "read_mb": read_bytes / (1024 * 1024),
                        "write_mb": write_bytes / (1024 * 1024),
                        "read_ops": read_ops,
                        "write_ops": write_ops,
                    }
                ]

        search_major_faults_per_query: float | None = None
        if num_queries > 0:
            search_major_faults_per_query = search_res.pgmajfault_delta / num_queries
        search_major_faults_per_second: float | None = None
        if io_time_base > 0:
            search_major_faults_per_second = search_res.pgmajfault_delta / io_time_base

        search_file_cache_avg_mb = search_res.avg_file_bytes / (1024 * 1024)
        search_file_cache_peak_mb = search_res.peak_file_bytes / (1024 * 1024)

        warmup_io_stall_percent: float | None = None
        warmup_major_faults_per_second: float | None = None
        warmup_file_cache_avg_mb: float | None = None
        warmup_file_cache_peak_mb: float | None = None
        if warmup_res:
            warmup_time_base = warmup_res.duration_seconds
            if warmup_time_base > 0 and warmup_res.io_pressure_some_total_usec > 0:
                warmup_io_stall_percent = (
                    warmup_res.io_pressure_some_total_usec / (warmup_time_base * 1_000_000)
                ) * 100.0
            if warmup_time_base > 0:
                warmup_major_faults_per_second = warmup_res.pgmajfault_delta / warmup_time_base
            warmup_file_cache_avg_mb = warmup_res.avg_file_bytes / (1024 * 1024)
            warmup_file_cache_peak_mb = warmup_res.peak_file_bytes / (1024 * 1024)

        disk_io = DiskIOMetrics(
            # WARMUP phase I/O
            warmup_read_mb=warmup_read_mb,
            warmup_write_mb=warmup_write_mb,
            warmup_io_stall_percent=warmup_io_stall_percent,
            warmup_major_faults_per_second=warmup_major_faults_per_second,
            warmup_file_cache_avg_mb=warmup_file_cache_avg_mb,
            warmup_file_cache_peak_mb=warmup_file_cache_peak_mb,
            # SEARCH phase IOPS (using consistent query_duration time base)
            search_avg_read_iops=search_avg_read_iops,
            search_avg_write_iops=search_avg_write_iops,
            # SEARCH phase throughput (using consistent query_duration time base)
            search_avg_read_throughput_mbps=(
                search_total_read_mb / io_time_base if io_time_base > 0 else 0.0
            ),
            search_avg_write_throughput_mbps=(
                search_total_write_mb / io_time_base if io_time_base > 0 else 0.0
            ),
            # SEARCH phase page metrics (standardized 4KB pages)
            search_total_read_mb=search_total_read_mb,
            search_total_pages_read=search_total_pages_read,
            search_total_pages_written=search_total_pages_written,
            search_pages_per_query=(
                search_total_pages_read / num_queries if num_queries > 0 else None
            ),
            # Service time proxy metrics (bytes per operation)
            search_avg_bytes_per_read_op=search_avg_bytes_per_read_op,
            search_avg_bytes_per_write_op=search_avg_bytes_per_write_op,
            search_avg_read_service_time_ms=search_avg_read_service_time_ms,
            search_avg_write_service_time_ms=search_avg_write_service_time_ms,
            # Tail metrics (p95/max IOPS)
            search_p95_read_iops=search_p95_read_iops,
            search_max_read_iops=search_max_read_iops,
            search_p95_read_mbps=search_p95_read_mbps,
            search_max_read_mbps=search_max_read_mbps,
            search_p95_read_service_time_ms=search_p95_read_service_time_ms,
            search_max_read_service_time_ms=search_max_read_service_time_ms,
            # PSI stall metrics
            search_io_stall_percent=search_io_stall_percent,
            search_major_faults_per_query=search_major_faults_per_query,
            search_major_faults_per_second=search_major_faults_per_second,
            search_file_cache_avg_mb=search_file_cache_avg_mb,
            search_file_cache_peak_mb=search_file_cache_peak_mb,
            # Per-device summary
            per_device_summary=per_device_summary,
            # Metadata for transparency
            physical_block_size=search_res.block_size,
            sample_count=search_res.sample_count,
        )

        # Warn about unexpected writes during search
        if search_total_write_mb > 10:
            logger.warning(
                f"Unexpected write I/O during search: {search_total_write_mb:.1f} MB. "
                "This may indicate logging, temp files, or mmap metadata writes."
            )

        # Latency metrics from container output
        latency = LatencyMetrics(
            mean_ms=search_output.get("mean_latency_ms", 0.0),
            p50_ms=search_output.get("p50_latency_ms", 0.0),
            p95_ms=search_output.get("p95_latency_ms", 0.0),
            p99_ms=search_output.get("p99_latency_ms", 0.0),
            max_ms=search_output.get("max_latency_ms"),  # Default None if not provided
        )

        # Warn if latency percentiles appear to be estimated (all equal = batch_mode)
        if latency.mean_ms > 0 and latency.p50_ms == latency.p95_ms == latency.p99_ms:
            logger.warning(
                f"Latency percentiles are identical (p50=p95=p99={latency.p50_ms:.3f}ms). "
                "This typically indicates batch_mode=True was used, which estimates "
                "per-query latency rather than measuring it. Set batch_mode=False in "
                "search config for accurate latency distribution."
            )

        # Combine hyperparameters - use override if provided (for parameter sweeps)
        hyperparameters = {
            "build": algo_config.build.args,
            "search": search_params_override if search_params_override else algo_config.search.args,
            "k": algo_config.search.k,
        }

        return BenchmarkResult(
            algorithm=algo_config.name,
            dataset=dataset_config.name,
            timestamp=datetime.now(),
            build_result=build_result,
            search_result=search_result,
            # Structured metrics
            cpu=cpu,
            memory=memory,
            disk_io=disk_io,
            latency=latency,
            # Quality metrics
            recall=search_output.get("recall"),
            qps=search_output.get("qps"),
            # Build summary
            total_build_time_seconds=build_result.duration_seconds,
            index_size_bytes=build_output.get("index_size_bytes"),
            # Configuration
            hyperparameters=hyperparameters,
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        self.container_runner.cleanup()


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Convenience function to run a benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        List of BenchmarkResult objects
    """
    evaluator = BenchmarkEvaluator(config)
    try:
        return evaluator.run()
    finally:
        evaluator.cleanup()
