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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
        )
        self.results_storage = ResultsStorage(self.results_dir)

    def run(self) -> list[BenchmarkResult]:
        """Run the complete benchmark suite.

        Returns:
            List of BenchmarkResult objects for all algorithm/dataset pairs
        """
        results: list[BenchmarkResult] = []
        algorithms = self.config.enabled_algorithms

        logger.info(
            f"Starting benchmark: {len(algorithms)} algorithms, "
            f"{len(self.config.datasets)} datasets"
        )

        # Cache loaded datasets to avoid reloading
        dataset_cache: dict[str, tuple] = {}

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
                logger.info(f"Benchmarking: {algo_config.name} on {dataset_config.name}")

                # Load dataset from cache or prepare it
                if dataset_config.name not in dataset_cache:
                    try:
                        dataset_cache[dataset_config.name] = self._prepare_dataset(dataset_config)
                    except Exception as e:
                        logger.error(f"Failed to load dataset {dataset_config.name}: {e}")
                        continue

                base_vectors, query_vectors, ground_truth = dataset_cache[dataset_config.name]

                # Expand parameter sweeps (e.g., ef: [50, 100, 200] -> 3 runs)
                search_param_combos = expand_sweep_params(algo_config.search.args)
                n_combos = len(search_param_combos)
                if n_combos > 1:
                    logger.info(f"  Running {n_combos} parameter combinations for sweep")

                for param_idx, search_params in enumerate(search_param_combos):
                    # Create modified config with expanded params
                    if n_combos > 1:
                        sweep_info = ", ".join(f"{k}={v}" for k, v in search_params.items())
                        logger.info(f"  [{param_idx + 1}/{n_combos}] {sweep_info}")

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
                            f"Completed: {algo_config.name} - "
                            f"recall={result.recall:.4f}, qps={result.qps:.1f}"
                            if result.recall and result.qps
                            else f"Completed: {algo_config.name}"
                        )
                    except Exception as e:
                        logger.error(f"Benchmark failed for {algo_config.name}: {e}")
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
        """
        # Check if dataset exists
        base_path = (
            config.base_path
            if config.base_path.is_absolute()
            else self.data_dir / config.base_path
        )


        if not base_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {base_path}\n"
                f"Please download the dataset first:\n"
                f"  uv run python library/datasets/download.py --dataset {config.name} "
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
                    "search": search_params_override if search_params_override else algo_config.search.args,
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
        )

        return PhaseResult(
            phase="build",
            success=container_result.success,
            error_message=container_result.error_message,
            duration_seconds=container_result.duration_seconds,
            resources=resources,
            output=container_result.output,
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

        search_config: dict[str, Any] = {
            "index_path": f"/data/index/{algo_config.name}/{dataset_config.name}",
            "queries_path": f"/data/{dataset_config.name}/queries.npy",
            "k": algo_config.search.k,
            "search_args": search_args,
            "dimension": dataset_config.dimension,
            "metric": dataset_config.distance_metric.value,
        }

        if gt_path is not None:
            search_config["ground_truth_path"] = f"/data/{dataset_config.name}/ground_truth.npy"

        container_result, resources = self.container_runner.run_phase(
            algorithm=algo_config,
            mode="search",
            config=search_config,
            timeout_seconds=algo_config.search.timeout_seconds,
        )

        return PhaseResult(
            phase="search",
            success=container_result.success,
            error_message=container_result.error_message,
            duration_seconds=container_result.duration_seconds,
            resources=resources,
            output=container_result.output,
        )

    def _aggregate_results(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        build_result: PhaseResult,
        search_result: PhaseResult,
    ) -> BenchmarkResult:
        """Aggregate build and search results into a single BenchmarkResult.

        Populates structured metrics from both phase resources and container output.
        """
        search_output = search_result.output
        build_output = build_result.output
        build_res = build_result.resources
        search_res = search_result.resources

        # Get number of queries for per-query metrics
        num_queries = search_output.get("total_queries", 0)

        # Aggregate CPU metrics (use search phase as primary)
        cpu = CPUMetrics(
            cpu_time_total_seconds=search_res.cpu_time_total_seconds,
            avg_cpu_percent=search_res.avg_cpu_percent,
            peak_cpu_percent=max(build_res.peak_cpu_percent, search_res.peak_cpu_percent),
        )

        # Aggregate Memory metrics
        memory = MemoryMetrics(
            peak_rss_mb=max(build_res.peak_memory_mb, search_res.peak_memory_mb),
            avg_rss_mb=search_res.avg_memory_mb,
        )

        # Aggregate Disk I/O metrics (CRITICAL)
        total_read_bytes = search_res.total_blkio_read_mb * 1024 * 1024
        total_write_bytes = search_res.total_blkio_write_mb * 1024 * 1024
        total_pages_read = int(total_read_bytes / 4096)
        total_pages_written = int(total_write_bytes / 4096)

        disk_io = DiskIOMetrics(
            avg_read_iops=search_res.avg_read_iops,
            avg_write_iops=search_res.avg_write_iops,
            avg_read_throughput_mbps=(
                search_res.total_blkio_read_mb / search_res.duration_seconds
                if search_res.duration_seconds > 0
                else 0.0
            ),
            avg_write_throughput_mbps=(
                search_res.total_blkio_write_mb / search_res.duration_seconds
                if search_res.duration_seconds > 0
                else 0.0
            ),
            total_pages_read=total_pages_read,
            total_pages_written=total_pages_written,
            pages_per_query=(
                total_pages_read / num_queries if num_queries > 0 else None
            ),
        )

        # Latency metrics from container output
        latency = LatencyMetrics(
            mean_ms=search_output.get("mean_latency_ms", 0.0),
            p50_ms=search_output.get("p50_latency_ms", 0.0),
            p95_ms=search_output.get("p95_latency_ms", 0.0),
            p99_ms=search_output.get("p99_latency_ms", 0.0),
        )

        # Combine hyperparameters
        hyperparameters = {
            "build": algo_config.build.args,
            "search": algo_config.search.args,
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
