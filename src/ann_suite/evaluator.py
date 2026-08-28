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

import hashlib
import itertools
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ann_suite.core.constants import STANDARD_PAGE_SIZE
from ann_suite.core.schemas import (
    AlgorithmConfig,
    AlgorithmStats,
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


def build_combo_slug(args: dict[str, Any]) -> str:
    """Generate a deterministic, filesystem-safe slug for a build parameter combination.

    The slug uniquely identifies an index build: identical args always produce the
    same slug regardless of key order, so it can key on-disk index directories and
    the in-run build cache.

    Format: sanitized sorted "k=v" pairs (truncated) + short md5 suffix of the
    canonical JSON form. Empty args produce "default".

    Args:
        args: Build parameter dictionary

    Returns:
        Filesystem-safe slug string
    """
    if not args:
        return "default"

    canonical = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.md5(canonical.encode()).hexdigest()[:8]

    parts = []
    for key in sorted(args):
        pair = re.sub(r"[^A-Za-z0-9]+", "-", f"{key}={args[key]}").strip("-")
        parts.append(pair)
    readable = "_".join(parts)[:48].strip("_-")

    return f"{readable}-{digest}"


@dataclass
class _BuildContext:
    """Context for one index build, shared by all search points that use it.

    Attributes:
        build_result: Phase result from the build container run.
        host_index_dir: Host path where the index was written.
        container_index_path: Corresponding path inside the container (/data/index/...).
        build_params: The exact build arguments used for this index.
    """

    build_result: PhaseResult
    host_index_dir: Path
    container_index_path: str
    build_params: dict[str, Any]


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

        # Cache of built indices within this evaluator's lifetime, keyed by
        # (algorithm name, dataset name, build combo slug). Only populated when
        # build.reuse_index is enabled; lets search sweep points share one build.
        self._build_cache: dict[tuple[str, str, str], _BuildContext] = {}

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
            algo_datasets = self._datasets_for_algorithm(algo_config)
            if not algo_datasets:
                continue

            # Ensure image is available once per algorithm (not per benchmark point)
            if not self.container_runner.pull_image(algo_config.docker_image):
                error_msg = f"Failed to pull image: {algo_config.docker_image}"
                logger.error(f"[{self._run_id}] {error_msg}")
                for dataset_config in algo_datasets:
                    for search_params in expand_sweep_params(algo_config.search.args):
                        results.append(
                            self._failed_result(
                                algo_config,
                                dataset_config,
                                dict(algo_config.build.args),
                                search_params,
                            )
                        )
                continue

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

                # Save container-accessible .npy files once per (algorithm, dataset)
                base_path, queries_path, gt_path = self._prepare_dataset_files(
                    dataset_config, base_vectors, query_vectors, ground_truth
                )

                # Expand build and search sweeps into their cartesian product.
                # Each unique build combo builds its index once; every search combo
                # then runs against it (subject to build.reuse_index).
                build_param_combos = expand_sweep_params(algo_config.build.args)
                search_param_combos = expand_sweep_params(algo_config.search.args)
                n_build_combos = len(build_param_combos)
                n_search_combos = len(search_param_combos)
                if n_build_combos > 1 or n_search_combos > 1:
                    logger.info(
                        f"[{self._run_id}]   Running {n_build_combos} build x "
                        f"{n_search_combos} search parameter combinations"
                    )

                # Memoize failed builds by slug so one failed attempt fans out to
                # all of its search combos instead of re-running an expensive,
                # known-bad build for each point. Reset per (algorithm, dataset).
                failed_builds: dict[str, PhaseResult | None] = {}

                for build_params in build_param_combos:
                    slug = build_combo_slug(build_params)

                    for param_idx, search_params in enumerate(search_param_combos):
                        if slug in failed_builds:
                            results.append(
                                self._failed_result(
                                    algo_config,
                                    dataset_config,
                                    build_params,
                                    search_params,
                                    build_result=failed_builds[slug],
                                )
                            )
                            continue

                        try:
                            context = self._ensure_build(
                                algo_config, dataset_config, base_path, build_params
                            )
                        except Exception as e:
                            logger.error(
                                f"[{self._run_id}] Build failed for {algo_config.name} "
                                f"with args {build_params}: {e}"
                            )
                            failed_builds[slug] = None
                            results.append(
                                self._failed_result(
                                    algo_config, dataset_config, build_params, search_params
                                )
                            )
                            continue

                        if not context.build_result.success:
                            logger.warning(
                                f"[{self._run_id}] Build phase failed for {algo_config.name}, "
                                f"skipping search: {context.build_result.error_message}"
                            )
                            failed_builds[slug] = context.build_result
                            results.append(
                                self._failed_result(
                                    algo_config,
                                    dataset_config,
                                    build_params,
                                    search_params,
                                    build_result=context.build_result,
                                )
                            )
                            continue

                        sweep_info = ", ".join(f"{k}={v}" for k, v in search_params.items())
                        logger.info(
                            f"[{self._run_id}]   Search [{param_idx + 1}/{n_search_combos}] "
                            f"build={context.container_index_path} ({sweep_info})"
                        )

                        try:
                            result = self._run_benchmark_point(
                                algo_config,
                                dataset_config,
                                context,
                                queries_path,
                                gt_path,
                                search_params,
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
                            results.append(
                                self._failed_result(
                                    algo_config, dataset_config, build_params, search_params
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

    def _datasets_for_algorithm(self, algo_config: AlgorithmConfig) -> list[DatasetConfig]:
        """Return the datasets an algorithm should run on, honoring its dataset filter."""
        if algo_config.datasets:
            algo_datasets = [d for d in self.config.datasets if d.name in algo_config.datasets]
            if not algo_datasets:
                logger.warning(
                    f"Algorithm {algo_config.name} specifies datasets {algo_config.datasets} "
                    f"but none match configured datasets"
                )
            return algo_datasets
        return self.config.datasets  # All datasets

    def _prepare_dataset_files(
        self,
        dataset_config: DatasetConfig,
        base_vectors: NDArray[np.float32],
        query_vectors: NDArray[np.float32],
        ground_truth: NDArray[np.int32] | None,
    ) -> tuple[Path, Path, Path]:
        """Save container-accessible .npy copies of the dataset (idempotent).

        Returns:
            Tuple of (base_path, queries_path, ground_truth_path)
        """
        dataset_dir = self.data_dir / dataset_config.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        base_path = dataset_dir / "base.npy"
        queries_path = dataset_dir / "queries.npy"
        gt_path = dataset_dir / "ground_truth.npy"

        if not base_path.exists():
            np.save(base_path, base_vectors)
        if not queries_path.exists():
            np.save(queries_path, query_vectors)
        if ground_truth is not None and not gt_path.exists():
            np.save(gt_path, ground_truth)

        return base_path, queries_path, gt_path

    def _failed_result(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        build_params: dict[str, Any],
        search_params: dict[str, Any],
        build_result: PhaseResult | None = None,
    ) -> BenchmarkResult:
        """Create a placeholder result for a failed benchmark point.

        Preserves row-count semantics (one row per planned search combo) and
        records the intended hyperparameters for debugging. Errors are surfaced
        via ``build_result.error_message`` and the run log.
        """
        return BenchmarkResult(
            algorithm=algo_config.name,
            dataset=dataset_config.name,
            build_result=build_result,
            hyperparameters={
                "build": build_params,
                "search": search_params,
                "k": algo_config.search.k,
            },
        )

    def _ensure_build(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        base_path: Path,
        build_params: dict[str, Any],
    ) -> _BuildContext:
        """Ensure an index exists for this build combo, building it if necessary.

        When build.reuse_index is enabled and this exact combo was already built
        during this run, returns the cached context without rebuilding.

        Args:
            algo_config: Algorithm configuration
            dataset_config: Dataset configuration
            base_path: Host path to base vectors (.npy)
            build_params: Build arguments for this combination

        Returns:
            Build context (build result may still be unsuccessful; callers must check)
        """
        slug = build_combo_slug(build_params)
        cache_key = (algo_config.name, dataset_config.name, slug)
        host_index_dir = self.index_dir / algo_config.name / dataset_config.name / slug
        container_index_path = f"/data/index/{algo_config.name}/{dataset_config.name}/{slug}"

        if algo_config.build.reuse_index and cache_key in self._build_cache:
            logger.info(f"[{self._run_id}] Reusing index built at {host_index_dir}")
            return self._build_cache[cache_key]

        host_index_dir.mkdir(parents=True, exist_ok=True)
        build_result = self._run_build_phase(
            algo_config,
            dataset_config,
            base_path,
            host_index_dir,
            container_index_path,
            build_params,
        )

        context = _BuildContext(
            build_result=build_result,
            host_index_dir=host_index_dir,
            container_index_path=container_index_path,
            build_params=build_params,
        )
        # Cache only successful builds so failures are retried per search point
        if algo_config.build.reuse_index and build_result.success:
            self._build_cache[cache_key] = context
        return context

    def _run_benchmark_point(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        context: _BuildContext,
        queries_path: Path,
        gt_path: Path | None,
        search_params: dict[str, Any],
    ) -> BenchmarkResult:
        """Run one search point against a built index and aggregate with its build metrics.

        Caller must guarantee the build in ``context`` succeeded.
        """
        search_result = self._run_search_phase(
            algo_config,
            dataset_config,
            context.container_index_path,
            queries_path,
            gt_path,
            search_params_override=search_params,
        )

        return self._aggregate_results(
            algo_config,
            dataset_config,
            context.build_result,
            search_result,
            build_params=context.build_params,
            search_params_override=search_params,
        )

    def _run_build_phase(
        self,
        algo_config: AlgorithmConfig,
        dataset_config: DatasetConfig,
        base_path: Path,
        index_dir: Path,
        container_index_path: str,
        build_args: dict[str, Any],
    ) -> PhaseResult:
        """Run the index building phase.

        Args:
            algo_config: Algorithm configuration
            dataset_config: Dataset configuration
            base_path: Host path to base vectors
            index_dir: Host directory where the index should be written
            container_index_path: Index path as seen inside the container
            build_args: Build arguments for this specific build combination
        """
        # Build config for container
        build_config = {
            "dataset_path": f"/data/{dataset_config.name}/base.npy",
            "index_path": container_index_path,
            "dimension": dataset_config.dimension,
            "metric": dataset_config.distance_metric.value,
            "build_args": build_args,
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
        container_index_path: str,
        queries_path: Path,
        gt_path: Path | None,
        search_params_override: dict[str, Any] | None = None,
    ) -> PhaseResult:
        """Run the search/query phase.

        Args:
            algo_config: Algorithm configuration
            dataset_config: Dataset configuration
            container_index_path: Index path as seen inside the container
            queries_path: Host path to query vectors
            gt_path: Host path to ground truth (optional)
            search_params_override: Optional dict to override search.args for parameter sweeps
        """
        # Use override if provided (for parameter sweeps), otherwise use config
        search_args = search_params_override if search_params_override else algo_config.search.args

        # Get warmup configuration
        warmup_config = algo_config.search.warmup

        search_config: dict[str, Any] = {
            "index_path": container_index_path,
            "queries_path": f"/data/{dataset_config.name}/queries.npy",
            "k": algo_config.search.k,
            "query_rounds": algo_config.search.query_rounds,
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

        # Cold-start benchmarking: drop the OS page cache so index reads hit disk
        if warmup_config.drop_caches_before:
            if self.container_runner.drop_caches():
                logger.info(f"[{self._run_id}] Dropped OS page caches before search phase")
            else:
                logger.warning(
                    f"[{self._run_id}] drop_caches_before=true but the cache drop FAILED; "
                    "this search point will run with a warm page cache"
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
        build_params: dict[str, Any] | None = None,
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
        # Record the exact build combo used (may differ from algo_config.build.args
        # when sweeping build parameters)
        effective_build_params = (
            build_params if build_params is not None else dict(algo_config.build.args)
        )
        build_output = build_result.output
        build_res = build_result.resources

        # Validate build phase I/O: non-zero index size should have non-zero I/O
        index_size_bytes = build_output.get("index_size_bytes", 0)
        build_total_io_bytes = (
            (build_res.total_blkio_read_mb + build_res.total_blkio_write_mb) * 1024 * 1024
        )
        if (
            algo_config.algorithm_type.value == "disk"
            and index_size_bytes
            and index_size_bytes > 0
            and build_total_io_bytes == 0
        ):
            logger.warning(
                f"Build phase produced non-zero index ({index_size_bytes / (1024 * 1024):.1f} MB) "
                "but reported zero I/O. This is physically implausible and indicates "
                "I/O metrics collection failed to capture the actual disk writes. "
                "Possible causes: cgroups io.stat not updated, container exited before "
                "final sample, or I/O was buffered in page cache without sync."
            )

        # Handle search failure: return result with build info but no search metrics
        if not search_result.success:
            logger.warning(
                f"[{self._run_id}] Search phase failed for {algo_config.name}: "
                f"{search_result.error_message}. Quality metrics will be None."
            )

            # Combine hyperparameters
            hyperparameters = {
                "build": effective_build_params,
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
                    search_nr_throttled=0,
                    search_throttled_usec=0,
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
                    search_total_write_mb=0.0,
                    search_total_pages_read=0,
                    search_total_pages_written=0,
                    search_pages_per_query=None,
                    # Use schema defaults for metadata (physical_block_size defaults to 4096)
                    sample_count=0,
                ),
                # Empty Latency metrics
                latency=LatencyMetrics(mean_ms=0.0),
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
            # CPU throttling (from cgroups cpu.stat)
            search_nr_throttled=search_res.nr_throttled_delta,
            search_throttled_usec=search_res.throttled_usec_delta,
        )

        # Page-cache hit rate from cgroup page-fault counters.
        # pgfault = total page faults (major + minor); pgmajfault = major faults (served
        # from disk). The hit rate is the fraction of faulted accesses served without
        # disk I/O. Algorithm-internal caches that avoid faults entirely are not
        # visible at this level (those arrive via algorithm_stats.cache_hits/misses).
        #
        # NOTE: this metric is only meaningful for fault-based (mmap) workloads.
        # Algorithms that read via O_DIRECT/pread (e.g. DiskANN's StaticDiskIndex)
        # do their disk I/O without any page faults, so pgfault/pgmajfault are ~0 and a
        # "hit rate" computed from them is vacuous (a 0/0 that previously rendered as a
        # misleading 1.0). When there are no faults to judge, report None so consumers
        # fall back to the authoritative block-device metrics (pages_per_query,
        # bytes_read_per_query, IOPS) instead of a bogus 100% cache hit.
        page_cache_hit_rate: float | None = None
        if search_res.pgfault_delta > 0:
            page_cache_hit_rate = 1.0 - (search_res.pgmajfault_delta / search_res.pgfault_delta)
            # Clamp to valid range [0, 1] to handle edge cases
            page_cache_hit_rate = max(0.0, min(1.0, page_cache_hit_rate))

        # Aggregate Memory metrics (separated by phase: BUILD, WARMUP, SEARCH)
        memory = MemoryMetrics(
            build_peak_rss_mb=build_res.peak_memory_mb,
            warmup_peak_rss_mb=warmup_res.peak_memory_mb if warmup_res else 0.0,
            search_peak_rss_mb=search_res.peak_memory_mb,
            search_avg_rss_mb=search_res.avg_memory_mb,
            search_major_faults=search_res.pgmajfault_delta,
            search_page_cache_hit_rate=page_cache_hit_rate,
        )

        # Aggregate Disk I/O metrics (CRITICAL for disk-based algorithms)
        # Use cgroups sample span as the PRIMARY time base for all rate metrics.
        # This ensures consistency: numerators come from cgroups counters (measured over the
        # sample span), so denominators must use the same window. Fall back to algorithm
        # wall-clock only if cgroups duration is unavailable.
        io_time_base = (
            search_res.duration_seconds if search_res.duration_seconds > 0 else query_duration
        )

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
        # Issue #5 fix: total_read_usec/total_write_usec may be None if kernel doesn't expose rusec/wusec
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

        # PSI stall metrics. Report a measured percentage when PSI counters were
        # observed (psi_available) even if the delta is zero (0.0% stall is a real,
        # meaningful reading for an async-I/O workload that overlaps reads with
        # compute). Only report None when PSI was never observed in the window.
        search_io_stall_percent: float | None = None
        if io_time_base > 0 and search_res.psi_available:
            search_io_stall_percent = (
                search_res.io_pressure_some_total_usec / (io_time_base * 1_000_000)
            ) * 100.0

        search_major_faults_per_query: float | None = None
        if num_queries > 0:
            search_major_faults_per_query = search_res.pgmajfault_delta / num_queries
        search_major_faults_per_second: float | None = None
        if io_time_base > 0:
            search_major_faults_per_second = search_res.pgmajfault_delta / io_time_base

        search_file_cache_avg_mb = search_res.avg_file_bytes / (1024 * 1024)
        search_file_cache_peak_mb = search_res.peak_file_bytes / (1024 * 1024)

        # File cache breakdown: mapped, active, inactive
        search_file_mapped_avg_mb: float | None = None
        search_file_active_avg_mb: float | None = None
        search_file_inactive_avg_mb: float | None = None
        if search_res.avg_file_mapped_bytes > 0:
            search_file_mapped_avg_mb = search_res.avg_file_mapped_bytes / (1024 * 1024)
        if search_res.avg_active_file_bytes > 0:
            search_file_active_avg_mb = search_res.avg_active_file_bytes / (1024 * 1024)
        if search_res.avg_inactive_file_bytes > 0:
            search_file_inactive_avg_mb = search_res.avg_inactive_file_bytes / (1024 * 1024)

        # PSI io.full stall percent (complete I/O blockage)
        search_io_full_stall_percent: float | None = None
        if io_time_base > 0 and search_res.psi_available:
            search_io_full_stall_percent = (
                search_res.io_pressure_full_total_usec / (io_time_base * 1_000_000)
            ) * 100.0

        warmup_io_stall_percent: float | None = None
        warmup_major_faults_per_second: float | None = None
        warmup_file_cache_avg_mb: float | None = None
        warmup_file_cache_peak_mb: float | None = None
        if warmup_res:
            warmup_time_base = warmup_res.duration_seconds
            if warmup_time_base > 0 and warmup_res.psi_available:
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
            search_total_write_mb=search_total_write_mb,
            search_total_pages_read=search_total_pages_read,
            search_total_pages_written=search_total_pages_written,
            search_pages_per_query=(
                search_total_pages_read / num_queries if num_queries > 0 else None
            ),
            search_reads_per_query=(
                search_res.total_read_ops / num_queries if num_queries > 0 else None
            ),
            search_bytes_read_per_query=(
                search_total_read_bytes / num_queries if num_queries > 0 else None
            ),
            # Queue depth (system-wide gauge sampled per monitor interval)
            search_avg_queue_depth=search_res.avg_queue_depth,
            search_p95_queue_depth=search_res.p95_queue_depth,
            search_max_queue_depth=search_res.max_queue_depth,
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
            search_io_full_stall_percent=search_io_full_stall_percent,
            search_major_faults_per_query=search_major_faults_per_query,
            search_major_faults_per_second=search_major_faults_per_second,
            search_file_cache_avg_mb=search_file_cache_avg_mb,
            search_file_cache_peak_mb=search_file_cache_peak_mb,
            # File cache breakdown
            search_file_mapped_avg_mb=search_file_mapped_avg_mb,
            search_file_active_avg_mb=search_file_active_avg_mb,
            search_file_inactive_avg_mb=search_file_inactive_avg_mb,
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
            p50_ms=search_output.get("p50_latency_ms"),
            p95_ms=search_output.get("p95_latency_ms"),
            p99_ms=search_output.get("p99_latency_ms"),
            max_ms=search_output.get("max_latency_ms"),
        )

        # Algorithm-reported stats (optional protocol extension). Normalize the
        # well-known totals to per-query values; unknown counters pass through.
        algorithm_stats: AlgorithmStats | None = None
        raw_stats = search_output.get("stats")
        if isinstance(raw_stats, dict) and raw_stats:
            algorithm_stats = AlgorithmStats.from_output(raw_stats).with_per_query(num_queries)

        # Combine hyperparameters - use override if provided (for parameter sweeps)
        hyperparameters = {
            "build": effective_build_params,
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
            algorithm_stats=algorithm_stats,
            # Quality metrics
            recall=search_output.get("recall"),
            qps=search_output.get("qps"),
            # Build summary
            total_build_time_seconds=build_output.get(
                "build_time_seconds", build_result.duration_seconds
            ),
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
