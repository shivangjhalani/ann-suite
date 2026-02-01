"""Docker container runner for ANN algorithms.

This module manages the complete lifecycle of algorithm containers:
- Image pulling
- Container creation with proper volume mounts
- Execution with monitoring
- Result collection
- Cleanup

Critical for disk-based algorithms: Mounts host directories to /data
so disk I/O is captured accurately and indices persist on host storage.

"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker
from docker.errors import ContainerError, ImageNotFound, NotFound

from ann_suite.core.schemas import AlgorithmConfig, ResourceSummary
from ann_suite.monitoring.base import get_system_block_size
from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector

# Maximum number of log files to keep per algorithm/mode combination
MAX_LOG_FILES_PER_TYPE = 50

if TYPE_CHECKING:
    from docker.models.containers import Container

logger = logging.getLogger(__name__)


@dataclass
class ContainerResult:
    """Result from container execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    output: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    warmup_resources: ResourceSummary | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class ContainerRunner:
    """Manages Docker container lifecycle for ANN algorithm benchmarking.

    This class handles:
    - Image management (pull, verify)
    - Container creation with proper volume mounts and resource limits
    - Execution with integrated resource monitoring
    - Output collection and parsing
    - Cleanup

    Example:
        ```python
        runner = ContainerRunner(
            data_dir=Path("./data"),
            index_dir=Path("./indices"),
            results_dir=Path("./results"),
        )

        # Run build phase
        build_result, resources = runner.run_phase(
            algorithm=algo_config,
            mode="build",
            config={"dataset_path": "/data/base.npy", ...},
        )

        # Run search phase
        search_result, resources = runner.run_phase(
            algorithm=algo_config,
            mode="search",
            config={"index_path": "/data/index/", ...},
        )
        ```
    """

    def __init__(
        self,
        data_dir: Path,
        index_dir: Path,
        results_dir: Path,
        monitor_interval_ms: int = 100,
    ) -> None:
        """Initialize the container runner.

        Args:
            data_dir: Host directory containing datasets (mounted to /data)
            index_dir: Host directory for indices (mounted to /data/index)
            results_dir: Host directory for results (mounted to /results)
            monitor_interval_ms: Resource monitoring interval
        """
        self.data_dir = Path(data_dir).resolve()
        self.index_dir = Path(index_dir).resolve()
        self.results_dir = Path(results_dir).resolve()
        self.monitor_interval_ms = monitor_interval_ms

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Docker client
        self._client = docker.from_env()

        # Verify cgroups v2 is available - required for metrics collection
        if not CgroupsV2Collector.check_available():
            raise RuntimeError(
                "cgroups v2 is required for metrics collection but is not available. "
                "Ensure your system uses unified cgroup v2 hierarchy. "
                "See docs/METRICS.md for setup instructions."
            )

        self._cgroups_collector = CgroupsV2Collector(interval_ms=monitor_interval_ms)

        # Detect system block size once at initialization
        self._block_size = get_system_block_size()
        logger.info(
            f"CgroupsV2Collector available - will collect metrics (block_size={self._block_size})"
        )

        # Clean up old logs on initialization
        self._cleanup_old_logs()

    def _cleanup_old_logs(self) -> None:
        """Remove old log files to prevent disk space accumulation.

        Keeps the most recent MAX_LOG_FILES_PER_TYPE log files per algorithm/mode
        combination, removing older files.
        """
        logs_dir = self.results_dir / "logs"
        if not logs_dir.exists():
            return

        try:
            # Group log files by algorithm and mode (e.g., "HNSW_build", "DiskANN_search")
            log_groups: dict[str, list[Path]] = {}

            for log_file in logs_dir.glob("*.log"):
                # Log filename format: {algorithm}_{mode}_{phase_id}.{stdout|stderr}.log
                parts = log_file.stem.split("_")
                if len(parts) >= 2:
                    group_key = f"{parts[0]}_{parts[1]}"  # e.g., "HNSW_build"
                    if group_key not in log_groups:
                        log_groups[group_key] = []
                    log_groups[group_key].append(log_file)

            # For each group, keep only the most recent files
            files_removed = 0
            for _group_key, files in log_groups.items():
                if len(files) <= MAX_LOG_FILES_PER_TYPE:
                    continue

                # Sort by modification time (newest first)
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                # Remove oldest files beyond the limit
                for old_file in files[MAX_LOG_FILES_PER_TYPE:]:
                    try:
                        old_file.unlink()
                        files_removed += 1
                    except OSError:
                        pass

            if files_removed > 0:
                logger.debug(f"Cleaned up {files_removed} old log files from {logs_dir}")

        except Exception as e:
            logger.debug(f"Error cleaning up old logs: {e}")

    def pull_image(self, image: str, force: bool = False) -> bool:
        """Pull a Docker image if not present.

        Args:
            image: Docker image tag (e.g., 'ann-suite/hnsw:latest')
            force: Force pull even if image exists

        Returns:
            True if image is available
        """
        try:
            if not force:
                try:
                    self._client.images.get(image)
                    logger.debug(f"Image {image} already present")
                    return True
                except ImageNotFound:
                    pass

            logger.info(f"Pulling image {image}...")
            self._client.images.pull(image)
            logger.info(f"Successfully pulled {image}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False

    def run_phase(
        self,
        algorithm: AlgorithmConfig,
        mode: str,
        config: dict[str, Any],
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> tuple[ContainerResult, ResourceSummary]:
        """Run a benchmark phase (build or search) in a container.

        Args:
            algorithm: Algorithm configuration
            mode: Phase mode ('build' or 'search')
            config: Phase configuration to pass as JSON
            timeout_seconds: Maximum execution time
            run_id: Optional run identifier for log correlation

        Returns:
            Tuple of (ContainerResult, ResourceSummary)
        """
        import time
        from uuid import uuid4

        # Generate unique ID for log files, incorporating run_id if provided
        phase_id = str(uuid4())[:8]
        log_prefix = f"[{run_id}]" if run_id else ""

        # Determine timeout
        if timeout_seconds is None:
            if mode == "build":
                timeout_seconds = algorithm.build.timeout_seconds
            else:
                timeout_seconds = algorithm.search.timeout_seconds

        # Prepare command arguments (ENTRYPOINT already runs 'python -m algorithm.runner')
        config_json = json.dumps(config)
        command = ["--mode", mode, "--config", config_json]

        # Prepare volume mounts
        # CRITICAL: Mount host directories so disk I/O is real (not overlay FS)
        volumes = self._prepare_volumes(algorithm)

        # Prepare resource limits
        resource_limits = self._prepare_resource_limits(algorithm)

        container: Container | None = None

        start_time = time.monotonic()

        try:
            # Create and start container
            logger.info(f"{log_prefix}Starting container for {algorithm.name} ({mode} phase)")
            container = self._client.containers.run(
                algorithm.docker_image,
                command=command,
                volumes=volumes,
                environment=algorithm.env_vars,
                detach=True,
                **resource_limits,
            )

            # Start cgroups collector for metrics
            self._cgroups_collector.start(container.id)

            # Wait for container to complete
            try:
                result = container.wait(timeout=timeout_seconds)
                exit_code = result.get("StatusCode", -1) if isinstance(result, dict) else result
            except Exception as e:
                logger.warning(f"Container wait failed: {e}")
                exit_code = -1

            # Stop monitoring and collect metrics
            cgroups_result = self._cgroups_collector.stop()

            # Build ResourceSummary from cgroups metrics
            resources = ResourceSummary(
                peak_memory_mb=cgroups_result.peak_memory_mb,
                avg_memory_mb=cgroups_result.avg_memory_mb,
                cpu_time_total_seconds=cgroups_result.cpu_time_total_seconds,
                avg_cpu_percent=cgroups_result.avg_cpu_percent,
                peak_cpu_percent=cgroups_result.peak_cpu_percent,
                total_blkio_read_mb=cgroups_result.total_read_bytes / (1024 * 1024),
                total_blkio_write_mb=cgroups_result.total_write_bytes / (1024 * 1024),
                total_read_ops=cgroups_result.total_read_ops,
                total_write_ops=cgroups_result.total_write_ops,
                avg_read_iops=cgroups_result.avg_read_iops,
                avg_write_iops=cgroups_result.avg_write_iops,
                total_read_usec=cgroups_result.total_read_usec,
                total_write_usec=cgroups_result.total_write_usec,
                io_pressure_some_total_usec=cgroups_result.io_pressure_some_total_usec,
                io_pressure_full_total_usec=cgroups_result.io_pressure_full_total_usec,
                pgmajfault_delta=cgroups_result.pgmajfault_delta,
                pgfault_delta=cgroups_result.pgfault_delta,
                avg_file_bytes=cgroups_result.avg_file_bytes,
                peak_file_bytes=cgroups_result.peak_file_bytes,
                avg_file_mapped_bytes=cgroups_result.avg_file_mapped_bytes,
                peak_file_mapped_bytes=cgroups_result.peak_file_mapped_bytes,
                avg_active_file_bytes=cgroups_result.avg_active_file_bytes,
                peak_active_file_bytes=cgroups_result.peak_active_file_bytes,
                avg_inactive_file_bytes=cgroups_result.avg_inactive_file_bytes,
                peak_inactive_file_bytes=cgroups_result.peak_inactive_file_bytes,
                nr_throttled_delta=cgroups_result.nr_throttled_delta,
                throttled_usec_delta=cgroups_result.throttled_usec_delta,
                top_read_device={
                    "device": cgroups_result.top_read_device.device,
                    "total_read_bytes": cgroups_result.top_read_device.total_read_bytes,
                    "total_write_bytes": cgroups_result.top_read_device.total_write_bytes,
                    "total_read_ops": cgroups_result.top_read_device.total_read_ops,
                    "total_write_ops": cgroups_result.top_read_device.total_write_ops,
                }
                if cgroups_result.top_read_device
                else None,
                p95_read_iops=cgroups_result.p95_read_iops,
                max_read_iops=cgroups_result.max_read_iops,
                p95_read_mbps=cgroups_result.p95_read_mbps,
                max_read_mbps=cgroups_result.max_read_mbps,
                p95_read_service_time_ms=cgroups_result.p95_read_service_time_ms,
                max_read_service_time_ms=cgroups_result.max_read_service_time_ms,
                sample_count=cgroups_result.sample_count,
                duration_seconds=cgroups_result.duration_seconds,
                block_size=self._block_size,
            )

            logger.debug(
                f"Collected metrics from cgroups: "
                f"cpu_time={cgroups_result.cpu_time_total_seconds:.2f}s, "
                f"avg_cpu={cgroups_result.avg_cpu_percent:.1f}%, "
                f"peak_cpu={cgroups_result.peak_cpu_percent:.1f}%, "
                f"read_iops={cgroups_result.avg_read_iops:.1f}"
            )

            # Stream logs directly to files to reduce memory usage, keeping only a tail
            # for parsing (JSON output is expected at the end of stdout)
            logs_dir = self.results_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            stdout_path = logs_dir / f"{algorithm.name}_{mode}_{phase_id}.stdout.log"
            stderr_path = logs_dir / f"{algorithm.name}_{mode}_{phase_id}.stderr.log"

            # Number of bytes to keep in memory for parsing (enough for JSON output)
            tail_size = 64 * 1024  # 64KB tail

            stdout_tail, stderr_tail = self._stream_logs_to_files(
                container, stdout_path, stderr_path, tail_size=tail_size
            )
            stdout = stdout_tail
            stderr = stderr_tail

            duration = time.monotonic() - start_time

            # Parse output JSON
            # Metrics file is expected at /results/metrics.json in the container
            metrics_file = self.results_dir / "metrics.json"

            output = self._parse_output(stdout, stderr, mode=mode, metrics_file=metrics_file)

            # Refine metrics using query window if available (Issue #1)
            query_start = output.get("query_start_timestamp")
            query_end = output.get("query_end_timestamp")

            if query_start and query_end and mode == "search":
                try:
                    from datetime import datetime

                    start_dt = datetime.fromisoformat(query_start)
                    end_dt = datetime.fromisoformat(query_end)

                    logger.debug(
                        f"{log_prefix}Refining metrics to query window: {query_start} -> {query_end}"
                    )

                    # Re-aggregate metrics filtering for the specific query window
                    cgroups_result = self._cgroups_collector.get_summary(start_dt, end_dt)

                    # Update the resources object with the refined metrics
                    resources = ResourceSummary(
                        peak_memory_mb=cgroups_result.peak_memory_mb,
                        avg_memory_mb=cgroups_result.avg_memory_mb,
                        cpu_time_total_seconds=cgroups_result.cpu_time_total_seconds,
                        avg_cpu_percent=cgroups_result.avg_cpu_percent,
                        peak_cpu_percent=cgroups_result.peak_cpu_percent,
                        total_blkio_read_mb=cgroups_result.total_read_bytes / (1024 * 1024),
                        total_blkio_write_mb=cgroups_result.total_write_bytes / (1024 * 1024),
                        total_read_ops=cgroups_result.total_read_ops,
                        total_write_ops=cgroups_result.total_write_ops,
                        avg_read_iops=cgroups_result.avg_read_iops,
                        avg_write_iops=cgroups_result.avg_write_iops,
                        total_read_usec=cgroups_result.total_read_usec,
                        total_write_usec=cgroups_result.total_write_usec,
                        io_pressure_some_total_usec=cgroups_result.io_pressure_some_total_usec,
                        io_pressure_full_total_usec=cgroups_result.io_pressure_full_total_usec,
                        pgmajfault_delta=cgroups_result.pgmajfault_delta,
                        pgfault_delta=cgroups_result.pgfault_delta,
                        avg_file_bytes=cgroups_result.avg_file_bytes,
                        peak_file_bytes=cgroups_result.peak_file_bytes,
                        avg_file_mapped_bytes=cgroups_result.avg_file_mapped_bytes,
                        peak_file_mapped_bytes=cgroups_result.peak_file_mapped_bytes,
                        avg_active_file_bytes=cgroups_result.avg_active_file_bytes,
                        peak_active_file_bytes=cgroups_result.peak_active_file_bytes,
                        avg_inactive_file_bytes=cgroups_result.avg_inactive_file_bytes,
                        peak_inactive_file_bytes=cgroups_result.peak_inactive_file_bytes,
                        nr_throttled_delta=cgroups_result.nr_throttled_delta,
                        throttled_usec_delta=cgroups_result.throttled_usec_delta,
                        top_read_device={
                            "device": cgroups_result.top_read_device.device,
                            "total_read_bytes": cgroups_result.top_read_device.total_read_bytes,
                            "total_write_bytes": cgroups_result.top_read_device.total_write_bytes,
                            "total_read_ops": cgroups_result.top_read_device.total_read_ops,
                            "total_write_ops": cgroups_result.top_read_device.total_write_ops,
                        }
                        if cgroups_result.top_read_device
                        else None,
                        p95_read_iops=cgroups_result.p95_read_iops,
                        max_read_iops=cgroups_result.max_read_iops,
                        p95_read_mbps=cgroups_result.p95_read_mbps,
                        max_read_mbps=cgroups_result.max_read_mbps,
                        p95_read_service_time_ms=cgroups_result.p95_read_service_time_ms,
                        max_read_service_time_ms=cgroups_result.max_read_service_time_ms,
                        sample_count=cgroups_result.sample_count,
                        duration_seconds=cgroups_result.duration_seconds,
                        samples=[],  # Don't duplicate raw samples to save space
                        block_size=self._block_size,
                    )
                    logger.info(
                        f"{log_prefix}Refined metrics: cpu={resources.avg_cpu_percent:.1f}%, "
                        f"read_iops={resources.avg_read_iops:.1f}, "
                        f"duration={resources.duration_seconds:.2f}s"
                    )

                except Exception as e:
                    logger.warning(f"Failed to refine metrics to query window: {e}")

            # Clean up metrics file for next run
            if metrics_file.exists():
                try:
                    metrics_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup metrics file: {e}")
            error_msg = output.get("error_message") if output.get("status") == "error" else None

            # Calculate warmup phase metrics if timestamps available
            # Support both old "load_" and new "warmup_" field names for backward compatibility
            warmup_start = output.get("warmup_start_timestamp", output.get("load_start_timestamp"))
            warmup_end = output.get("warmup_end_timestamp", output.get("load_end_timestamp"))
            warmup_resources_obj = None

            if warmup_start and warmup_end:
                try:
                    from datetime import datetime

                    ws_dt = datetime.fromisoformat(warmup_start)
                    we_dt = datetime.fromisoformat(warmup_end)
                    warmup_res = self._cgroups_collector.get_summary(ws_dt, we_dt)

                    warmup_resources_obj = ResourceSummary(
                        peak_memory_mb=warmup_res.peak_memory_mb,
                        avg_memory_mb=warmup_res.avg_memory_mb,
                        cpu_time_total_seconds=warmup_res.cpu_time_total_seconds,
                        avg_cpu_percent=warmup_res.avg_cpu_percent,
                        peak_cpu_percent=warmup_res.peak_cpu_percent,
                        total_blkio_read_mb=warmup_res.total_read_bytes / (1024 * 1024),
                        total_blkio_write_mb=warmup_res.total_write_bytes / (1024 * 1024),
                        total_read_ops=warmup_res.total_read_ops,
                        total_write_ops=warmup_res.total_write_ops,
                        avg_read_iops=warmup_res.avg_read_iops,
                        avg_write_iops=warmup_res.avg_write_iops,
                        total_read_usec=warmup_res.total_read_usec,
                        total_write_usec=warmup_res.total_write_usec,
                        io_pressure_some_total_usec=warmup_res.io_pressure_some_total_usec,
                        io_pressure_full_total_usec=warmup_res.io_pressure_full_total_usec,
                        pgmajfault_delta=warmup_res.pgmajfault_delta,
                        pgfault_delta=warmup_res.pgfault_delta,
                        avg_file_bytes=warmup_res.avg_file_bytes,
                        peak_file_bytes=warmup_res.peak_file_bytes,
                        avg_file_mapped_bytes=warmup_res.avg_file_mapped_bytes,
                        peak_file_mapped_bytes=warmup_res.peak_file_mapped_bytes,
                        avg_active_file_bytes=warmup_res.avg_active_file_bytes,
                        peak_active_file_bytes=warmup_res.peak_active_file_bytes,
                        avg_inactive_file_bytes=warmup_res.avg_inactive_file_bytes,
                        peak_inactive_file_bytes=warmup_res.peak_inactive_file_bytes,
                        nr_throttled_delta=warmup_res.nr_throttled_delta,
                        throttled_usec_delta=warmup_res.throttled_usec_delta,
                        top_read_device={
                            "device": warmup_res.top_read_device.device,
                            "total_read_bytes": warmup_res.top_read_device.total_read_bytes,
                            "total_write_bytes": warmup_res.top_read_device.total_write_bytes,
                            "total_read_ops": warmup_res.top_read_device.total_read_ops,
                            "total_write_ops": warmup_res.top_read_device.total_write_ops,
                        }
                        if warmup_res.top_read_device
                        else None,
                        p95_read_iops=warmup_res.p95_read_iops,
                        max_read_iops=warmup_res.max_read_iops,
                        p95_read_mbps=warmup_res.p95_read_mbps,
                        max_read_mbps=warmup_res.max_read_mbps,
                        p95_read_service_time_ms=warmup_res.p95_read_service_time_ms,
                        max_read_service_time_ms=warmup_res.max_read_service_time_ms,
                        sample_count=warmup_res.sample_count,
                        duration_seconds=warmup_res.duration_seconds,
                        block_size=self._block_size,
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate warmup metrics: {e}")

            return (
                ContainerResult(
                    success=exit_code == 0,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    output=output,
                    error_message=error_msg,
                    warmup_resources=warmup_resources_obj,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                ),
                resources,
            )

        except ContainerError as e:
            duration = time.monotonic() - start_time
            logger.error(f"Container error: {e}")
            return (
                ContainerResult(
                    success=False,
                    exit_code=e.exit_status or -1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=duration,
                    error_message=str(e),
                ),
                ResourceSummary(
                    peak_memory_mb=0,
                    avg_memory_mb=0,
                    avg_cpu_percent=0,
                    peak_cpu_percent=0,
                    total_blkio_read_mb=0,
                    total_blkio_write_mb=0,
                    avg_read_iops=0,
                    avg_write_iops=0,
                    sample_count=0,
                    duration_seconds=0,
                ),
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error(f"Unexpected error: {e}")
            return (
                ContainerResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=duration,
                    error_message=str(e),
                ),
                ResourceSummary(
                    peak_memory_mb=0,
                    avg_memory_mb=0,
                    avg_cpu_percent=0,
                    peak_cpu_percent=0,
                    total_blkio_read_mb=0,
                    total_blkio_write_mb=0,
                    avg_read_iops=0,
                    avg_write_iops=0,
                    sample_count=0,
                    duration_seconds=0,
                ),
            )

        finally:
            # Cleanup
            if container is not None:
                try:
                    container.remove(force=True)
                    logger.debug(f"Removed container {container.short_id}")
                except NotFound:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

    def _prepare_volumes(self, algorithm: AlgorithmConfig) -> dict[str, dict[str, str]]:
        """Prepare volume mounts for the container.

        For disk-based algorithms, mounting host directories is CRITICAL
        to ensure accurate I/O metrics and persistent indices.
        """
        volumes = {
            str(self.data_dir): {"bind": "/data", "mode": "rw"},
            str(self.index_dir): {"bind": "/data/index", "mode": "rw"},
            str(self.results_dir): {"bind": "/results", "mode": "rw"},
            # Mount logs dir as well if needed, but we write from host side
        }
        return volumes

    def _prepare_resource_limits(self, algorithm: AlgorithmConfig) -> dict[str, Any]:
        """Prepare resource limits for the container."""
        limits: dict[str, Any] = {
            "network_mode": "host",  # Eliminate NAT overhead for accurate latency
            "shm_size": "2g",  # Large workloads (FAISS, etc.) need more than 64MB default
            "security_opt": ["seccomp=unconfined"],  # Allow advanced syscalls (io_uring, mmap)
        }

        if algorithm.cpu_limit:
            limits["cpuset_cpus"] = algorithm.cpu_limit

        if algorithm.memory_limit:
            limits["mem_limit"] = algorithm.memory_limit

        return limits

    def _stream_logs_to_files(
        self,
        container: Container,
        stdout_path: Path,
        stderr_path: Path,
        tail_size: int = 64 * 1024,
    ) -> tuple[str, str]:
        """Stream container logs to files, returning only a tail for parsing.

        This avoids holding the entire log in memory for long-running containers
        while still providing a tail for JSON output parsing.

        Args:
            container: Docker container to read logs from
            stdout_path: Path to write stdout log file
            stderr_path: Path to write stderr log file
            tail_size: Number of bytes to keep in memory for parsing

        Returns:
            Tuple of (stdout_tail, stderr_tail) as strings
        """
        from collections import deque

        # Use streaming API to avoid loading entire log into memory
        # Docker SDK logs() with stream=True yields chunks
        stdout_chunks: deque[bytes] = deque()
        stderr_chunks: deque[bytes] = deque()
        stdout_bytes = 0
        stderr_bytes = 0

        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            # Stream stdout
            for chunk in container.logs(stdout=True, stderr=False, stream=True):
                stdout_file.write(chunk)
                stdout_chunks.append(chunk)
                stdout_bytes += len(chunk)
                # Trim tail buffer if it exceeds limit
                while stdout_bytes > tail_size and len(stdout_chunks) > 1:
                    removed = stdout_chunks.popleft()
                    stdout_bytes -= len(removed)

            # Stream stderr
            for chunk in container.logs(stdout=False, stderr=True, stream=True):
                stderr_file.write(chunk)
                stderr_chunks.append(chunk)
                stderr_bytes += len(chunk)
                while stderr_bytes > tail_size and len(stderr_chunks) > 1:
                    removed = stderr_chunks.popleft()
                    stderr_bytes -= len(removed)

        # Decode tails for parsing
        stdout_tail = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_tail = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        return stdout_tail, stderr_tail

    def _parse_output(
        self, stdout: str, stderr: str = "", mode: str = "search", metrics_file: Path | None = None
    ) -> dict[str, Any]:
        """Parse container output into structured data.

        Prioritizes reading from metrics.json if available, otherwise falls back to stdout parsing.
        """
        from ann_suite.core.schemas import ContainerProtocol

        output_data = {}

        # 1. Try reading from metrics file (Preferred)
        if metrics_file and metrics_file.exists():
            try:
                content = metrics_file.read_text()
                if content.strip():
                    output_data = json.loads(content)
                    logger.debug(f"Loaded output from {metrics_file}")
            except Exception as e:
                logger.warning(f"Failed to read metrics file {metrics_file}: {e}")

        # 2. Fall back to stdout parsing if file failed or not present
        if not output_data:
            try:
                # Look for JSON in stdout (last non-empty line)
                lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
                if lines:
                    for line in reversed(lines):
                        if line.startswith("{"):
                            output_data = json.loads(line)
                            break
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse output JSON from stdout: {e}")

        # If we still have no output data, it's a failure
        if not output_data:
            logger.error("No metrics found in metrics.json or stdout")

        # If we found valid JSON, validate against protocol schema
        if output_data:
            try:
                if mode == "build":
                    ContainerProtocol.BuildOutput.model_validate(output_data)
                else:
                    ContainerProtocol.SearchOutput.model_validate(output_data)
            except Exception as e:
                # Log actionable validation warning but continue with raw data
                expected_fields = (
                    "status, build_time_seconds, index_size_bytes"
                    if mode == "build"
                    else "status, total_queries, total_time_seconds, qps, recall, "
                    "mean_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, "
                    "query_start_timestamp, query_end_timestamp"
                )
                logger.warning(
                    f"Container output doesn't fully match {mode} protocol schema: {e}. "
                    f"Expected fields: {expected_fields}. "
                    f"Received keys: {list(output_data.keys())}. "
                    "Some metrics may be missing or incorrectly typed."
                )
            return output_data

        # If no JSON found, implies error (or empty output)
        return {
            "status": "error",
            "error_message": stderr or "No JSON output found in stdout or metrics file",
        }

    def build_image(
        self,
        dockerfile_path: Path,
        image_tag: str,
        build_args: dict[str, str] | None = None,
        context_path: Path | None = None,
    ) -> bool:
        """Build a Docker image from a Dockerfile.

        Args:
            dockerfile_path: Path to Dockerfile
            image_tag: Tag for the built image
            build_args: Build arguments to pass
            context_path: Optional build context path (defaults to Dockerfile parent)

        Returns:
            True if build succeeded
        """
        try:
            logger.info(f"Building image {image_tag} from {dockerfile_path}")

            # Use provided context path or default to Dockerfile directory
            build_context = context_path if context_path else dockerfile_path.parent

            # Dockerfile path relative to context
            # If context is parent-of-parent, we need relative path
            # But docker-py takes 'dockerfile' as path within context
            if context_path:
                rel_dockerfile = dockerfile_path.relative_to(context_path)
                dockerfile_arg = str(rel_dockerfile)
            else:
                dockerfile_arg = dockerfile_path.name

            self._client.images.build(
                path=str(build_context),
                dockerfile=dockerfile_arg,
                tag=image_tag,
                buildargs=build_args or {},
                rm=True,
            )
            logger.info(f"Successfully built {image_tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up any resources."""
        with contextlib.suppress(Exception):
            self._client.close()
