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
from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector
from ann_suite.monitoring.resource_monitor import ResourceMonitor

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

        # Check if cgroups v2 collector is available for enhanced I/O metrics
        self._cgroups_collector = CgroupsV2Collector(interval_ms=monitor_interval_ms)
        self._use_cgroups = self._cgroups_collector.is_available()
        if self._use_cgroups:
            logger.info("CgroupsV2Collector available - will collect enhanced I/O metrics")
        else:
            logger.info("CgroupsV2Collector not available - using Docker stats only")

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
    ) -> tuple[ContainerResult, ResourceSummary]:
        """Run a benchmark phase (build or search) in a container.

        Args:
            algorithm: Algorithm configuration
            mode: Phase mode ('build' or 'search')
            config: Phase configuration to pass as JSON
            timeout_seconds: Maximum execution time

        Returns:
            Tuple of (ContainerResult, ResourceSummary)
        """
        import time

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
        monitor: ResourceMonitor | None = None

        start_time = time.monotonic()

        try:
            # Create and start container
            logger.info(f"Starting container for {algorithm.name} ({mode} phase)")
            container = self._client.containers.run(
                algorithm.docker_image,
                command=command,
                volumes=volumes,
                environment=algorithm.env_vars,
                detach=True,
                **resource_limits,
            )

            # Start resource monitoring immediately - the monitor's start() method
            # collects an initial sample synchronously to capture fast containers
            monitor = ResourceMonitor(container, interval_ms=self.monitor_interval_ms)
            monitor.start()

            # Also start cgroups collector for enhanced I/O metrics if available
            if self._use_cgroups:
                self._cgroups_collector.start(container.id)

            # Wait for container to complete
            try:
                result = container.wait(timeout=timeout_seconds)
                exit_code = result.get("StatusCode", -1) if isinstance(result, dict) else result
            except Exception as e:
                logger.warning(f"Container wait failed: {e}")
                exit_code = -1

            # Stop monitoring
            resources = monitor.stop()

            # If cgroups collector was used, enhance I/O metrics
            if self._use_cgroups:
                cgroups_result = self._cgroups_collector.stop()
                # Merge more accurate I/O metrics from cgroups
                if cgroups_result.avg_read_iops > 0 or cgroups_result.avg_write_iops > 0:
                    resources = ResourceSummary(
                        peak_memory_mb=resources.peak_memory_mb,
                        avg_memory_mb=resources.avg_memory_mb,
                        cpu_time_total_seconds=cgroups_result.cpu_time_total_seconds,
                        avg_cpu_percent=resources.avg_cpu_percent,
                        peak_cpu_percent=resources.peak_cpu_percent,
                        # Use cgroups I/O metrics (more accurate)
                        total_blkio_read_mb=cgroups_result.total_read_bytes / (1024 * 1024),
                        total_blkio_write_mb=cgroups_result.total_write_bytes / (1024 * 1024),
                        avg_read_iops=cgroups_result.avg_read_iops,
                        avg_write_iops=cgroups_result.avg_write_iops,
                        sample_count=resources.sample_count,
                        duration_seconds=resources.duration_seconds,
                        samples=resources.samples,
                    )
                    iops = cgroups_result.avg_read_iops
                    logger.debug(f"Enhanced I/O: read={iops:.1f} IOPS")

            # Collect output
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            duration = time.monotonic() - start_time

            # Parse output JSON
            # Metrics file is expected at /results/metrics.json in the container
            metrics_file = self.results_dir / "metrics.json"

            output = self._parse_output(stdout, stderr, mode=mode, metrics_file=metrics_file)

            # Clean up metrics file for next run
            if metrics_file.exists():
                try:
                    metrics_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup metrics file: {e}")
            error_msg = output.get("error_message") if output.get("status") == "error" else None

            return (
                ContainerResult(
                    success=exit_code == 0,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    output=output,
                    error_message=error_msg,
                ),
                resources,
            )

        except ContainerError as e:
            duration = time.monotonic() - start_time
            logger.error(f"Container error: {e}")
            resources = monitor.stop() if monitor else ResourceSummary(
                peak_memory_mb=0, avg_memory_mb=0, avg_cpu_percent=0, peak_cpu_percent=0,
                total_blkio_read_mb=0, total_blkio_write_mb=0, avg_read_iops=0, avg_write_iops=0,
                sample_count=0, duration_seconds=0,
            )
            return (
                ContainerResult(
                    success=False,
                    exit_code=e.exit_status or -1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=duration,
                    error_message=str(e),
                ),
                resources,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error(f"Unexpected error: {e}")
            resources = monitor.stop() if monitor else ResourceSummary(
                peak_memory_mb=0, avg_memory_mb=0, avg_cpu_percent=0, peak_cpu_percent=0,
                total_blkio_read_mb=0, total_blkio_write_mb=0, avg_read_iops=0, avg_write_iops=0,
                sample_count=0, duration_seconds=0,
            )
            return (
                ContainerResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration_seconds=duration,
                    error_message=str(e),
                ),
                resources,
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
        }
        return volumes

    def _prepare_resource_limits(self, algorithm: AlgorithmConfig) -> dict[str, Any]:
        """Prepare resource limits for the container."""
        limits: dict[str, Any] = {}

        if algorithm.cpu_limit:
            limits["cpuset_cpus"] = algorithm.cpu_limit

        if algorithm.memory_limit:
            limits["mem_limit"] = algorithm.memory_limit

        return limits

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

        # If we found valid JSON, validate against protocol schema
        if output_data:
            try:
                if mode == "build":
                    ContainerProtocol.BuildOutput.model_validate(output_data)
                else:
                    ContainerProtocol.SearchOutput.model_validate(output_data)
            except Exception as e:
                # Log validation warning but continue with raw data
                logger.warning(f"Container output doesn't match protocol schema: {e}")
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
    ) -> bool:
        """Build a Docker image from a Dockerfile.

        Args:
            dockerfile_path: Path to Dockerfile
            image_tag: Tag for the built image
            build_args: Build arguments to pass

        Returns:
            True if build succeeded
        """
        try:
            logger.info(f"Building image {image_tag} from {dockerfile_path}")
            self._client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_path.name,
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
