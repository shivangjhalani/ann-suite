"""Tests for OS page cache dropping (cold-start benchmarking support).

Covers:
- ContainerRunner.drop_caches fallback chain: direct write -> sudo -n -> sudo -S
- ANN_SUITE_SUDO_PASSWORD environment variable handling
- Failure path returns False and logs a warning
- Evaluator hook: drop_caches_before triggers the runner call before search
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ann_suite.core.schemas import (
    AlgorithmConfig,
    BenchmarkConfig,
    DatasetConfig,
    ResourceSummary,
)
from ann_suite.evaluator import BenchmarkEvaluator
from ann_suite.runners.container_runner import (
    SUDO_PASSWORD_ENV,
    ContainerRunner,
    cpuset_to_numa_nodes,
    expand_cpuset,
)


@pytest.fixture
def runner(tmp_path: Path) -> ContainerRunner:
    return ContainerRunner(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "indices",
        results_dir=tmp_path / "results",
    )


class TestDropCaches:
    def test_direct_write_success(self, runner: ContainerRunner) -> None:
        with (
            patch.object(runner, "_write_drop_caches_direct", return_value=True) as direct,
            patch.object(runner, "_run_sudo_drop_caches") as sudo,
        ):
            assert runner.drop_caches() is True
        direct.assert_called_once()
        sudo.assert_not_called()

    def test_passwordless_sudo_fallback(self, runner: ContainerRunner) -> None:
        with (
            patch.object(runner, "_write_drop_caches_direct", return_value=False),
            patch.object(runner, "_run_sudo_drop_caches", return_value=True) as sudo,
        ):
            assert runner.drop_caches() is True
        sudo.assert_called_once_with(None)

    def test_sudo_with_env_password(self, runner: ContainerRunner) -> None:
        with (
            patch.object(runner, "_write_drop_caches_direct", return_value=False),
            patch.object(runner, "_run_sudo_drop_caches", side_effect=[False, True]) as sudo,
            patch.dict(os.environ, {SUDO_PASSWORD_ENV: "secret"}),
        ):
            assert runner.drop_caches() is True
        assert sudo.call_args_list[0] == ((None,),)
        assert sudo.call_args_list[1] == (("secret",),)

    def test_total_failure_returns_false_and_warns(
        self, runner: ContainerRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(runner, "_write_drop_caches_direct", return_value=False),
            patch.object(runner, "_run_sudo_drop_caches", return_value=False),
            patch.dict(os.environ, clear=False),
        ):
            os.environ.pop(SUDO_PASSWORD_ENV, None)
            assert runner.drop_caches() is False
        assert any("WARM page cache" in r.message for r in caplog.records)

    def test_sudo_invocation_commands(self, runner: ContainerRunner) -> None:
        """sudo -n runs without stdin; sudo -S feeds the password via stdin."""
        with patch("ann_suite.runners.container_runner.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0)
            assert runner._run_sudo_drop_caches(None) is True
            command = run.call_args.args[0]
            assert "sudo" in command and "-n" in command
            assert run.call_args.kwargs["input"] is None

            assert runner._run_sudo_drop_caches("pw") is True
            command = run.call_args.args[0]
            assert "-S" in command
            assert run.call_args.kwargs["input"] == "pw\n"


class TestResourceLimits:
    """Tests for ContainerRunner._prepare_resource_limits (CPU/memory caps)."""

    def test_cpu_limit_maps_to_nano_cpus(self, runner: ContainerRunner) -> None:
        algo = AlgorithmConfig(name="A", docker_image="a:latest", cpu_limit=8.0)
        limits = runner._prepare_resource_limits(algo)
        assert limits["nano_cpus"] == 8_000_000_000

    def test_no_nano_cpus_when_cpu_limit_unset(self, runner: ContainerRunner) -> None:
        algo = AlgorithmConfig(name="A", docker_image="a:latest")
        limits = runner._prepare_resource_limits(algo)
        assert "nano_cpus" not in limits

    def test_cpu_affinity_sets_cpuset(self, runner: ContainerRunner) -> None:
        algo = AlgorithmConfig(name="A", docker_image="a:latest", cpu_affinity="0-3")
        limits = runner._prepare_resource_limits(algo)
        assert limits["cpuset_cpus"] == "0-3"

    def test_memory_limit_sets_mem_and_swap(self, runner: ContainerRunner) -> None:
        algo = AlgorithmConfig(name="A", docker_image="a:latest", memory_limit="8g")
        limits = runner._prepare_resource_limits(algo)
        assert limits["mem_limit"] == "8g"
        assert limits["memswap_limit"] == "8g"

    def test_cpu_affinity_sets_cpuset_mems_when_numa_available(
        self, tmp_path: Path
    ) -> None:
        """cpu_affinity should also pin memory to the matching NUMA node."""
        with (
            patch("ann_suite.runners.container_runner.docker.from_env") as from_env,
            patch(
                "ann_suite.runners.container_runner.CgroupsV2Collector.check_available",
                return_value=True,
            ),
        ):
            from_env.return_value = MagicMock()
            runner = ContainerRunner(
                data_dir=tmp_path / "data",
                index_dir=tmp_path / "indices",
                results_dir=tmp_path / "results",
            )
            algo = AlgorithmConfig(name="A", docker_image="a:latest", cpu_affinity="0-3")
            with patch(
                "ann_suite.runners.container_runner.cpuset_to_numa_nodes", return_value="0"
            ):
                limits = runner._prepare_resource_limits(algo)
        assert limits["cpuset_cpus"] == "0-3"
        assert limits["cpuset_mems"] == "0"

    def test_cpu_affinity_omits_cpuset_mems_when_no_numa(
        self, tmp_path: Path
    ) -> None:
        """On single-node/non-NUMA hosts, memory placement is left at the default."""
        with (
            patch("ann_suite.runners.container_runner.docker.from_env") as from_env,
            patch(
                "ann_suite.runners.container_runner.CgroupsV2Collector.check_available",
                return_value=True,
            ),
        ):
            from_env.return_value = MagicMock()
            runner = ContainerRunner(
                data_dir=tmp_path / "data",
                index_dir=tmp_path / "indices",
                results_dir=tmp_path / "results",
            )
            algo = AlgorithmConfig(name="A", docker_image="a:latest", cpu_affinity="0-3")
            with patch(
                "ann_suite.runners.container_runner.cpuset_to_numa_nodes", return_value=None
            ):
                limits = runner._prepare_resource_limits(algo)
        assert limits["cpuset_cpus"] == "0-3"
        assert "cpuset_mems" not in limits

    def test_no_cpuset_when_no_affinity(self, tmp_path: Path) -> None:
        with (
            patch("ann_suite.runners.container_runner.docker.from_env") as from_env,
            patch(
                "ann_suite.runners.container_runner.CgroupsV2Collector.check_available",
                return_value=True,
            ),
        ):
            from_env.return_value = MagicMock()
            runner = ContainerRunner(
                data_dir=tmp_path / "data",
                index_dir=tmp_path / "indices",
                results_dir=tmp_path / "results",
            )
            algo = AlgorithmConfig(name="A", docker_image="a:latest")
            limits = runner._prepare_resource_limits(algo)
        assert "cpuset_cpus" not in limits
        assert "cpuset_mems" not in limits


class TestNumaHelpers:
    """Tests for cpuset parsing and NUMA-node mapping helpers."""

    def test_expand_single_range(self) -> None:
        assert expand_cpuset("0-3") == {0, 1, 2, 3}

    def test_expand_discrete_cores(self) -> None:
        assert expand_cpuset("0,2,4,6") == {0, 2, 4, 6}

    def test_expand_single_core(self) -> None:
        assert expand_cpuset("4") == {4}

    def test_expand_empty(self) -> None:
        assert expand_cpuset("") == set()

    def test_expand_ignores_bad_tokens(self) -> None:
        assert expand_cpuset("0-x,2") == {2}
        assert expand_cpuset("3-1") == set()

    def test_numa_mapping_single_node(self, tmp_path: Path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        (node0 / "cpulist").write_text("0-7\n")
        fake = [str(node0 / "cpulist")]
        with patch("ann_suite.runners.container_runner.glob.glob", return_value=fake):
            assert cpuset_to_numa_nodes("0-3") == "0"

    def test_numa_mapping_spans_nodes(self, tmp_path: Path) -> None:
        node0 = tmp_path / "node0"
        node1 = tmp_path / "node1"
        node0.mkdir()
        node1.mkdir()
        (node0 / "cpulist").write_text("0-3\n")
        (node1 / "cpulist").write_text("4-7\n")
        fake = [str(node0 / "cpulist"), str(node1 / "cpulist")]
        with patch("ann_suite.runners.container_runner.glob.glob", return_value=fake):
            assert cpuset_to_numa_nodes("2-5") == "0,1"

    def test_numa_mapping_unavailable(self) -> None:
        with patch("ann_suite.runners.container_runner.glob.glob", return_value=[]):
            assert cpuset_to_numa_nodes("0-3") is None


class TestEvaluatorDropCachesHook:
    def test_hook_runs_before_search_phase(self, tmp_path: Path) -> None:
        evaluator = BenchmarkEvaluator(
            BenchmarkConfig(
                data_dir=tmp_path / "data",
                results_dir=tmp_path / "results",
                index_dir=tmp_path / "indices",
            )
        )
        evaluator.container_runner = MagicMock()
        evaluator.container_runner.pull_image.return_value = True
        evaluator.container_runner.drop_caches.return_value = True
        evaluator.container_runner.run_phase.return_value = (
            SimpleNamespace(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=1.0,
                output={"status": "success", "qps": 1.0, "recall": 1.0},
                error_message=None,
                warmup_resources=None,
                stdout_path=None,
                stderr_path=None,
            ),
            ResourceSummary(
                peak_memory_mb=0.0,
                avg_memory_mb=0.0,
                avg_cpu_percent=0.0,
                peak_cpu_percent=0.0,
                total_blkio_read_mb=0.0,
                total_blkio_write_mb=0.0,
                avg_read_iops=0.0,
                avg_write_iops=0.0,
                sample_count=0,
                duration_seconds=0.0,
            ),
        )
        evaluator._prepare_dataset = MagicMock(  # type: ignore[method-assign]
            return_value=(None, None, None)
        )
        evaluator._prepare_dataset_files = MagicMock(  # type: ignore[method-assign]
            return_value=(tmp_path / "b.npy", tmp_path / "q.npy", None)
        )

        algo = AlgorithmConfig(
            name="A",
            docker_image="a:latest",
            search={"k": 10, "warmup": {"drop_caches_before": True}},
        )
        dataset = DatasetConfig(name="ds", base_path=Path("base.npy"), dimension=8)
        evaluator.config = BenchmarkConfig(
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            index_dir=tmp_path / "indices",
            algorithms=[algo],
            datasets=[dataset],
        )
        results: list[Any] = evaluator.run()

        assert len(results) == 1
        evaluator.container_runner.drop_caches.assert_called_once()
        # Order: pull_image -> build run_phase -> drop_caches -> search run_phase
        calls = [c[0] for c in evaluator.container_runner.method_calls]
        assert calls == ["pull_image", "run_phase", "drop_caches", "run_phase"]

    def test_no_hook_when_disabled(self, tmp_path: Path) -> None:
        evaluator = BenchmarkEvaluator(
            BenchmarkConfig(
                data_dir=tmp_path / "data",
                results_dir=tmp_path / "results",
                index_dir=tmp_path / "indices",
            )
        )
        evaluator.container_runner = MagicMock()
        evaluator.container_runner.pull_image.return_value = True
        evaluator._prepare_dataset = MagicMock(  # type: ignore[method-assign]
            return_value=(None, None, None)
        )
        evaluator._prepare_dataset_files = MagicMock(  # type: ignore[method-assign]
            return_value=(tmp_path / "b.npy", tmp_path / "q.npy", None)
        )
        algo = AlgorithmConfig(name="A", docker_image="a:latest")
        dataset = DatasetConfig(name="ds", base_path=Path("base.npy"), dimension=8)
        evaluator.config = BenchmarkConfig(
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            index_dir=tmp_path / "indices",
            algorithms=[algo],
            datasets=[dataset],
        )
        evaluator.run()
        evaluator.container_runner.drop_caches.assert_not_called()
