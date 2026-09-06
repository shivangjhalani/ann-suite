"""Tests for build parameter sweeps and index reuse.

Covers:
- build_combo_slug: determinism, order-insensitivity, uniqueness, sanitization
- BuildConfig.reuse_index schema field
- Evaluator loop restructure: build-once/search-many with reuse_index=True
- Legacy behavior: rebuild per point with reuse_index=False
- Build x search cartesian product pairing
- Build failure fan-out (one failed result per search combo)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ann_suite.core.config import load_config
from ann_suite.core.schemas import (
    AlgorithmConfig,
    BenchmarkConfig,
    DatasetConfig,
    PhaseResult,
    ResourceSummary,
)
from ann_suite.evaluator import BenchmarkEvaluator, build_combo_slug, expand_sweep_params


def make_resource_summary() -> ResourceSummary:
    """Minimal valid ResourceSummary for mocked phases."""
    return ResourceSummary(
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
    )


def make_phase_result(success: bool = True, output: dict[str, Any] | None = None) -> PhaseResult:
    """Minimal PhaseResult for mocked container runs."""
    return PhaseResult(
        phase="build",
        success=success,
        duration_seconds=1.0,
        resources=make_resource_summary(),
        output=output or {"status": "success", "index_size_bytes": 100},
    )


def make_container_result(success: bool = True) -> SimpleNamespace:
    """Mimics ContainerRunner.ContainerResult for mocking run_phase."""
    return SimpleNamespace(
        success=success,
        exit_code=0 if success else 1,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        output={"status": "success" if success else "error", "index_size_bytes": 100},
        error_message=None if success else "build failed",
        warmup_resources=None,
        stdout_path=None,
        stderr_path=None,
    )


def make_evaluator(tmp_path: Path) -> tuple[BenchmarkEvaluator, MagicMock]:
    """Create an evaluator with a mocked ContainerRunner; returns (evaluator, mock)."""
    config = BenchmarkConfig(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        index_dir=tmp_path / "indices",
    )
    evaluator = BenchmarkEvaluator(config)
    evaluator.container_runner = MagicMock()
    evaluator.container_runner.pull_image.return_value = True
    return evaluator, evaluator.container_runner


def make_algo(name: str = "TestAlgo", **kwargs: Any) -> AlgorithmConfig:
    return AlgorithmConfig(name=name, docker_image="test:latest", **kwargs)


def make_dataset(name: str = "ds") -> DatasetConfig:
    return DatasetConfig(name=name, base_path=Path("base.npy"), dimension=8)


class TestBuildComboSlug:
    def test_empty_args_produce_default(self) -> None:
        assert build_combo_slug({}) == "default"

    def test_deterministic(self) -> None:
        assert build_combo_slug({"R": 64, "L": 100}) == build_combo_slug({"R": 64, "L": 100})

    def test_order_insensitive(self) -> None:
        assert build_combo_slug({"a": 1, "b": 2}) == build_combo_slug({"b": 2, "a": 1})

    def test_distinct_combos_get_distinct_slugs(self) -> None:
        slugs = {build_combo_slug({"R": r, "L": degree}) for r in (32, 64) for degree in (50, 100)}
        assert len(slugs) == 4

    def test_sanitizes_special_characters(self) -> None:
        slug = build_combo_slug({"alpha": 1.2, "path/to": "some value"})
        assert "/" not in slug
        assert " " not in slug
        assert slug.startswith("alpha-1-2_path-to-some-value-")

    def test_readable_prefix_contains_sorted_pairs(self) -> None:
        slug = build_combo_slug({"R": 64, "L": 100})
        assert slug.startswith("L-100_R-64-")

    def test_long_values_truncated_but_unique(self) -> None:
        long_a = {"key": "a" * 200}
        long_b = {"key": "b" * 200}
        slug_a, slug_b = build_combo_slug(long_a), build_combo_slug(long_b)
        assert len(slug_a) < 80
        assert slug_a != slug_b

    def test_non_scalar_values_via_json_fallback(self) -> None:
        slug = build_combo_slug({"layers": [1, 2, 3]})
        assert slug != "default"
        assert build_combo_slug({"layers": [1, 2, 3]}) == slug


class TestBuildSweepExpansion:
    def test_build_args_expand(self) -> None:
        combos = expand_sweep_params({"R": [32, 64], "L": 100})
        assert combos == [{"R": 32, "L": 100}, {"R": 64, "L": 100}]

    def test_build_and_search_product(self) -> None:
        builds = expand_sweep_params({"R": [32, 64]})
        searches = expand_sweep_params({"Ls": [10, 20, 30]})
        assert len(builds) * len(searches) == 6


class TestReuseIndexSchema:
    def test_default_is_true(self) -> None:
        algo = make_algo()
        assert algo.build.reuse_index is True

    def test_explicit_false(self) -> None:
        algo = make_algo(build={"reuse_index": False})
        assert algo.build.reuse_index is False

    def test_prebuilt_path_is_optional(self, tmp_path: Path) -> None:
        algo = make_algo(build={"prebuilt_path": tmp_path / "index"})
        assert algo.build.prebuilt_path == tmp_path / "index"

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
name: "reuse test"
algorithms:
  - name: DiskANN
    docker_image: ann-suite/diskann:latest
    algorithm_type: disk
    build:
      args:
        R: [32, 64]
      reuse_index: false
    search:
      k: 10
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    dimension: 128
"""
        )
        config = load_config(config_file)
        assert config.algorithms[0].build.reuse_index is False
        assert config.algorithms[0].build.args == {"R": [32, 64]}


class TestPrebuiltIndexes:
    def test_prebuilt_path_skips_build_and_mounts_resolved_directory(
        self, tmp_path: Path
    ) -> None:
        evaluator, container_runner = make_evaluator(tmp_path)
        prebuilt = tmp_path / "indices" / "diskann"
        prebuilt.mkdir(parents=True)
        algo = make_algo(build={"prebuilt_path": Path("diskann")})

        context = evaluator._ensure_build(algo, make_dataset(), tmp_path / "base.npy", {})

        assert context.prebuilt is True
        assert context.host_index_dir == prebuilt.resolve()
        assert context.container_index_path == "/data/prebuilt-index"
        assert context.build_result.output["prebuilt"] is True
        container_runner.run_phase.assert_not_called()

    def test_missing_prebuilt_path_fails_before_container_start(self, tmp_path: Path) -> None:
        evaluator, container_runner = make_evaluator(tmp_path)
        algo = make_algo(build={"prebuilt_path": Path("missing")})

        with pytest.raises(FileNotFoundError, match="Prebuilt index directory"):
            evaluator._ensure_build(algo, make_dataset(), tmp_path / "base.npy", {})

        container_runner.run_phase.assert_not_called()

    def test_prebuilt_file_symlinks_add_external_mounts(self, tmp_path: Path) -> None:
        evaluator, _ = make_evaluator(tmp_path)
        source = tmp_path / "external-index"
        source.mkdir()
        (source / "ann_disk.index").write_bytes(b"index")
        wrapper = tmp_path / "indices" / "wrapper"
        wrapper.mkdir(parents=True)
        (wrapper / "ann_disk.index").symlink_to(source / "ann_disk.index")
        algo = make_algo(build={"prebuilt_path": wrapper})

        context = evaluator._ensure_build(algo, make_dataset(), tmp_path / "base.npy", {})

        assert context.prebuilt_additional_volumes == {
            str(source): {"bind": str(source), "mode": "rw"}
        }


class TestEnsureBuildCaching:
    def _evaluator(self, tmp_path: Path) -> tuple[BenchmarkEvaluator, MagicMock]:
        evaluator, runner = make_evaluator(tmp_path)
        evaluator.container_runner.run_phase.return_value = (
            make_container_result(success=True),
            make_resource_summary(),
        )
        return evaluator, runner

    def test_reuse_enabled_builds_once(self, tmp_path: Path) -> None:
        evaluator, runner = self._evaluator(tmp_path)
        algo = make_algo()
        dataset = make_dataset()

        ctx1 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})
        ctx2 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})

        assert runner.run_phase.call_count == 1
        assert ctx1 is ctx2

    def test_reuse_disabled_rebuilds(self, tmp_path: Path) -> None:
        evaluator, runner = self._evaluator(tmp_path)
        algo = make_algo(build={"reuse_index": False})
        dataset = make_dataset()

        evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})
        evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})

        assert runner.run_phase.call_count == 2

    def test_different_combos_not_shared(self, tmp_path: Path) -> None:
        evaluator, runner = self._evaluator(tmp_path)
        algo = make_algo()
        dataset = make_dataset()

        ctx_r32 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 32})
        ctx_r64 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})

        assert runner.run_phase.call_count == 2
        assert ctx_r32.host_index_dir != ctx_r64.host_index_dir
        assert ctx_r32.container_index_path != ctx_r64.container_index_path

    def test_failed_build_not_cached(self, tmp_path: Path) -> None:
        evaluator, runner = make_evaluator(tmp_path)
        runner.run_phase.return_value = (
            make_container_result(success=False),
            make_resource_summary(),
        )
        algo = make_algo()
        dataset = make_dataset()

        ctx1 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})
        ctx2 = evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64})

        assert not ctx1.build_result.success
        assert ctx1 is not ctx2
        assert runner.run_phase.call_count == 2

    def test_container_receives_slugged_paths_and_args(self, tmp_path: Path) -> None:
        evaluator, runner = self._evaluator(tmp_path)
        algo = make_algo()
        dataset = make_dataset()

        evaluator._ensure_build(algo, dataset, tmp_path / "base.npy", {"R": 64, "L": 100})

        call_kwargs = runner.run_phase.call_args.kwargs
        config = call_kwargs["config"]
        expected_suffix = f"/{algo.name}/{dataset.name}/{build_combo_slug({'R': 64, 'L': 100})}"
        assert config["index_path"].endswith(expected_suffix)
        assert config["build_args"] == {"R": 64, "L": 100}


class TestRunLoopNesting:
    """Integration-style tests over evaluator.run() with fully mocked containers."""

    BASE_VECTORS = np.zeros((10, 8), dtype=np.float32)
    QUERIES = np.zeros((3, 8), dtype=np.float32)

    def _run(
        self,
        tmp_path: Path,
        algo: AlgorithmConfig,
        build_success: bool = True,
    ) -> tuple[list[Any], MagicMock]:
        evaluator, runner = make_evaluator(tmp_path)

        def run_side_effect(**kwargs: Any) -> tuple[SimpleNamespace, ResourceSummary]:
            mode = kwargs["mode"]
            if mode == "build":
                return make_container_result(success=build_success), make_resource_summary()
            return make_container_result(success=True), make_resource_summary()

        runner.run_phase.side_effect = run_side_effect
        evaluator._prepare_dataset = MagicMock(  # type: ignore[method-assign]
            return_value=(self.BASE_VECTORS.copy(), self.QUERIES.copy(), None)
        )

        config = BenchmarkConfig(
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            index_dir=tmp_path / "indices",
            algorithms=[algo],
            datasets=[make_dataset()],
        )
        evaluator.config = config
        results = evaluator.run()
        return results, runner

    def test_build_once_search_many(self, tmp_path: Path) -> None:
        algo = make_algo(
            search={"k": 10, "args": {"Ls": [10, 20, 30]}},
        )
        results, runner = self._run(tmp_path, algo)

        modes = [call.kwargs["mode"] for call in runner.run_phase.call_args_list]
        assert modes.count("build") == 1
        assert modes.count("search") == 3
        assert len(results) == 3

    def test_cartesian_pairing_recorded(self, tmp_path: Path) -> None:
        algo = make_algo(
            build={"args": {"R": [32, 64]}},
            search={"k": 10, "args": {"Ls": [10, 20]}},
        )
        results, runner = self._run(tmp_path, algo)

        pairs = {
            (r.hyperparameters["build"]["R"], r.hyperparameters["search"]["Ls"]) for r in results
        }
        assert pairs == {(32, 10), (32, 20), (64, 10), (64, 20)}
        assert len(results) == 4
        assert [c.kwargs["mode"] for c in runner.run_phase.call_args_list].count("build") == 2
        assert [c.kwargs["mode"] for c in runner.run_phase.call_args_list].count("search") == 4

    def test_legacy_mode_rebuilds_per_point(self, tmp_path: Path) -> None:
        algo = make_algo(
            build={"reuse_index": False},
            search={"k": 10, "args": {"Ls": [10, 20]}},
        )
        _, runner = self._run(tmp_path, algo)

        modes = [call.kwargs["mode"] for call in runner.run_phase.call_args_list]
        assert modes.count("build") == 2
        assert modes.count("search") == 2

    def test_search_points_share_identical_build_metrics(self, tmp_path: Path) -> None:
        algo = make_algo(search={"k": 10, "args": {"Ls": [10, 20]}})
        results, _ = self._run(tmp_path, algo)

        build_times = {r.total_build_time_seconds for r in results}
        index_sizes = {r.index_size_bytes for r in results}
        assert len(build_times) == 1
        assert len(index_sizes) == 1

    def test_distinct_build_combos_use_distinct_index_dirs(self, tmp_path: Path) -> None:
        algo = make_algo(
            build={"args": {"R": [32, 64]}},
            search={"k": 10, "args": {"Ls": 10}},
        )
        _, runner = self._run(tmp_path, algo)

        build_configs = [
            call.kwargs["config"]
            for call in runner.run_phase.call_args_list
            if call.kwargs["mode"] == "build"
        ]
        search_configs = [
            call.kwargs["config"]
            for call in runner.run_phase.call_args_list
            if call.kwargs["mode"] == "search"
        ]
        assert len({c["index_path"] for c in build_configs}) == 2
        # Each search must target the index matching its recorded build params
        slug_by_dir = {c["index_path"]: c["build_args"]["R"] for c in build_configs}
        for search_config in search_configs:
            assert search_config["index_path"] in slug_by_dir

    def test_build_failure_fans_out_one_failure_per_search_combo(self, tmp_path: Path) -> None:
        algo = make_algo(search={"k": 10, "args": {"Ls": [10, 20]}})
        results, runner = self._run(tmp_path, algo, build_success=False)

        assert len(results) == 2
        assert all(not r.search_result for r in results)
        assert all(r.build_result is not None and not r.build_result.success for r in results)
        assert all(r.hyperparameters["search"]["Ls"] in (10, 20) for r in results)
        # No search phase was ever invoked
        assert [c.kwargs["mode"] for c in runner.run_phase.call_args_list].count("search") == 0

    def test_pull_failure_emits_one_failure_per_search_combo(self, tmp_path: Path) -> None:
        evaluator, runner = make_evaluator(tmp_path)
        runner.pull_image.return_value = False
        evaluator._prepare_dataset = MagicMock(  # type: ignore[method-assign]
            return_value=(self.BASE_VECTORS.copy(), self.QUERIES.copy(), None)
        )

        algo = make_algo(search={"k": 10, "args": {"Ls": [10, 20]}})
        evaluator.config = BenchmarkConfig(
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            index_dir=tmp_path / "indices",
            algorithms=[algo],
            datasets=[make_dataset()],
        )
        results = evaluator.run()

        assert len(results) == 2
        assert all(r.build_result is None and r.search_result is None for r in results)
        runner.run_phase.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, 1),
        ({"ef": [50, 100]}, 2),
        ({"ef": [50, 100], "Ls": [10, 20, 30]}, 6),
    ],
)
def test_expand_sweep_params_counts(args: dict[str, Any], expected: int) -> None:
    assert len(expand_sweep_params(args)) == expected
