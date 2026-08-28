"""Tests for dashboard export and hyperparameter/algorithm-stats flattening."""

from pathlib import Path

from ann_suite.core.schemas import AlgorithmStats, BenchmarkResult, _flatten_dict
from ann_suite.export import export_dashboard_json, to_dashboard_rows
from ann_suite.results.storage import ResultsStorage


class TestFlattenDict:
    def test_nested_dicts_flatten_with_sep(self) -> None:
        assert _flatten_dict({"build": {"M": 16}, "k": 10}) == {"build_M": 16, "k": 10}

    def test_scalars_and_lists_preserved(self) -> None:
        assert _flatten_dict({"a": [1, 2], "b": "x"}) == {"a": [1, 2], "b": "x"}

    def test_deep_nesting(self) -> None:
        assert _flatten_dict({"a": {"b": {"c": 1}}}) == {"a_b_c": 1}


class TestToFlatDict:
    def test_hp_columns(self) -> None:
        result = BenchmarkResult(
            algorithm="hnsw",
            dataset="sift1m",
            hyperparameters={"build": {"M": 16}, "search": {"ef": 100}, "k": 10},
        )
        flat = result.to_flat_dict()
        assert flat["hp_build_M"] == 16
        assert flat["hp_search_ef"] == 100
        assert flat["hp_k"] == 10

    def test_algorithm_stats_flattened_to_stats_columns(self) -> None:
        result = BenchmarkResult(
            algorithm="hnsw",
            dataset="sift1m",
            algorithm_stats=AlgorithmStats(
                distance_computations=1000,
                hops=200,
                extra={"custom_counter": 42.5},
            ),
        )
        flat = result.to_flat_dict()
        assert flat["stats_distance_computations"] == 1000
        assert flat["stats_hops"] == 200
        assert flat["stats_custom_counter"] == 42.5


class TestDashboardRows:
    def test_run_name_tagged(self) -> None:
        rows = to_dashboard_rows(
            [BenchmarkResult(algorithm="hnsw", dataset="sift1m")], run_name="exp"
        )
        assert rows[0]["run_name"] == "exp"

    def test_complex_values_json_encoded(self) -> None:
        result = BenchmarkResult(
            algorithm="hnsw",
            dataset="sift1m",
            hyperparameters={"build": {"M": 16}, "search": {"ef": [50, 100]}, "k": 10},
        )
        rows = to_dashboard_rows([result], run_name="exp")
        # list-valued hyperparameter leaf is JSON-encoded with a _raw twin
        assert "hp_search_ef" in rows[0]
        assert rows[0]["hp_search_ef_raw"] == [50, 100]


class TestExportDashboardJson:
    def test_exports_across_runs(self, tmp_path: Path) -> None:
        storage = ResultsStorage(tmp_path)
        storage.save([BenchmarkResult(algorithm="a", dataset="d1")], run_name="run1")
        storage.save([BenchmarkResult(algorithm="b", dataset="d2")], run_name="run2")

        out = export_dashboard_json(tmp_path)
        assert out.name == "dashboard_data.json"

        import json

        data = json.loads(out.read_text())
        assert len(data) == 2
        run_names = {row["run_name"] for row in data}
        assert len(run_names) == 2
        assert all(name.startswith("run") for name in run_names)
