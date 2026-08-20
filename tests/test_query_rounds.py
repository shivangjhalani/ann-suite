"""Regression tests for timed query repetition."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_hnsw_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "hnswlib", types.SimpleNamespace(Index=object))
    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(
            compute_recall=lambda indices, _ground_truth, _k: float(indices[0, 0])
        ),
    )

    spec = importlib.util.spec_from_file_location(
        "test_hnsw_runner_query_rounds",
        REPO_ROOT / "library/algorithms/hnsw/algorithm/runner.py",
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hnsw_query_rounds_uses_first_round_for_latency_and_all_rounds_for_qps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_hnsw_runner(monkeypatch)
    queries = np.arange(6, dtype=np.float32).reshape(3, 2)
    ground_truth = np.zeros((3, 1), dtype=np.int32)
    gt_path = tmp_path / "ground_truth.npy"
    gt_path.touch()

    def fake_np_load(path: Path) -> np.ndarray:
        if str(path).endswith("queries.npy"):
            return queries
        return ground_truth

    search_call_markers: list[int] = []

    class FakeHNSWIndex:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load(self, _index_path: Path, _dimension: int, _metric: str) -> None:
            pass

        def search(
            self,
            round_queries: np.ndarray,
            *,
            k: int,
            ef: int,
            batch_mode: bool,
        ) -> tuple[np.ndarray, np.ndarray, list[float]]:
            del ef, batch_mode
            marker = 111 if not search_call_markers else 222 + len(search_call_markers)
            search_call_markers.append(marker)
            indices = np.full((len(round_queries), k), marker, dtype=np.int32)
            distances = np.zeros((len(round_queries), k), dtype=np.float32)
            latencies = [2.0 if len(search_call_markers) == 1 else 9.0] * len(round_queries)
            return indices, distances, latencies

    perf_counter_values = iter([0.0, 0.0, 0.1, 0.2, 1.0, 1.6])

    monkeypatch.setattr(runner.np, "load", fake_np_load)
    monkeypatch.setattr(runner, "HNSWIndex", FakeHNSWIndex)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(perf_counter_values))

    result = runner.run_search(
        {
            "index_path": "/tmp/index",
            "queries_path": "/tmp/queries.npy",
            "ground_truth_path": str(gt_path),
            "dimension": 2,
            "metric": "L2",
            "k": 1,
            "query_rounds": 3,
            "batch_mode": False,
            "search_args": {"ef": 10},
        }
    )

    assert len(search_call_markers) == 3
    assert result["total_queries"] == 9
    assert result["qps"] == pytest.approx(15.0)
    assert result["mean_latency_ms"] == pytest.approx(2.0)
    assert result["p50_latency_ms"] == pytest.approx(2.0)
    assert result["p95_latency_ms"] == pytest.approx(2.0)
    assert result["p99_latency_ms"] == pytest.approx(2.0)
    assert result["max_latency_ms"] == pytest.approx(2.0)
    assert result["recall"] == pytest.approx(111.0)
