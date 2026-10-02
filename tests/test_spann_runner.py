"""Tests for the SPANN runner's result validation (runner is loaded by file path)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "library/algorithms/spann/algorithm/runner.py"
# The runner does `from utils import ...` (shared helpers shipped next to it in the image).
sys.path.insert(0, str(_RUNNER_PATH.parents[2]))
_spec = importlib.util.spec_from_file_location("_spann_runner_under_test", _RUNNER_PATH)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)


class TestCheckResults:
    def test_accepts_normal_results(self) -> None:
        runner.check_results(np.arange(40).reshape(4, 10))

    def test_accepts_partially_filled_rows(self) -> None:
        indices = np.arange(1000, dtype=np.int64).reshape(100, 10)
        indices[:, 5:] = -1
        runner.check_results(indices)

    def test_rejects_empty_results(self) -> None:
        with pytest.raises(RuntimeError, match="out of memory"):
            runner.check_results(np.full((100, 10), -1, dtype=np.int64))


T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)

# Real indexsearcher output (10M SPANN index, IRN 64, 8 threads), abridged.
SEARCH_LOG = [
    (0.0, "[1] Load Vector(10000,128)\n"),
    (2.0, "[1] [query]\t\t[maxcheck]\t[avg] \t[99%] \t[95%] \t[recall] \t[qps] \t[mem]\n"),
    (5.25, "[1] 0-10000\t8192\t0.0026\t0.0032\t0.0030\t0.0000\t\t3078.5825\t\t0GB\n"),
    (5.26, "[1] 0-10000\t8192\t0.0026\t0.0032\t0.0030\t0.0000\t3078.5825\n"),
    (5.40, "[1] Output results finish!\n"),
]


def _lines(log: list[tuple[float, str]]) -> list[tuple[datetime, str]]:
    return [(T0 + timedelta(seconds=dt), line) for dt, line in log]


class TestParseRound:
    def test_uses_in_process_timing_not_process_wall(self) -> None:
        r = runner.parse_round(T0, _lines(SEARCH_LOG), "0.00259665 0.000279371 0.001521 0.005822 ")
        assert r.queries == 10000
        assert r.search_seconds == pytest.approx(10000 / 3078.5825)
        assert r.search_start == T0 + timedelta(seconds=2.0)
        # The batch line, not the summary line or result-file output, ends the search.
        assert r.search_end == T0 + timedelta(seconds=5.25)
        assert r.mean_latency_s == pytest.approx(0.00259665)
        assert r.max_latency_s == pytest.approx(0.005822)
        assert r.p99_latency_s == pytest.approx(0.0032)
        assert r.p95_latency_s == pytest.approx(0.0030)

    def test_multiple_batches_are_query_weighted(self) -> None:
        log = [
            (1.0, "[1] [query]\t\t[maxcheck]\n"),
            (2.0, "[1] 0-3000\t8192\t0.0010\t0.0020\t0.0015\t0.0000\t\t3000.0\t\t0GB\n"),
            (3.0, "[1] 3000-4000\t8192\t0.0050\t0.0060\t0.0055\t0.0000\t\t1000.0\t\t0GB\n"),
        ]
        r = runner.parse_round(T0, _lines(log), "0.001 0 0 0.004 0.005 0 0 0.009 ")
        assert r.queries == 4000
        assert r.search_seconds == pytest.approx(2.0)
        assert r.mean_latency_s == pytest.approx((3000 * 0.001 + 1000 * 0.005) / 4000)
        assert r.max_latency_s == pytest.approx(0.009)
        assert r.p99_latency_s == pytest.approx((3000 * 0.002 + 1000 * 0.006) / 4000)

    def test_missing_search_phase_fails(self) -> None:
        with pytest.raises(RuntimeError, match="search phase"):
            runner.parse_round(T0, _lines(SEARCH_LOG[:1]), "")

    def test_latency_log_mismatch_fails(self) -> None:
        with pytest.raises(RuntimeError, match="Recall-result.out"):
            runner.parse_round(T0, _lines(SEARCH_LOG), "0.1 0.2")


class TestIndexLoaderChecks:
    def _write(self, tmp_path: Path, text: str) -> Path:
        (tmp_path / "indexloader.ini").write_text(text)
        return tmp_path

    def test_thread_cap_from_persisted_ini(self, tmp_path: Path) -> None:
        d = self._write(
            tmp_path, "[Index]\nIndexAlgoType=SPANN\n[BuildSSDIndex]\nNumberOfThreads=4\n"
        )
        assert runner.search_thread_cap(d) == 4

    def test_thread_cap_defaults_to_sptag_default(self, tmp_path: Path) -> None:
        d = self._write(tmp_path, "[Index]\nIndexAlgoType=SPANN\n[BuildSSDIndex]\nResultNum=10\n")
        assert runner.search_thread_cap(d) == runner.SPTAG_DEFAULT_SSD_THREADS

    def test_head_parameters_accepted_with_duplicate_keys(self, tmp_path: Path) -> None:
        # SPANN's SaveConfig writes BuildHead and head-index keys into one section.
        d = self._write(
            tmp_path,
            "[BuildHead]\nNumberOfThreads=24\nDistCalcMethod=L2\nNumberOfThreads=24\n",
        )
        runner.check_head_parameters(d)

    def test_missing_head_parameters_rejected(self, tmp_path: Path) -> None:
        d = self._write(tmp_path, "[BuildHead]\nisExecute=false\n")
        with pytest.raises(ValueError, match="Cosine"):
            runner.check_head_parameters(d)
