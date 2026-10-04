"""Re-scoring stored evolve reports against a changed frontier (evolve_bench rescore)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evolve"))
from evolve_bench import FAIL, _rescore_report  # noqa: E402
from frontier import Point  # noqa: E402

CFG = {"score": {"dram_tiers_mb": [640], "recall_targets": [0.90], "dram_margin_mb": 6}}


def stage(pages: float, ok: bool = True) -> dict:
    pts = [
        {
            "point": i,
            "ok": True,
            "params": {},
            "recall": r,
            "pages": pages * f,
            "peak_anon_mb": 100.0,
            "rounds": 1.0,
            "cpu_ms_per_query": 1.0,
        }
        for i, (r, f) in enumerate([(0.85, 0.5), (0.95, 1.0)])
    ]
    return {"ok": ok, "index_bytes": 10**9, "points": pts}


def base(pages: float) -> list[Point]:
    return [
        Point(system="b", label=f"b:{r}", miss=1 - r, pages=pages * f, dram_mb=50, index_gb=1)
        for r, f in [(0.85, 0.5), (0.95, 1.0)]
    ]


def test_rescore_tracks_the_frontier_and_keeps_failures() -> None:
    r = {"combined_score": 9.0, "stages": {"full": stage(20)}}
    assert _rescore_report(CFG, r, 0.0, base(40))["combined_score"] == pytest.approx(1.0)
    assert _rescore_report(CFG, r, 0.0, base(20))["combined_score"] == pytest.approx(0.0)
    broken = {"combined_score": FAIL, "stages": {"sanity": stage(20)}}
    assert _rescore_report(CFG, broken, 0.0, base(40)) == {
        "combined_score": FAIL,
        "validated": False,
    }


def test_rescore_takes_the_minimum_over_stored_validation_stages() -> None:
    r = {
        "stages": {"full": stage(20)},
        "validation": {"stages": {"full": stage(20), "hidden": stage(40)}},
    }
    out = _rescore_report(CFG, r, 0.0, base(40))
    assert out["validated"]
    assert out["validation"]["scores"] == pytest.approx(
        {"first": 1.0, "full_rerun": 1.0, "hidden": 0.0}
    )
    assert out["combined_score"] == pytest.approx(0.0)
    r["validation"]["stages"]["hidden"]["ok"] = False
    assert _rescore_report(CFG, r, 0.0, base(40))["combined_score"] == FAIL
