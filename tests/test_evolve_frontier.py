"""Budget-cell scoring of evolved programs (tools/evolve/frontier.py)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evolve"))
from frontier import FLOOR, Cells, Point, pages_at, score  # noqa: E402

CELLS = Cells(dram_tiers_mb=[32, 640], recall_targets=[0.90, 0.95], dram_margin_mb=6)


def pt(recall: float, pages: float, dram: float, label: str = "p") -> Point:
    return Point(system="s", label=label, miss=1 - recall, pages=pages, dram_mb=dram, index_gb=1)


def test_pages_at_interpolates_log_log_between_front_neighbours() -> None:
    pts = [pt(0.80, 10, 1), pt(0.98, 100, 1)]
    pages, hi = pages_at(pts, 0.90)
    # miss 0.2 -> 0.02 is one decade; 0.1 is log10(2) of the way along it.
    t = math.log(0.2 / 0.1) / math.log(0.2 / 0.02)
    assert pages == pytest.approx(10 * (100 / 10) ** t)
    assert hi.miss == pytest.approx(0.02)


def test_pages_at_ignores_dominated_points_and_does_not_extrapolate() -> None:
    pts = [pt(0.80, 10, 1), pt(0.85, 50, 1), pt(0.84, 60, 1), pt(0.95, 80, 1)]
    assert pages_at(pts, 0.96) is None
    assert pages_at(pts, 0.95)[0] == pytest.approx(80)
    # Only reached from above: the cheapest point at or above the target.
    assert pages_at([pt(0.97, 40, 1)], 0.90)[0] == pytest.approx(40)


def test_placing_a_point_just_above_the_target_gains_nothing() -> None:
    base = [pt(0.85, 20, 1), pt(0.95, 60, 1)]
    exact = pages_at(base, 0.90)[0]
    # A candidate on the same curve, sampled right at the target, ties.
    assert pages_at([pt(0.90, exact, 1)], 0.90)[0] == pytest.approx(exact)


def test_score_is_best_covered_gain_and_reports_uncovered_cells() -> None:
    baselines = [pt(0.85, 20, 300, "big"), pt(0.96, 40, 300, "big")]
    cand = [pt(0.85, 200, 2, "c"), pt(0.96, 400, 2, "c")]
    sc = score(cand, baselines, CELLS)
    # Only the 640 MB cells have an opponent; the candidate needs 10x the pages.
    assert sc["combined_score"] == pytest.approx(math.log2(40 / 400), abs=0.01)
    assert {(u["dram_mb"], u["recall"]) for u in sc["uncovered"]} == {(32, 0.90), (32, 0.95)}


def test_dram_margin_keeps_jitter_out_of_a_budget() -> None:
    baselines = [pt(0.85, 20, 30), pt(0.96, 40, 30)]  # credited into the 32 MB cell
    cand = [pt(0.85, 10, 28), pt(0.96, 20, 28)]  # charged out of it (28 + 6 > 32)
    sc = score(cand, baselines, CELLS)
    rows = {(r["dram_mb"], r["recall"]): r for r in sc["cells"]}
    assert rows[(32, 0.95)]["covered"] and rows[(32, 0.95)]["candidate_pages"] is None
    assert rows[(640, 0.95)]["gain"] == pytest.approx(1.0)


def test_unreached_targets_fall_below_floor_by_recall_shortfall() -> None:
    baselines = [pt(0.85, 20, 300), pt(0.96, 40, 300)]
    sc = score([pt(0.80, 5, 10)], baselines, CELLS)
    assert sc["best_cell"] is None
    assert sc["combined_score"] == pytest.approx(FLOOR - 0.10)
