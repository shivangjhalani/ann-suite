"""Score v2: throughput and latency in budget cells (tools/evolve/frontier.py)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evolve"))
from frontier import Cells, Host, Point, operating_points, score_v2  # noqa: E402

CELLS = Cells(dram_tiers_mb=[128], recall_targets=[0.90], dram_margin_mb=6)
# 0.1 ms for a single read, +0.01 ms per extra page in the round up to 64 pages,
# then 100k pages/s; 8 cores.
HOST = Host(batch_ms=[(1, 0.1), (64, 0.73)], iops_max=100_000.0, cores=8)


def pt(recall: float, pages: float, latency: float, cpu: float, label: str = "m:p") -> Point:
    return Point(
        system="s",
        label=label,
        miss=1 - recall,
        pages=pages,
        dram_mb=50,
        index_gb=1,
        extra={"latency_ms": latency, "cpu_ms": cpu},
    )


def test_round_time_interpolates_then_follows_the_saturated_rate() -> None:
    assert HOST.round_ms(1) == pytest.approx(0.1)
    assert HOST.round_ms(32.5) == pytest.approx(0.415)
    assert HOST.round_ms(1064) == pytest.approx(0.73 + 1000 / 100_000 * 1000)
    assert HOST.round_ms(0) == 0.0


def test_io_time_from_round_histogram_and_from_means() -> None:
    # 2 queries: one round of 1 page each, plus one round of 64 pages in total.
    assert HOST.io_ms({"1": 2, "64": 1}, 2, 0, 0) == pytest.approx((2 * 0.1 + 0.73) / 2)
    assert HOST.io_ms(None, 0, rounds=2, pages=128) == pytest.approx(2 * 0.73)


def test_throughput_is_limited_by_ssd_or_cpu() -> None:
    assert HOST.throughput(pt(0.9, 100, 1, 0.01)) == pytest.approx(1000)  # SSD-bound
    assert HOST.throughput(pt(0.9, 1, 1, 4.0)) == pytest.approx(2000)  # 8 cores / 4 ms


def test_equal_to_the_best_known_scores_zero_and_twice_better_scores_one() -> None:
    base = [pt(0.92, 100, 2.0, 1.0, "known:a")]
    same = score_v2([pt(0.92, 100, 2.0, 1.0)], base, CELLS, HOST)
    assert same["combined_score"] == pytest.approx(0.0)
    better = score_v2([pt(0.92, 50, 1.0, 0.5)], base, CELLS, HOST)
    assert better["combined_score"] == pytest.approx(1.0)


def test_best_throughput_and_best_latency_may_come_from_different_methods() -> None:
    base = [pt(0.92, 50, 4.0, 0.5, "fast-qps:a"), pt(0.92, 400, 0.5, 0.5, "fast-lat:a")]
    res = score_v2([pt(0.92, 50, 0.5, 0.5)], base, CELLS, HOST)
    row = res["cells"][0]
    assert row["opponent_qps_by"].startswith("fast-qps")
    assert row["opponent_latency_by"].startswith("fast-lat")
    assert res["combined_score"] == pytest.approx(0.0)  # matches the ideal point
    worse = score_v2([pt(0.92, 100, 1.0, 0.5)], base, CELLS, HOST)
    assert worse["combined_score"] == pytest.approx(0.5 * (math.log2(0.5) + math.log2(0.5)))


def test_saving_pages_by_spending_cpu_is_not_free() -> None:
    base = [pt(0.92, 100, 2.0, 0.5, "known:a")]
    # Half the pages for 20x the CPU: CPU-bound throughput and longer latency.
    res = score_v2([pt(0.92, 50, 11.0, 10.0)], base, CELLS, HOST)
    assert res["combined_score"] < 0


def test_operating_point_interpolates_within_a_configuration_only() -> None:
    pts = [pt(0.80, 10, 1.0, 0.1, "a:p0"), pt(0.98, 100, 4.0, 0.1, "a:p1")]
    ops = operating_points(pts, 0.90, HOST)
    interp = [o for o in ops if o["interpolated"]]
    assert len(interp) == 1
    f = math.log(0.2 / 0.1) / math.log(0.2 / 0.02)
    assert interp[0]["t"] == pytest.approx(1.0 * 4.0**f)
    # Across configurations nothing is interpolated: "b" alone cannot reach 0.90.
    res = score_v2([pt(0.98, 100, 4.0, 0.1)], [pt(0.80, 1, 0.1, 0.01, "b:p0")], CELLS, HOST)
    assert res["cells"][0]["covered"] is False


def test_points_without_v2_costs_are_ignored_as_opponents() -> None:
    old = Point(system="s", label="old:p", miss=0.05, pages=1, dram_mb=10, index_gb=1)
    res = score_v2([pt(0.92, 100, 2.0, 1.0)], [old], CELLS, HOST)
    assert res["cells"][0]["covered"] is False
    assert res["combined_score"] < 0
