"""Budget-cell scoring for evolved disk-ANN programs.

A benchmark point is a vector of costs, all minimized:
    miss      1 - recall@10 (recall recomputed from returned ids where possible)
    pages     4 KB SSD page reads per query
    dram_mb   search-phase peak anonymous memory minus the image's runtime floor
    index_gb  index size on SSD + DRAM-resident index files

Scoring follows big-ann-benchmarks practice: fixed budgets, fixed accuracy
targets, and comparison with the best known method under the same budget. A cell
is (DRAM budget T, recall target R). In each cell the opponent value is the fewest
pages/query any baseline (published systems and textbook reference designs)
needs to reach recall R with DRAM <= T, and the candidate value is the same for
the candidate's points:

    gain(T, R) = log2(opponent pages / candidate pages)     (> 0: fewer reads)
    combined_score = max over covered cells of gain, floored at FLOOR

A cell is covered only if some baseline reaches R within T. Cells without an
opponent give no credit (they are reported so a reference can be added): an
empty region of the trade-off space means nobody tried, not that it is hard.
Both sides are measured with the same rule: pages at exactly R come from the 2-D
Pareto front of (recall, pages), interpolated log-log in (miss, pages) between
the two front points around R and never extrapolated, so a sweep point placed
just above R gains nothing over a coarser sweep. DRAM jitter (a few MB of
allocator noise) is absorbed by a margin charged against the candidate and
credited to the baselines. If no candidate point reaches the lowest target
within the largest budget, the score is FLOOR minus the recall shortfall.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("miss", "pages", "dram_mb", "index_gb")
FLOOR = -4.0  # 16x more pages than the opponent; worse is not distinguished
MIN_MISS = 1e-4


@dataclass
class Point:
    system: str
    label: str
    miss: float
    pages: float
    dram_mb: float
    index_gb: float
    extra: dict[str, Any] = field(default_factory=dict)

    def costs(self) -> list[float]:
        return [self.miss, self.pages, self.dram_mb, self.index_gb]


@dataclass
class Cells:
    dram_tiers_mb: list[float]
    recall_targets: list[float]
    dram_margin_mb: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Cells:
        return cls(
            dram_tiers_mb=[float(t) for t in cfg["dram_tiers_mb"]],
            recall_targets=[float(r) for r in cfg["recall_targets"]],
            dram_margin_mb=float(cfg["dram_margin_mb"]),
        )


def pages_at(points: list[Point], recall: float) -> tuple[float, Point] | None:
    """Fewest pages/query to reach `recall` along the points' (recall, pages)
    Pareto front, interpolated log-log between the neighbours of `recall`; None
    if no point reaches it. Returns (pages, the front point at or above recall)."""
    front: list[Point] = []  # recall descending, pages strictly descending
    for p in sorted(points, key=lambda p: (p.miss, p.pages)):
        if not front or p.pages < front[-1].pages:
            front.append(p)
    above = [p for p in front if 1.0 - p.miss >= recall]
    if not above:
        return None
    hi = above[-1]
    below = [p for p in front if 1.0 - p.miss < recall]
    if not below or 1.0 - hi.miss == recall:
        return hi.pages, hi
    lo = below[0]
    m_hi, m_lo, m = max(hi.miss, MIN_MISS), lo.miss, max(1.0 - recall, MIN_MISS)
    if m_lo <= m_hi:
        return hi.pages, hi
    t = (math.log(m_lo) - math.log(m)) / (math.log(m_lo) - math.log(m_hi))
    return math.exp(math.log(lo.pages) + t * (math.log(hi.pages) - math.log(lo.pages))), hi


def score(candidates: list[Point], baselines: list[Point], cells: Cells) -> dict[str, Any]:
    if not candidates:
        return {"combined_score": -10.0, "cells": [], "best_cell": None, "uncovered": []}
    margin = cells.dram_margin_mb
    table = []
    for tier in cells.dram_tiers_mb:
        opp = [b for b in baselines if b.dram_mb - margin <= tier]
        mine = [c for c in candidates if c.dram_mb + margin <= tier]
        for target in cells.recall_targets:
            o = pages_at(opp, target)
            c = pages_at(mine, target)
            table.append(
                {
                    "dram_mb": tier,
                    "recall": target,
                    "covered": o is not None,
                    "opponent_pages": o[0] if o else None,
                    "opponent": o[1].label if o else None,
                    "opponent_point": _brief(o[1]) if o else None,
                    "candidate_pages": c[0] if c else None,
                    "candidate_point": _brief(c[1]) if c else None,
                    "gain": math.log2(o[0] / c[0]) if o and c else None,
                }
            )
    scored = [row for row in table if row["gain"] is not None]
    uncovered = [
        {"dram_mb": r["dram_mb"], "recall": r["recall"], "candidate_pages": r["candidate_pages"]}
        for r in table
        if not r["covered"] and r["candidate_pages"] is not None
    ]
    if scored:
        best = max(scored, key=lambda r: r["gain"])
        combined = max(best["gain"], FLOOR)
        shortfall = 0.0
    else:
        best = None
        eligible = [c for c in candidates if c.dram_mb + margin <= max(cells.dram_tiers_mb)]
        top = max((1.0 - c.miss for c in eligible), default=0.0)
        shortfall = max(0.0, min(cells.recall_targets) - top) if eligible else 0.9
        combined = FLOOR - shortfall
    return {
        "combined_score": combined,
        "best_cell": (
            {"dram_mb": best["dram_mb"], "recall": best["recall"], "gain": best["gain"]}
            if best
            else None
        ),
        "recall_shortfall": shortfall,
        "cells": table,
        "uncovered": uncovered,
    }


def _brief(p: Point) -> dict[str, Any]:
    return {
        "label": p.label,
        "recall": 1.0 - p.miss,
        "pages": p.pages,
        "dram_mb": p.dram_mb,
        "index_gb": p.index_gb,
        "rounds": p.extra.get("rounds"),
    }


def pareto(points: list[Point]) -> list[Point]:
    c = np.array([p.costs() for p in points])
    keep = []
    for i in range(len(points)):
        dom = ((c <= c[i]).all(axis=1) & (c < c[i]).any(axis=1)).any()
        if not dom:
            keep.append(points[i])
    return keep


def save_points(points: list[Point], path: Path) -> None:
    path.write_text(json.dumps([asdict(p) for p in points], indent=1))


def load_points(path: Path) -> list[Point]:
    return [Point(**d) for d in json.loads(path.read_text())]


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    gt = gt[: ids.shape[0], :k]
    hits = sum(len(set(ids[i, :k].tolist()) & set(gt[i].tolist())) for i in range(ids.shape[0]))
    return hits / float(ids.shape[0] * k)


def point_from_result(
    result: Any,
    system: str,
    floor_mb: float,
    nq: int | None = None,
    recall: float | None = None,
    pages: float | None = None,
) -> Point:
    """Build a Point from an ann_suite BenchmarkResult.

    pages: the runner's own per-query 4 KB read count when it reports one
    (algorithm_stats.io_reads, DiskANN-family O_DIRECT sector reads), else the
    kernel io.stat pages per query for the query window (SPANN).
    """
    out = (result.search_result.output if result.search_result else None) or {}
    nq = nq or int(out.get("total_queries") or 0) or 1
    if pages is None:
        stats = result.algorithm_stats
        if stats is not None and stats.io_reads:
            pages = stats.io_reads / nq
        else:
            pages = float(result.disk_io.search_pages_per_query or 0.0)
    rec = recall if recall is not None else float(result.recall or 0.0)
    dram = max(0.0, float(result.memory.search_peak_anon_mb or 0.0) - floor_mb)
    hp = result.hyperparameters or {}
    label = f"{result.algorithm}:{json.dumps(hp.get('search', {}), sort_keys=True)}"
    return Point(
        system=system,
        label=label,
        miss=max(0.0, 1.0 - rec),
        pages=float(pages),
        dram_mb=dram,
        index_gb=float(result.index_size_bytes or 0) / 1e9,
        extra={
            "qps": result.qps,
            "recall": rec,
            "io_stat_pages_per_query": result.disk_io.search_pages_per_query,
            "peak_anon_mb": result.memory.search_peak_anon_mb,
            "hops_per_query": (
                result.algorithm_stats.hops_per_query if result.algorithm_stats else None
            ),
        },
    )
