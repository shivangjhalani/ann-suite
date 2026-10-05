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

That is score v1 (`score`). Score v2 (`score_v2`, docs/EVOLVE_SCORE.md) keeps the
cells but compares throughput and latency instead of pages; config `score.version`
selects one.
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


# ------------------------------------------------------------------- score v2
#
# docs/EVOLVE_SCORE.md. Every point carries latency_ms (mean per query, single
# thread: modelled device time per I/O round + CPU for harness programs, measured for
# the C++ baselines) and cpu_ms (CPU per query on one thread) in `extra`. In each
# (DRAM, recall) cell the gain is the mean of log2(throughput / best known
# throughput) and log2(best known latency / latency) at one operating point.


@dataclass
class Host:
    """The benchmark host as score v2 sees it (results/evolve/ssd_model.json)."""

    batch_ms: list[tuple[int, float]]  # (p, ms to complete p parallel 4 KB reads)
    iops_max: float  # saturated random 4 KB reads per second
    cores: int  # cores searches are pinned to

    @classmethod
    def load(cls, path: Path, cores: int) -> Host:
        d = json.loads(Path(path).read_text())
        return cls([(int(p), float(ms)) for p, ms in d["batch_ms"]], float(d["iops_max"]), cores)

    def round_ms(self, pages: float) -> float:
        """Device time of one I/O round of `pages` distinct pages: interpolated
        between calibrated batch sizes, at the saturated rate beyond the largest."""
        if pages <= 0:
            return 0.0
        xs = [float(p) for p, _ in self.batch_ms]
        ys = [ms for _, ms in self.batch_ms]
        if pages >= xs[-1]:
            return ys[-1] + (pages - xs[-1]) * 1000.0 / self.iops_max
        return float(np.interp(pages, xs, ys))

    def io_ms(self, round_hist: dict[str, int] | None, queries: int, rounds: float,
              pages: float) -> float:  # fmt: skip
        """Mean device time per query: exact from the per-round histogram {pages:
        rounds} when there is one, else from mean rounds and pages (equal rounds)."""
        if round_hist:
            total = sum(n * self.round_ms(float(p)) for p, n in round_hist.items())
            return total / max(queries, 1)
        return rounds * self.round_ms(pages / rounds) if rounds > 0 else 0.0

    def throughput(self, p: Point) -> float:
        """Queries/s of the host: SSD reads or CPU, whichever runs out first."""
        io = self.iops_max / p.pages if p.pages > 0 else math.inf
        cpu = self.cores * 1000.0 / max(float(p.extra["cpu_ms"]), 1e-3)
        return min(io, cpu)


def has_v2(p: Point) -> bool:
    return p.extra.get("latency_ms") is not None and p.extra.get("cpu_ms") is not None


def _config_of(p: Point) -> str:
    """One method configuration (e.g. "DiskANN-B0.1", "IVFADC+R-refine8+2"): points
    are interpolated only within a configuration."""
    return p.label.split(":")[0].split("@")[0]


def operating_points(points: list[Point], recall: float, host: Host) -> list[dict[str, Any]]:
    """Operating points of one configuration at recall >= `recall`: its points that
    reach it, plus the point at exactly `recall` interpolated log-log (in miss
    rate) between the two points around it, which carries throughput and latency
    of the same pair. Points dominated in (recall, throughput, latency) are dropped
    first; nothing is extrapolated."""
    rows = [(p, host.throughput(p), max(float(p.extra["latency_ms"]), 1e-3)) for p in points]
    front = [
        r
        for r in rows
        if not any(
            o[0].miss <= r[0].miss and o[1] >= r[1] and o[2] <= r[2]
            and (o[0].miss < r[0].miss or o[1] > r[1] or o[2] < r[2])
            for o in rows
        )
    ]  # fmt: skip
    above = [r for r in front if 1.0 - r[0].miss >= recall]
    ops = [{"q": q, "t": t, "point": p, "interpolated": False} for p, q, t in above]
    below = [r for r in front if 1.0 - r[0].miss < recall]
    if above and below:
        hi = min(above, key=lambda r: 1.0 - r[0].miss)
        lo = max(below, key=lambda r: 1.0 - r[0].miss)
        m_hi, m_lo, m = max(hi[0].miss, MIN_MISS), lo[0].miss, max(1.0 - recall, MIN_MISS)
        if m_lo > m_hi and m < m_lo:
            f = (math.log(m_lo) - math.log(m)) / (math.log(m_lo) - math.log(m_hi))
            ops.append(
                {
                    "q": math.exp(math.log(lo[1]) + f * (math.log(hi[1]) - math.log(lo[1]))),
                    "t": math.exp(math.log(lo[2]) + f * (math.log(hi[2]) - math.log(lo[2]))),
                    "point": hi[0],
                    "interpolated": True,
                }
            )
    return ops


def score_v2(
    candidates: list[Point],
    baselines: list[Point],
    cells: Cells,
    host: Host,
    floor: float = FLOOR,
) -> dict[str, Any]:
    if not candidates:
        return {"combined_score": -10.0, "cells": [], "best_cell": None, "uncovered": []}
    margin = cells.dram_margin_mb
    known = [b for b in baselines if has_v2(b)]
    configs = sorted({_config_of(b) for b in known})
    table = []
    for tier in cells.dram_tiers_mb:
        mine = [c for c in candidates if c.dram_mb + margin <= tier and has_v2(c)]
        for target in cells.recall_targets:
            opp = [
                op
                for name in configs
                for op in operating_points(
                    [b for b in known if _config_of(b) == name and b.dram_mb - margin <= tier],
                    target,
                    host,
                )
            ]
            ours = operating_points(mine, target, host)
            row: dict[str, Any] = {"dram_mb": tier, "recall": target, "covered": bool(opp)}
            if opp:
                bq = max(opp, key=lambda o: o["q"])
                bt = min(opp, key=lambda o: o["t"])
                row |= {
                    "opponent_qps": bq["q"],
                    "opponent_qps_by": bq["point"].label,
                    "opponent_latency_ms": bt["t"],
                    "opponent_latency_by": bt["point"].label,
                }
            if ours:
                if opp:
                    for o in ours:
                        o["gain"] = 0.5 * (
                            math.log2(o["q"] / bq["q"]) + math.log2(bt["t"] / o["t"])
                        )
                    best = max(ours, key=lambda o: o["gain"])
                else:
                    best = max(ours, key=lambda o: math.log2(o["q"]) - math.log2(o["t"]))
                row |= {
                    "candidate_qps": best["q"],
                    "candidate_latency_ms": best["t"],
                    "candidate_point": _brief(best["point"]),
                    "gain": best.get("gain"),
                }
            else:
                row["gain"] = None
            table.append(row)
    scored = [r for r in table if r.get("gain") is not None]
    uncovered = [
        {"dram_mb": r["dram_mb"], "recall": r["recall"], "candidate_qps": r["candidate_qps"]}
        for r in table
        if not r["covered"] and "candidate_qps" in r
    ]
    if scored:
        best_row = max(scored, key=lambda r: r["gain"])
        combined, shortfall = max(best_row["gain"], floor), 0.0
        best_cell = {k: best_row[k] for k in ("dram_mb", "recall", "gain")}
        best_cell["latency_ms"] = best_row["candidate_latency_ms"]
    else:
        best_cell = None
        eligible = [c for c in candidates if c.dram_mb + margin <= max(cells.dram_tiers_mb)]
        top = max((1.0 - c.miss for c in eligible), default=0.0)
        shortfall = max(0.0, min(cells.recall_targets) - top) if eligible else 0.9
        combined = floor - shortfall
    return {
        "combined_score": combined,
        "best_cell": best_cell,
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
        "latency_ms": p.extra.get("latency_ms"),
        "cpu_ms": p.extra.get("cpu_ms"),
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
    search_threads: int | None = None,
) -> Point:
    """Build a Point from an ann_suite BenchmarkResult.

    pages: the runner's own per-query 4 KB read count when it reports one
    (algorithm_stats.io_reads, DiskANN-family O_DIRECT sector reads), else the
    kernel io.stat pages per query for the query window (SPANN).

    With search_threads == 1 the point also gets score v2's latency_ms (the
    runner's mean latency, i.e. 1 / QPS on one thread) and cpu_ms (search CPU per
    query from the container cgroup); with more threads those would measure a loaded
    server, so they are left out.
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
            "search_threads": search_threads,
            **(_measured_v2(result) if search_threads == 1 else {}),
        },
    )


def _measured_v2(result: Any) -> dict[str, float | None]:
    lat = getattr(result, "latency", None)
    cpu = getattr(result, "cpu", None)
    return {
        "latency_ms": getattr(lat, "mean_ms", None)
        or (1000.0 / result.qps if result.qps else None),
        "cpu_ms": getattr(cpu, "search_cpu_time_per_query_ms", None),
    }
