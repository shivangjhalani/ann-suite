"""Frontier-gain scoring for evolved disk-ANN programs.

A benchmark point is a vector of costs, all minimized:
    miss      1 - recall@10 (recall recomputed from returned ids where possible)
    pages     4 KB SSD page reads per query
    dram_mb   search-phase peak anonymous memory minus the image's runtime floor
    index_gb  index size on SSD + DRAM-resident index files

Each axis is mapped to [0, 1] in log space over a fixed box (configs/evolve/*.yaml);
points beyond the box's upper edge contribute no volume. The baseline frontier is
the set of points of the published systems (PipeANN, DiskANN, SPANN, Starling,
PageANN, LAANN) measured in ann-suite under the same DRAM cap and queries.

Score of a candidate (its points C) against baselines B:
    hv_gain  = (HV(B u C) - HV(B)) / HV(B)        if C adds any volume (> 0)
    distance = min_c max_b min_i (c_i - b_i)^+    otherwise: the uniform log-space
               improvement the best candidate point needs to escape domination
    combined_score = hv_gain if hv_gain > 0 else -distance
so the score is positive exactly when the candidate extends the frontier in any
direction, and still has a gradient while it is dominated. HV is a fixed-seed
Monte Carlo estimate (deterministic for a given box and sample count).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("miss", "pages", "dram_mb", "index_gb")


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
class Box:
    lo: dict[str, float]
    hi: dict[str, float]
    samples: int = 1_000_000
    seed: int = 12345

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Box:
        return cls(
            lo={a: float(cfg[a][0]) for a in AXES},
            hi={a: float(cfg[a][1]) for a in AXES},
            samples=int(cfg.get("samples", 1_000_000)),
            seed=int(cfg.get("seed", 12345)),
        )

    def normalize(self, points: list[Point], clip: bool = True) -> np.ndarray:
        """Log-space position in the box; 0 = best edge, 1 = worst edge."""
        out = np.empty((len(points), len(AXES)))
        for j, a in enumerate(AXES):
            lo, hi = math.log(self.lo[a]), math.log(self.hi[a])
            vals = np.array([max(getattr(p, a), self.lo[a] * 1e-3) for p in points], dtype=float)
            out[:, j] = (np.log(vals) - lo) / (hi - lo)
        return np.clip(out, 0.0, None) if clip else out

    def _samples(self) -> np.ndarray:
        return np.random.default_rng(self.seed).random((self.samples, len(AXES)))


def dominated_mask(u: np.ndarray, samples: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Samples dominated by at least one point of u (all coordinates <=)."""
    mask = np.zeros(samples.shape[0], dtype=bool)
    if u.size == 0:
        return mask
    inside = u[(u <= 1.0).all(axis=1)]
    for s in range(0, samples.shape[0], chunk):
        block = samples[s : s + chunk]
        for p in inside:
            mask[s : s + chunk] |= (block >= p).all(axis=1)
    return mask


def pareto(points: list[Point]) -> list[Point]:
    c = np.array([p.costs() for p in points])
    keep = []
    for i in range(len(points)):
        dom = ((c <= c[i]).all(axis=1) & (c < c[i]).any(axis=1)).any()
        if not dom:
            keep.append(points[i])
    return keep


def score(candidates: list[Point], baselines: list[Point], box: Box) -> dict[str, Any]:
    if not candidates:
        return {"combined_score": -10.0, "hv_gain": 0.0, "distance": None, "points": []}
    samples = box._samples()
    ub = box.normalize(baselines)
    uc = box.normalize(candidates)
    base_mask = dominated_mask(ub, samples)
    both_mask = base_mask | dominated_mask(uc, samples)
    hv_base = base_mask.mean()
    hv_gain = float((both_mask.sum() - base_mask.sum()) / max(1, base_mask.sum()))

    # Unclipped coordinates so moving toward the box from outside still counts.
    ucu = box.normalize(candidates, clip=False)
    ubu = box.normalize(baselines, clip=False)
    per_point = []
    for i, p in enumerate(candidates):
        diff = ucu[i][None, :] - ubu  # >0 where the candidate is worse
        need = np.clip(diff.min(axis=1), 0.0, None)  # escape baseline b: beat it on 1 axis
        worst = int(need.argmax())
        nearest = int(np.abs(diff).sum(axis=1).argmin())
        # A point outside the box (e.g. recall < 0.8) adds no volume even when
        # undominated; charge its distance to the box so it still has a gradient.
        outside = float(np.clip(ucu[i] - 1.0, 0.0, None).max())
        per_point.append(
            {
                "label": p.label,
                **{a: getattr(p, a) for a in AXES},
                "escape_distance": max(float(need.max()), outside),
                "outside_box_by": outside,
                "dominated_by": baselines[worst].label if need.max() > 0 else None,
                "nearest_baseline": {
                    "label": baselines[nearest].label,
                    **{a: getattr(baselines[nearest], a) for a in AXES},
                },
                "in_box": bool((ucu[i] <= 1.0).all()),
            }
        )
    distance = min(pp["escape_distance"] for pp in per_point)
    # hv_gain > 0 needs an in-box, undominated point, whose distance is 0, so the
    # two branches meet at 0 and the score is continuous across the frontier.
    combined = hv_gain if hv_gain > 0 else -distance
    return {
        "combined_score": combined,
        "hv_gain": hv_gain,
        "distance": distance,
        "hv_baseline": float(hv_base),
        "points": per_point,
    }


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
