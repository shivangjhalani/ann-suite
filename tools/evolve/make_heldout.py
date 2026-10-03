"""Build held-out evaluation sets for evolved programs (DEEP-10M, T2I-10M).

For each set: the first 10M base vectors of the big-ann-benchmarks file, the first
`--queries` queries, and exact top-100 ground truth by chunked brute force (GEMM
per block + running top-k merge; ann-suite's compute_ground_truth loops per query,
too slow at 10M). Writes two directories:

  <data_dir>/<name>/          base.npy, queries.npy, ground_truth.npy  (scorer side)
  <evolve_data_dir>/<name>/   hard links to base.npy and queries.npy only

Usage (ann-suite root):
  uv run python tools/evolve/make_heldout.py deep10m-q2k
  uv run python tools/evolve/make_heldout.py t2i10m-q2k
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SETS = {
    # name: (base .fbin, query .fbin, metric)
    "deep10m-q2k": ("deep/base100m.fbin", "deep/query.fbin", "L2"),
    "t2i10m-q2k": ("t2i-10m/base10m.fbin", "t2i-10m/query100k.fbin", "IP"),
}


def read_fbin(path: Path, count: int | None = None) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    n = int(n) if count is None else min(int(n), count)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, int(d)))


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    q = np.ascontiguousarray(queries, dtype=np.float32)
    best_d = np.full((len(q), k), np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), k), dtype=np.int64)
    qn = (q * q).sum(1)[:, None]
    step = 250_000  # 2000 x 250k float32 blocks: ~2 GB per temporary
    for s in range(0, base.shape[0], step):
        b = np.ascontiguousarray(base[s : s + step], dtype=np.float32)
        dots = q @ b.T
        d = -dots if metric == "IP" else qn - 2.0 * dots + (b * b).sum(1)[None, :]
        part = np.argpartition(d, k, axis=1)[:, :k]
        cand_d = np.concatenate([best_d, np.take_along_axis(d, part, 1)], axis=1)
        cand_i = np.concatenate([best_i, part + s], axis=1)
        keep = np.argpartition(cand_d, k, axis=1)[:, :k]
        best_d = np.take_along_axis(cand_d, keep, 1)
        best_i = np.take_along_axis(cand_i, keep, 1)
        print(f"  {s + b.shape[0]:,} / {base.shape[0]:,}", flush=True)
    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_i, order, 1).astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(SETS))
    ap.add_argument("--data-dir", type=Path, default=Path("/home/isfcr/data"))
    ap.add_argument("--evolve-data-dir", type=Path, default=Path("/home/isfcr/evolve_data"))
    ap.add_argument("--base-count", type=int, default=10_000_000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--k", type=int, default=100)
    ns = ap.parse_args()
    base_rel, query_rel, metric = SETS[ns.name]
    out = ns.data_dir / ns.name
    out.mkdir(parents=True, exist_ok=True)
    base = read_fbin(ns.data_dir / base_rel, ns.base_count)
    queries = np.array(read_fbin(ns.data_dir / query_rel, ns.queries))
    if not (out / "base.npy").exists():
        arr = np.lib.format.open_memmap(
            out / "base.npy", mode="w+", dtype=np.float32, shape=base.shape
        )
        for s in range(0, base.shape[0], 1_000_000):
            arr[s : s + 1_000_000] = base[s : s + 1_000_000]
        arr.flush()
        del arr
    np.save(out / "queries.npy", queries)
    print(f"{ns.name}: exact top-{ns.k} ({metric}) for {len(queries)} queries")
    np.save(out / "ground_truth.npy", exact_topk(base, queries, ns.k, metric))
    ev = ns.evolve_data_dir / ns.name
    ev.mkdir(parents=True, exist_ok=True)
    for f in ("base.npy", "queries.npy"):
        if (ev / f).exists():
            (ev / f).unlink()
        os.link(out / f, ev / f)
    print(f"wrote {out} (with ground truth) and {ev} (base + queries only)")


if __name__ == "__main__":
    main()
