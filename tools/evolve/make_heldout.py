"""Build held-out evaluation sets for evolved programs (DEEP-10M, T2I-10M) and
the hidden-query split of BIGANN-10M used to validate would-be records.

For each set: the first 10M base vectors of the big-ann-benchmarks file, the first
`--queries` queries, and exact top-100 ground truth by chunked brute force (GEMM
per block + running top-k merge; ann-suite's compute_ground_truth loops per query,
too slow at 10M). Writes two directories:

  <data_dir>/<name>/          base.npy, queries.npy, ground_truth.npy  (scorer side)
  <evolve_data_dir>/<name>/   hard links to base.npy and queries.npy only

Hidden split (bigann10m-hidden): the same 10M base (hard link), queries
2000-3999 of the public 10k, and the published big-ann-benchmarks ground truth for
the 10M prefix (gt10m.bin, which matches our brute force on queries 0-1999
exactly), spot-checked here against brute force.

Usage (ann-suite root):
  uv run python tools/evolve/make_heldout.py deep10m-q2k
  uv run python tools/evolve/make_heldout.py t2i10m-q2k
  uv run python tools/evolve/make_heldout.py bigann10m-hidden
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
SPLITS = {
    # name: (set whose base.npy is reused, query .u8bin, published GT .bin, offset)
    "bigann10m-hidden": ("bigann10m-q2k", "bigann/query.u8bin", "bigann/gt10m.bin", 2000),
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


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def make_split(name: str, data_dir: Path, evolve_dir: Path, nq: int) -> None:
    base_set, query_rel, gt_rel, off = SPLITS[name]
    out = data_dir / name
    out.mkdir(parents=True, exist_ok=True)
    _link(data_dir / base_set / "base.npy", out / "base.npy")
    base = np.load(out / "base.npy", mmap_mode="r")
    queries = np.fromfile(data_dir / query_rel, dtype=np.uint8, offset=8).reshape(-1, base.shape[1])
    queries = np.ascontiguousarray(queries[off : off + nq])
    n, k = (int(x) for x in np.fromfile(data_dir / gt_rel, dtype=np.uint32, count=2))
    gt = np.fromfile(data_dir / gt_rel, dtype=np.uint32, offset=8, count=n * k).reshape(n, k)
    gt = gt[off : off + nq].astype(np.int32)
    rng = np.random.default_rng(0)
    check = np.sort(rng.choice(nq, size=20, replace=False))
    exact = exact_topk(base, queries[check], 10, "L2")
    q = queries[check].astype(np.float32)
    for i, j in enumerate(check):  # compare distances: ties may order ids differently
        d_pub = ((base[gt[j, :10]].astype(np.float32) - q[i]) ** 2).sum(1)
        d_ref = ((base[exact[i]].astype(np.float32) - q[i]) ** 2).sum(1)
        if not np.allclose(np.sort(d_pub), np.sort(d_ref)):
            raise SystemExit(f"published ground truth disagrees with brute force at query {j}")
    np.save(out / "queries.npy", queries)
    np.save(out / "ground_truth.npy", gt)
    ev = evolve_dir / name
    ev.mkdir(parents=True, exist_ok=True)
    for f in ("base.npy", "queries.npy"):
        _link(out / f, ev / f)
    print(f"wrote {out} (queries {off}-{off + nq - 1}, GT spot-checked 20/20) and {ev}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(SETS) + sorted(SPLITS))
    ap.add_argument("--data-dir", type=Path, default=Path("/home/isfcr/data"))
    ap.add_argument("--evolve-data-dir", type=Path, default=Path("/home/isfcr/evolve_data"))
    ap.add_argument("--base-count", type=int, default=10_000_000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--k", type=int, default=100)
    ns = ap.parse_args()
    if ns.name in SPLITS:
        make_split(ns.name, ns.data_dir, ns.evolve_data_dir, ns.queries)
        return
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
