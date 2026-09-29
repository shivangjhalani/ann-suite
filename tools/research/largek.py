"""Co-location analysis swept over ground-truth depth k (10/100/1000): does
block-locality (span/probes) degrade gracefully as the recall target widens
from top-10 to top-1000?

Ported from ``~/research/py/largek.py`` on the isfcr benchmark host. Unlike
colocate.py (which fixes k=10 and sweeps group size), this fixes the group
sizes of interest and sweeps k, using a brute-force top-1000 ground truth
computed on the fly (the shipped ground truth is usually only top-100).

Usage:

    uv run python tools/research/largek.py \\
        --base data/sift1m/base.npy --queries data/sift1m/queries.npy \\
        --dim 128 --num-queries 2000
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=__import__("pathlib").Path)
    parser.add_argument("--queries", required=True, type=__import__("pathlib").Path)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=2000)
    parser.add_argument("--max-gt-depth", type=int, default=1000)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--num-probe-candidates", type=int, default=2048)
    parser.add_argument("--recall-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import faiss

    base = np.load(args.base).astype("float32")
    queries = np.load(args.queries).astype("float32")[: args.num_queries]

    idx = faiss.IndexFlatL2(args.dim)
    idx.add(base)
    _, gt = idx.search(queries, args.max_gt_depth)
    n = len(base)

    for group_size in args.group_sizes:
        k_clusters = n // group_size
        km = faiss.Kmeans(args.dim, k_clusters, niter=args.kmeans_iters, seed=args.seed)
        km.train(base)
        _, assign = km.index.search(base, 1)
        assign = assign[:, 0]
        _, order = km.index.search(queries, args.num_probe_candidates)

        for k in args.k_values:
            span = np.mean([len(set(assign[g[:k]])) for g in gt])
            need = []
            for i in range(len(queries)):
                counts = np.bincount(assign[gt[i, :k]], minlength=k_clusters)
                cumulative = np.cumsum(counts[order[i]])
                target = int(np.ceil(args.recall_fraction * k))
                j = np.searchsorted(cumulative, target)
                need.append(j + 1 if j < len(cumulative) else args.num_probe_candidates + 7951)
            need_arr = np.array(need)
            print(
                f"gs={group_size} k={k} span={span:.1f} span/k={span / k:.3f} "
                f"probes{int(args.recall_fraction * 100)}: mean={need_arr.mean():.1f} "
                f"med={np.median(need_arr):.0f} p90={np.percentile(need_arr, 90):.0f} "
                f"probes/k={need_arr.mean() / k:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
