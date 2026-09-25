"""Co-location / block-locality analysis: if the base set is clustered into
groups of a given size (a proxy for "blocks" written contiguously to disk),
how many groups (blocks) does a query's true top-k span, and how many
IVF-style probes (in centroid-distance order) are needed to recover 90% of
its true top-k?

Ported from ``~/research/py/colocate.py`` on the isfcr benchmark host. This
was an exploratory study of whether re-laying-out the base vectors by
k-means cluster (so a beam search's fan-out tends to land in fewer disk
blocks) would meaningfully cut PipeANN's I/Os-per-query.

Usage:

    uv run python tools/research/colocate.py \\
        --base data/sift1m/base.npy --queries data/sift1m/queries.npy \\
        --ground-truth data/sift1m/ground_truth.npy --dim 128 \\
        --group-sizes 8 32 128
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Base vectors (.npy)")
    parser.add_argument("--queries", type=Path, required=True, help="Query vectors (.npy)")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Ground-truth neighbor ids (.npy)")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--gt-depth", type=int, default=10, help="Depth of true top-k used (default 10)")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--max-points-per-centroid", type=int, default=64)
    parser.add_argument("--num-probe-candidates", type=int, default=512)
    parser.add_argument("--recall-target", type=int, default=9, help="neighbors-found threshold out of gt-depth")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import faiss

    base = np.load(args.base).astype("float32")
    queries = np.load(args.queries).astype("float32")
    gt = np.load(args.ground_truth)[:, : args.gt_depth]
    n = len(base)
    print(base.shape, queries.shape, gt.shape, flush=True)

    for group_size in args.group_sizes:
        k = n // group_size
        t0 = time.time()
        km = faiss.Kmeans(
            args.dim,
            k,
            niter=args.kmeans_iters,
            seed=args.seed,
            max_points_per_centroid=args.max_points_per_centroid,
        )
        km.train(base)
        _, assign = km.index.search(base, 1)
        assign = assign[:, 0]
        sizes = np.bincount(assign, minlength=k)

        # Distinct groups spanned by a query's true top-k.
        span = np.mean([len(set(assign[g])) for g in gt])

        # IVF-style view: probe clusters in centroid order, count how many
        # probes are needed until >= recall_target of the true top-k are found.
        _, order = km.index.search(queries, args.num_probe_candidates)
        need = []
        for i in range(len(queries)):
            counts: dict[int, int] = {}
            for c in assign[gt[i]]:
                counts[c] = counts.get(c, 0) + 1
            found = 0
            n_probes = None
            for j, c in enumerate(order[i]):
                found += counts.get(c, 0)
                if found >= args.recall_target:
                    n_probes = j + 1
                    break
            need.append(n_probes if n_probes else args.num_probe_candidates + 487)
        need_arr = np.array(need)
        frac_unfound = np.mean(need_arr == args.num_probe_candidates + 487)

        print(
            f"gs={group_size} K={k} size mean={sizes.mean():.1f} "
            f"p99={np.percentile(sizes, 99):.0f} max={sizes.max()} span{args.gt_depth}={span:.2f} "
            f"probes_for_r{args.recall_target / args.gt_depth:.1f}: "
            f"mean={need_arr.mean():.1f} median={np.median(need_arr):.0f} "
            f"p90={np.percentile(need_arr, 90):.0f} "
            f"frac_not_found={frac_unfound:.3f} t={time.time() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
