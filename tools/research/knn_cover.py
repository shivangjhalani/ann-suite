"""kNN-coverage analysis: how well do a handful of anchor points' own nearest
neighbors cover a query's true top-k?

Ported from the ANN-on-SSD research project's exploratory scripts
(``~/research/py/knn_cover.py`` on the isfcr benchmark host). Answers: if you
seed a graph/beam search from a query's rank-1 (or rank-2, rank-5) true
neighbor and expand its own k-NN list, what fraction of the query's true
top-10 do you recover? This bears on how much of PipeANN's beam-search
"co-location" benefit comes from true neighbors already being graph-adjacent.

Usage (BIGANN-10M, matching the original exploratory run):

    uv run python tools/research/knn_cover.py \\
        --base data/bigann-10m/base10m.u8bin --base-format bigann_bin --base-dtype uint8 \\
        --queries data/bigann-10m/query.u8bin --query-format bigann_bin --query-dtype uint8 \\
        --ground-truth data/bigann-10m/gt10m.bin --gt-format bigann_bin \\
        --num-queries 2000 --dim 128

Or against any ann-suite dataset registered as .npy:

    uv run python tools/research/knn_cover.py \\
        --base data/sift1m/base.npy --queries data/sift1m/queries.npy \\
        --ground-truth data/sift1m/ground_truth.npy --dim 128

Requires: faiss-cpu, numpy (see tools/research/README.md for the uv group).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def _read_bigann_bin(path: Path, dtype: np.dtype, count: int | None = None) -> np.ndarray:
    """Read a big-ann-benchmarks style ``<u4 n><u4 dim><data...>`` binary file."""
    with path.open("rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        if count is not None:
            n = min(n, count)
        data = np.fromfile(f, dtype=dtype, count=n * d)
    return data.reshape(n, d)


def load_vectors(path: Path, fmt: str, dtype: str, count: int | None = None) -> np.ndarray:
    if fmt == "npy":
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr[:count] if count else arr)
    if fmt == "bigann_bin":
        return _read_bigann_bin(path, np.dtype(dtype), count)
    raise ValueError(f"Unknown format: {fmt}")


def load_ground_truth(path: Path, fmt: str, count: int | None = None) -> np.ndarray:
    if fmt == "npy":
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr[:count] if count else arr)
    if fmt == "bigann_bin":
        # PipeANN/DiskANN "ids only" truthset: <u4 n><u4 k><u32 ids...>
        with path.open("rb") as f:
            n, k = struct.unpack("<II", f.read(8))
            if count is not None:
                n = min(n, count)
            ids = np.fromfile(f, dtype=np.uint32, count=n * k)
        return ids.reshape(n, k)
    raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-format", choices=["npy", "bigann_bin"], default="npy")
    parser.add_argument("--base-dtype", default="uint8", help="numpy dtype for bigann_bin base file")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-format", choices=["npy", "bigann_bin"], default="npy")
    parser.add_argument("--query-dtype", default="uint8", help="numpy dtype for bigann_bin query file")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--gt-format", choices=["npy", "bigann_bin"], default="npy")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=2000)
    parser.add_argument(
        "--anchor-ranks", type=int, nargs="+", default=[0, 1, 4],
        help="0-indexed ground-truth ranks to use as expansion anchors (default: rank1,2,5)",
    )
    parser.add_argument("--anchor-k", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--gt-depth", type=int, default=10, help="Depth of 'true top-k' being covered")
    parser.add_argument("--add-batch-size", type=int, default=1_000_000)
    args = parser.parse_args()

    print(f"Loading base vectors from {args.base} ({args.base_format})", file=sys.stderr)
    base = load_vectors(args.base, args.base_format, args.base_dtype)
    queries = load_vectors(args.queries, args.query_format, args.query_dtype, args.num_queries)
    gt = load_ground_truth(args.ground_truth, args.gt_format, args.num_queries)
    print(f"base={base.shape} queries={queries.shape} gt={gt.shape}", flush=True)

    import faiss

    idx = faiss.IndexFlatL2(args.dim)
    for start in range(0, len(base), args.add_batch_size):
        idx.add(base[start : start + args.add_batch_size].astype("float32"))

    top = gt[:, : args.gt_depth]
    max_anchor_k = max(args.anchor_k)
    knn: dict[int, np.ndarray] = {}
    for rank in args.anchor_ranks:
        anchors = gt[:, rank]
        _, neighbor_ids = idx.search(base[anchors].astype("float32"), max_anchor_k + 1)
        knn[rank] = neighbor_ids

    for rank in args.anchor_ranks:
        for k in args.anchor_k:
            cov = np.mean(
                [
                    len(set(top[i]) & set(knn[rank][i, : k + 1])) / args.gt_depth
                    for i in range(len(queries))
                ]
            )
            print(
                f"anchor=rank{rank + 1} k={k} "
                f"coverage_of_true_top{args.gt_depth}={cov:.3f}",
                flush=True,
            )

    if len(args.anchor_ranks) >= 2:
        r0, r1 = args.anchor_ranks[0], args.anchor_ranks[1]
        for k in args.anchor_k:
            if k > max(args.anchor_k[:2], default=k):
                continue
            cov = np.mean(
                [
                    len(set(top[i]) & (set(knn[r0][i, : k + 1]) | set(knn[r1][i, : k + 1])))
                    / args.gt_depth
                    for i in range(len(queries))
                ]
            )
            print(f"anchor=rank{r0 + 1}+rank{r1 + 1} k={k} coverage={cov:.3f}", flush=True)


if __name__ == "__main__":
    main()
