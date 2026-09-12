"""
Cross-algorithm extension of A1: measure epsilon_global on the HNSW graph too.

hnswlib doesn't expose the internal adjacency list through its stock Python
bindings, so we can't compute epsilon_local (edge-existence) the way we did
for DiskANN's disk layout. But `knn_query` with ef=1 IS single-path greedy
search (multi-layer entry-point descent, then a width-1 walk at layer 0),
so it directly gives us epsilon_global under the same protocol: sample a
random point t already in the index, search for it with its own vector as
the query, ef=1, k=1 -- "reached" iff the returned id is t itself.
"""
from __future__ import annotations
import sys
import numpy as np
import hnswlib


def main():
    base = np.load("/home/gem/shivang/ann-suite/data/sift-10k/base.npy").astype(np.float32)
    n, dim = base.shape
    print(f"base: n={n} dim={dim}", file=sys.stderr)

    index_path = "/home/gem/shivang/ann-suite/indices/HNSW/sift-10k/M-16_ef-construction-200_num-threads-4-4b41b197/index.bin"
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.load_index(index_path, max_elements=n)

    rng = np.random.default_rng(0)
    n_samples = 2000
    sample_ids = rng.choice(n, size=n_samples, replace=False)

    for ef in (1, 4, 10):
        idx.set_ef(ef)
        fails = 0
        for t in sample_ids:
            labels, _ = idx.knn_query(base[t], k=1)
            if labels[0][0] != t:
                fails += 1
        eps_g = fails / n_samples
        print(f"HNSW M=16 sift-10k: ef={ef:>3} eps_global={eps_g:.4f}")


if __name__ == "__main__":
    main()
