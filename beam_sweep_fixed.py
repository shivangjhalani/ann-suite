"""
Fixed beam-width recovery sweep for DiskANN (companion to the HNSW ef-sweep).

Bugs in the previous version, both fixed here:
  1. Success was checked via `distance == 0` instead of an exact target-id
     match, so landing on a near-duplicate vector (SIFT-1M has documented
     duplicates) was wrongly counted as "reached".
  2. There was no real termination condition, so wide exploration within the
     hop cap could wander into a duplicate almost every time, regardless of
     beam_width -- explaining the flat 0% across all widths.

This version implements real best-first beam search matching DiskANN's own
search-list algorithm: maintain a global candidate pool of (node, distance-
to-target) pairs; each round, expand the `beam_width` best *unvisited*
candidates (add their neighbors to the pool); stop when the target id is
visited, or the pool of unvisited candidates is exhausted.
"""
from __future__ import annotations
import sys
import time
import numpy as np
import navigability as nv


def beam_global_failure(vecs: np.ndarray, adj: list[list[int]], medoid: int,
                         rng: np.random.Generator, n_samples: int, beam_width: int,
                         max_rounds: int = 400) -> tuple[float, float]:
    n = len(vecs)
    fails = 0
    total_expansions = 0
    for _ in range(n_samples):
        t = int(rng.integers(0, n))
        d0 = float(np.linalg.norm(vecs[medoid] - vecs[t]))
        dist_known: dict[int, float] = {medoid: d0}
        visited: set[int] = set()
        pending: set[int] = {medoid}
        found = medoid == t
        rounds = 0
        while pending and not found and rounds < max_rounds:
            rounds += 1
            to_expand = sorted(pending, key=lambda i: dist_known[i])[:beam_width]
            for node in to_expand:
                pending.discard(node)
                visited.add(node)
                total_expansions += 1
                if node == t:
                    found = True
                    break
                nbrs = [nb for nb in adj[node] if nb not in dist_known]
                if nbrs:
                    nbrs_arr = np.array(nbrs)
                    d = np.linalg.norm(vecs[nbrs_arr] - vecs[t], axis=1)
                    for nb, dd in zip(nbrs, d):
                        dist_known[nb] = float(dd)
                        pending.add(nb)
            if found:
                break
        if not found:
            fails += 1
    return fails / n_samples, total_expansions / n_samples


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sift-10k"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    if dataset == "sift-10k":
        base = np.load("/home/gem/shivang/ann-suite/data/sift-10k/base.npy").astype(np.float32)
        path = "/home/gem/shivang/ann-suite/indices/DiskANN/sift-10k/L-100_R-32_alpha-1-2_build-memory-maximum-2-0_nu-8a557fc3/ann_disk.index"
        label = "DiskANN R=32, SIFT-10K"
    else:
        base = np.load("/home/gem/shivang/ann-suite/data/sift1m/base.npy").astype(np.float32)
        path = "/home/gem/shivang/ann-suite/indices/DiskANN/sift1m-prebuilt/L-50_R-32_build-memory-maximum-0-27_index-prefix-c9390d63/ann_disk.index"
        label = "DiskANN R=32, SIFT-1M"

    adj, medoid, npts = nv.parse_disk_index_graph_only(path)
    rng = np.random.default_rng(0)

    print(f"{label} -- unbounded beam search (sanity check)", file=sys.stderr)
    for bw in (1, 2, 4, 8):
        eps_g, mean_exp = beam_global_failure(base, adj, medoid, rng, n_samples, beam_width=bw, max_rounds=400)
        print(f"  beam_width={bw}: eps_global={eps_g:.4f} mean_expansions={mean_exp:.1f}", file=sys.stderr)

    print(f"\n{label} -- FIXED ROUND-TRIP BUDGET sweep (the real question)", file=sys.stderr)
    print("round_budget,beam_width,eps_global,mean_expansions")
    for round_budget in (2, 4, 6, 8, 10):
        rng2 = np.random.default_rng(0)  # same target sequence across beam widths, per budget
        for bw in (1, 2, 4, 8):
            t0 = time.time()
            eps_g, mean_exp = beam_global_failure(base, adj, medoid, rng2, n_samples, beam_width=bw,
                                                    max_rounds=round_budget)
            print(f"round_budget={round_budget:>2} beam_width={bw:>2}  eps_global={eps_g:.4f}  "
                  f"mean_expansions={mean_exp:.1f}  ({time.time()-t0:.1f}s)", file=sys.stderr)
            print(f"{round_budget},{bw},{eps_g:.4f},{mean_exp:.1f}")


if __name__ == "__main__":
    main()
