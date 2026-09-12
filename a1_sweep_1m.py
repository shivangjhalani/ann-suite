"""
Full A1 sweep once R=64/96 sift1m indices land: local vs global navigability
failure across R in {32,64,96} at 1M scale, plus a beam-width sweep on R=32
to quantify how much beam search recovers vs. pure greedy (mirrors the
HNSW ef sweep for direct algorithm comparison).
"""
from __future__ import annotations
import sys
import numpy as np
import navigability as nv


def beam_global_failure(vecs, adj, medoid, rng, n_samples, beam_width, max_hops=200):
    n = len(vecs)
    fails = 0
    for _ in range(n_samples):
        t = int(rng.integers(0, n))
        visited = {medoid}
        frontier = [medoid]
        best = medoid
        best_d = np.linalg.norm(vecs[medoid] - vecs[t])
        reached = best_d == 0
        for _ in range(max_hops):
            if reached or not frontier:
                break
            cand = set()
            for f in frontier:
                cand.update(adj[f])
            cand -= visited
            if not cand:
                break
            cand = list(cand)
            visited.update(cand)
            d = np.linalg.norm(vecs[np.array(cand)] - vecs[t], axis=1)
            order = np.argsort(d)[:beam_width]
            frontier = [cand[i] for i in order]
            if d[order[0]] < best_d:
                best_d = d[order[0]]
                best = frontier[0]
                if best == t or best_d == 0:
                    reached = True
        if not reached:
            fails += 1
    return fails / n_samples


def main():
    base = np.load("/home/gem/shivang/ann-suite/data/sift1m/base.npy").astype(np.float32)
    rng = np.random.default_rng(0)

    paths = {
        32: "/home/gem/shivang/ann-suite/indices/DiskANN/sift1m-prebuilt/L-50_R-32_build-memory-maximum-0-27_index-prefix-c9390d63/ann_disk.index",
    }
    import glob
    for R in (64, 96):
        hits = glob.glob(f"/home/gem/shivang/ann-suite/indices/DiskANN/sift1m/L-100_R-{R}_*/ann_disk.index")
        if hits:
            paths[R] = hits[0]
        else:
            print(f"WARNING: no built index found for R={R}", file=sys.stderr)

    print(f"{'R':>4} {'deg':>7} {'eps_local':>10} {'eps_global(beam1)':>18} {'1/R':>8}")
    for R, path in sorted(paths.items()):
        adj, medoid, npts = nv.parse_disk_index_graph_only(path)
        deg = nv.mean_out_degree(adj)
        eps_l = nv.local_failure(base, adj, rng, 20000)
        eps_g = nv.global_failure(base, adj, medoid, rng, 2000, max_hops=200)
        print(f"{R:>4} {deg:>7.2f} {eps_l:>10.4f} {eps_g:>18.4f} {1/deg:>8.4f}")

    print("\nBeam-width sweep on R=32 (does small beam width recover navigability, like HNSW ef?):")
    adj, medoid, npts = nv.parse_disk_index_graph_only(paths[32])
    for bw in (1, 2, 4, 8):
        eps_g_bw = beam_global_failure(base, adj, medoid, rng, 1000, beam_width=bw)
        print(f"  beam_width={bw:>2}: eps_global={eps_g_bw:.4f}")


if __name__ == "__main__":
    main()
