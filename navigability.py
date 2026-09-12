"""
Direction A1 (from flash-native-ann.html): measure the navigability gap.

Parses DiskANN's disk-index binary layout directly (format reverse-engineered
from diskann_code/src/disk_utils.cpp::create_disk_layout, confirmed byte-exact
against the R=32/64/96 sift-10k indices already built on this box) and computes:

  (a) LOCAL failure fraction epsilon_local: for sampled (p, t) pairs, does p have
      an out-edge strictly closer to t than p is? (the theorem's own quantity)
  (b) GLOBAL failure fraction epsilon_global: does pure greedy descent from a
      random start actually reach t (or t's neighborhood)?

then compares both against the Theta(1/R) frontier predicted by
arXiv:2607.14564 / arXiv:2609.02498.
"""
from __future__ import annotations

import struct
import sys
import numpy as np


def parse_disk_index_graph_only(path: str):
    """Parse only the adjacency lists + metadata, ignoring on-disk coords
    (which may be PQ-compressed proxies, not full-precision vectors)."""
    with open(path, "rb") as f:
        raw = f.read()

    npts_meta, ndims_meta = struct.unpack_from("<ii", raw, 0)
    assert ndims_meta == 1
    meta = struct.unpack_from(f"<{npts_meta}Q", raw, 8)
    npts, ndims, medoid, max_node_len, nnodes_per_sector = meta[0:5]
    print(f"  npts={npts} ndims={ndims} medoid={medoid} max_node_len={max_node_len} "
          f"nnodes_per_sector={nnodes_per_sector}", file=sys.stderr)

    SECTOR_LEN = 4096
    elem_size = None
    for cand in (4, 1, 2):
        rem = max_node_len - ndims * cand
        if rem > 0 and rem % 4 == 0 and 0 < (rem // 4 - 1) <= 512:
            elem_size = cand
            break
    assert elem_size is not None
    coord_bytes = ndims * elem_size

    adj: list[list[int]] = [[] for _ in range(npts)]
    n_sectors = (npts + nnodes_per_sector - 1) // nnodes_per_sector
    cur_id = 0
    for sector in range(n_sectors):
        sec_off = SECTOR_LEN * (sector + 1)
        for slot in range(nnodes_per_sector):
            if cur_id >= npts:
                break
            node_off = sec_off + slot * max_node_len
            nnbrs = struct.unpack_from("<I", raw, node_off + coord_bytes)[0]
            nbrs = struct.unpack_from(f"<{nnbrs}I", raw, node_off + coord_bytes + 4)
            adj[cur_id] = list(nbrs)
            cur_id += 1
    assert cur_id == npts
    return adj, medoid, npts


def parse_disk_index(path: str):
    with open(path, "rb") as f:
        raw = f.read()

    npts_meta, ndims_meta = struct.unpack_from("<ii", raw, 0)
    assert ndims_meta == 1
    meta = struct.unpack_from(f"<{npts_meta}Q", raw, 8)
    npts, ndims, medoid, max_node_len, nnodes_per_sector = meta[0:5]
    vamana_frozen_num, vamana_frozen_loc, append_reorder = meta[5:8]
    print(f"  npts={npts} ndims={ndims} medoid={medoid} max_node_len={max_node_len} "
          f"nnodes_per_sector={nnodes_per_sector}", file=sys.stderr)

    SECTOR_LEN = 4096
    # detect element size (float32 coords, or uint8 PQ-compressed disk_pq mode):
    # max_node_len = (width+1)*4 + ndims*elem_size, width in a sane range.
    elem_size = None
    for cand in (4, 1, 2):
        rem = max_node_len - ndims * cand
        if rem > 0 and rem % 4 == 0 and 0 < (rem // 4 - 1) <= 512:
            elem_size = cand
            break
    assert elem_size is not None, "could not infer coordinate element size"
    dtype = {4: "<f4", 1: "<u1", 2: "<u2"}[elem_size]

    vecs = np.zeros((npts, ndims), dtype=np.float32)
    adj: list[list[int]] = [[] for _ in range(npts)]

    n_sectors = (npts + nnodes_per_sector - 1) // nnodes_per_sector
    coord_bytes = ndims * elem_size

    cur_id = 0
    for sector in range(n_sectors):
        sec_off = SECTOR_LEN * (sector + 1)
        for slot in range(nnodes_per_sector):
            if cur_id >= npts:
                break
            node_off = sec_off + slot * max_node_len
            coords = np.frombuffer(raw, dtype=dtype, count=ndims, offset=node_off).astype(np.float32)
            vecs[cur_id] = coords
            nnbrs = struct.unpack_from("<I", raw, node_off + coord_bytes)[0]
            nbrs = struct.unpack_from(f"<{nnbrs}I", raw, node_off + coord_bytes + 4)
            adj[cur_id] = list(nbrs)
            cur_id += 1

    assert cur_id == npts
    return vecs, adj, medoid


def local_failure(vecs: np.ndarray, adj: list[list[int]], rng: np.random.Generator, n_samples: int) -> float:
    n = len(vecs)
    fails = 0
    trials = 0
    for _ in range(n_samples):
        p = int(rng.integers(0, n))
        t = int(rng.integers(0, n))
        if p == t or not adj[p]:
            continue
        trials += 1
        d_pt = np.linalg.norm(vecs[p] - vecs[t])
        nbrs = np.array(adj[p])
        d_nt = np.linalg.norm(vecs[nbrs] - vecs[t], axis=1)
        if not np.any(d_nt < d_pt):
            fails += 1
    return fails / trials if trials else float("nan")


def global_failure(vecs: np.ndarray, adj: list[list[int]], medoid: int, rng: np.random.Generator,
                    n_samples: int, max_hops: int = 100) -> float:
    n = len(vecs)
    fails = 0
    for _ in range(n_samples):
        t = int(rng.integers(0, n))
        cur = medoid
        d_cur = np.linalg.norm(vecs[cur] - vecs[t])
        reached = d_cur == 0
        for _ in range(max_hops):
            if reached:
                break
            nbrs = adj[cur]
            if not nbrs:
                break
            nbrs_arr = np.array(nbrs)
            dists = np.linalg.norm(vecs[nbrs_arr] - vecs[t], axis=1)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] >= d_cur:
                break  # stuck: no improving neighbor, greedy halts
            cur = nbrs_arr[best_idx]
            d_cur = dists[best_idx]
            if cur == t or d_cur == 0:
                reached = True
        if not reached:
            fails += 1
    return fails / n_samples


def mean_out_degree(adj: list[list[int]]) -> float:
    return float(np.mean([len(a) for a in adj]))


def main():
    paths = {
        32: "/home/gem/shivang/ann-suite/indices/DiskANN/sift-10k/L-100_R-32_alpha-1-2_build-memory-maximum-2-0_nu-8a557fc3/ann_disk.index",
        64: "/home/gem/shivang/ann-suite/indices/DiskANN/sift-10k/L-100_R-64_alpha-1-2_build-memory-maximum-2-0_nu-6368c164/ann_disk.index",
        96: "/home/gem/shivang/ann-suite/indices/DiskANN/sift-10k/L-100_R-96_alpha-1-2_build-memory-maximum-2-0_nu-551e7b7b/ann_disk.index",
    }
    rng = np.random.default_rng(0)
    n_local = 20000
    n_global = 2000

    print(f"{'R_nominal':>10} {'deg_mean':>9} {'eps_local':>10} {'eps_global':>11} {'1/R_pred':>9}")
    results = []
    for R, path in paths.items():
        print(f"Parsing R={R} ...", file=sys.stderr)
        vecs, adj, medoid = parse_disk_index(path)
        deg = mean_out_degree(adj)
        eps_l = local_failure(vecs, adj, rng, n_local)
        eps_g = global_failure(vecs, adj, medoid, rng, n_global)
        pred = 1.0 / deg
        results.append((R, deg, eps_l, eps_g, pred))
        print(f"{R:>10} {deg:>9.2f} {eps_l:>10.4f} {eps_g:>11.4f} {pred:>9.4f}")

    print("\nGap (eps_global / eps_local), i.e. how much worse end-to-end greedy is than the local edge condition:")
    for R, deg, eps_l, eps_g, pred in results:
        ratio = eps_g / eps_l if eps_l > 0 else float("inf")
        print(f"  R={R}: eps_local={eps_l:.4f} eps_global={eps_g:.4f} ratio={ratio:.2f}x  (1/R predicts {pred:.4f})")


if __name__ == "__main__":
    main()
