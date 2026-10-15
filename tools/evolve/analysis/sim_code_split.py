"""First-stage code variants on a built abl_fastscan_g40 index (same lists, same order):
shortlist coverage of the true top-10, recall at 24/32 pages after an 8-bit refinement
(m2 sub-quantizers of the remaining error, LUT form) and single-thread CPU of the
fast-scan search. Dimensions are zero-padded to a multiple of lcm(m1, m2).
python sim_fs2.py INDEX DATA L2|IP m1:m2:opq ..."""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

idx, data = Path(sys.argv[1]), Path(sys.argv[2])
IP = sys.argv[3] == "IP"
MT = faiss.METRIC_INNER_PRODUCT if IP else faiss.METRIC_L2
lay = json.loads((idx / "mem/layout.json").read_text())
dim, rec, per = lay["dim"], lay["rec"], lay["per_page"]
cs = np.load(idx / "mem/cstart.npy").astype(np.int64)
ps = np.load(idx / "mem/page_start.npy").astype(np.int64)
cent = np.load(idx / "mem/centroids.npy").astype(np.float32)
nlist, N = len(cs) - 1, int(cs[-1])
pages = np.memmap(idx / "disk/lists.pages", dtype=np.uint8, mode="r").reshape(-1, 4096)
vec = np.empty((N, dim), np.float32)
pid = np.empty(N, np.int64)
for c in range(nlist):
    r = np.asarray(pages[ps[c] : ps[c + 1], : per * rec]).reshape(-1, rec)[: cs[c + 1] - cs[c]]
    vec[cs[c] : cs[c + 1]] = r[:, : rec - 4].view(np.dtype(lay["dtype"])).astype(np.float32)
    pid[cs[c] : cs[c + 1]] = r[:, rec - 4 :].copy().view(np.uint32)[:, 0]
inv = np.empty(N, np.int64)
inv[pid] = np.arange(N)
lst_of = np.repeat(np.arange(nlist), np.diff(cs))
off_of = np.arange(N) - cs[lst_of]
Q = np.load(data / "queries.npy")[:1000].astype(np.float32)
GP = inv[np.load(data / "ground_truth.npy")[:1000, :10]]
rng = np.random.default_rng(0)
tr = np.sort(rng.choice(N, 400_000, replace=False))
print("loaded", flush=True)


def variant(name, m1, m2, opq):
    faiss.omp_set_num_threads(16)
    lcm = int(np.lcm(m1, m2))
    dp = int(-(-dim // lcm) * lcm)  # padded dimension
    R = np.eye(dp, dtype=np.float32)[:, :dim]  # y = x @ R.T, (dp, dim)
    if opq:
        o = faiss.OPQMatrix(dp, m1)
        o.niter = 20
        o.train(np.ascontiguousarray((vec[tr] - cent[lst_of[tr]]) @ R.T))
        R = faiss.vector_to_array(o.A).reshape(dp, dp).astype(np.float32) @ R
    rc = np.ascontiguousarray(cent @ R.T)
    quant = faiss.IndexHNSWFlat(dp, 32, MT)
    quant.hnsw.efConstruction = 40
    quant.add(rc)
    fs = faiss.IndexIVFPQFastScan(quant, dp, nlist, m1, 4, MT)
    fs.by_residual = True
    xt = np.ascontiguousarray(vec[tr] @ R.T)
    fs.train(xt)
    r1 = np.ascontiguousarray(xt - rc[lst_of[tr]])
    pq2 = faiss.ProductQuantizer(dp, m2, 8)
    pq2.train(np.ascontiguousarray(r1 - fs.pq.decode(fs.pq.compute_codes(r1))))
    codes2 = np.empty((N, m2), np.uint8)
    kap = np.empty(N, np.float32)
    moved = 0
    for s in range(0, N, 1_000_000):
        xs = np.ascontiguousarray(vec[s : s + 1_000_000] @ R.T)
        quant.hnsw.efSearch = 64
        _, a = quant.search(xs, 1)
        a = a[:, 0]
        fs.add_with_ids(xs, np.arange(s, s + xs.shape[0], dtype=np.int64))
        r1 = np.ascontiguousarray(xs - rc[a])
        x1 = fs.pq.decode(fs.pq.compute_codes(r1))
        c2 = pq2.compute_codes(np.ascontiguousarray(r1 - x1))
        e2 = pq2.decode(c2)
        codes2[s : s + xs.shape[0]] = c2
        kap[s : s + xs.shape[0]] = (
            0.0 if IP else 2 * (x1 * e2).sum(1) + (e2 * e2).sum(1) + 2 * (rc[a] * e2).sum(1)
        )
        moved += int((a != lst_of[s : s + xs.shape[0]]).sum())
    print(name, "vectors in a different faiss list than their page list:", moved, flush=True)
    ksc = max(1.0, float(np.abs(kap).max()) / 60000.0)
    k16 = (kap / ksc).astype(np.float16).astype(np.float32) * ksc
    print(
        name,
        f"kappa' |max| {np.abs(kap).max():.4g}  float16 abs err max {np.abs(k16 - kap).max():.4g} mean {np.abs(k16 - kap).mean():.4g}",
        flush=True,
    )
    kap = k16
    C2 = faiss.vector_to_array(pq2.centroids).reshape(m2, 256, -1)
    QR = np.ascontiguousarray(Q @ R.T)
    faiss.omp_set_num_threads(1)
    for npb, sl in (
        (64, 512),
        (96, 768),
        (128, 1024),
        (192, 1536),
        (256, 2048),
        (256, 3072),
        (384, 4096),
    ):
        fs.nprobe = npb
        quant.hnsw.efSearch = npb
        t0 = time.thread_time()
        D1, H = fs.search(QR[:300], sl)
        cpu = (time.thread_time() - t0) / 300 * 1e3
        faiss.omp_set_num_threads(16)
        D1, H = fs.search(QR, sl)
        faiss.omp_set_num_threads(1)
        cov = 0.0
        r24 = r32 = 0.0
        for i in range(len(Q)):
            ok = H[i] >= 0
            pos, d1 = H[i][ok], D1[i][ok]
            g = GP[i]
            cov += np.isin(g, pos).mean()
            lut = np.einsum("mjt,mt->mj", C2, QR[i].reshape(m2, -1))
            q2 = lut[np.arange(m2), codes2[pos]].sum(1)
            est = -(d1 + q2) if IP else d1 - 2 * q2 + kap[pos]
            pg = lst_of[pos] * 100000 + off_of[pos] // per
            u, first = np.unique(pg[np.argsort(est)], return_index=True)
            ranked = u[np.argsort(first)]
            gpg = lst_of[g] * 100000 + off_of[g] // per
            r24 += np.isin(gpg, ranked[:24]).mean()
            r32 += np.isin(gpg, ranked[:32]).mean()
        n = len(Q)
        print(
            f"{name:26} np{npb:3d} sl{sl:4d}: cov {cov / n:.4f}  rec@24 {r24 / n:.4f}  rec@32 {r32 / n:.4f}  fastscan cpu {cpu:.3f} ms",
            flush=True,
        )
    del fs, codes2, kap


for spec in sys.argv[4:]:
    m1, m2, opq = (int(v) for v in spec.split(":"))
    variant(f"m{m1}/{m2}{' opq' if opq else ''} ({m1 // 2}+{m2}B)", m1, m2, bool(opq))
