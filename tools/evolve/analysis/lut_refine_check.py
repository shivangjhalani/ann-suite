"""(1) LUT refinement: est = d1 - 2<q,e2> + kappa', kappa' = kappa + 2<c,e2> stored as
scaled float16: same ranking as the deployed est? (2) CPU of faiss fast-scan
implementations and of the HNSW probe at large k. python lut_check.py INDEX DATA"""

import json
import sys
import time
from pathlib import Path

import faiss
import numba as nb
import numpy as np

idx, data = Path(sys.argv[1]), Path(sys.argv[2])
faiss.omp_set_num_threads(1)
L = lambda n: np.load(idx / "mem" / f"{n}.npy")  # noqa: E731
codes2, pq2, cent = L("codes2"), L("pq2").astype(np.float32), L("centroids").astype(np.float32)
ks = float(json.loads((idx / "mem/kscale.json").read_text()))
kappa = L("kappa").astype(np.float32) * ks
cs = L("cstart").astype(np.int64)
N = int(cs[-1])
M2, K2, ds = pq2.shape
lst_all = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
# kappa' = kappa + 2 <c, e2>, in chunks
kp = np.empty(N, np.float32)
for s in range(0, N, 1_000_000):
    e = pq2[np.arange(M2), codes2[s : s + 1_000_000]].reshape(-1, M2 * ds)
    kp[s : s + 1_000_000] = kappa[s : s + 1_000_000] + 2.0 * (
        cent[lst_all[s : s + 1_000_000]] * e
    ).sum(1)
sc = max(1.0, float(np.abs(kp).max()) / 60000.0)
kp16 = (kp / sc).astype(np.float16).view(np.uint16)
print(
    "kappa range", float(np.abs(kappa).max()), "kappa' range", float(np.abs(kp).max()), "scale", sc
)


@nb.njit(fastmath=True, nogil=True)
def refine_lut(d1, pos, q, pq2, codes2, kp, ks):
    M, K, ds = pq2.shape
    lut = np.empty((M, K), np.float32)
    for m in range(M):
        for j in range(K):
            a = np.float32(0.0)
            for t in range(ds):
                a += q[m * ds + t] * pq2[m, j, t]
            lut[m, j] = a
    est = np.empty(pos.size, np.float32)
    for i in range(pos.size):
        r = pos[i]
        a = np.float32(0.0)
        for m in range(M):
            a += lut[m, codes2[r, m]]
        h = kp[r]
        e = (h >> 10) & 0x1F
        mm = np.float32(h & 0x3FF)
        v = (
            mm * np.float32(2.0**-24)
            if e == 0
            else (np.float32(1.0) + mm / np.float32(1024.0)) * np.float32(2.0) ** (np.int32(e) - 15)
        )
        v = -v if h >> 15 else v
        est[i] = d1[i] - 2.0 * a + v * ks
    return est


fs = faiss.read_index(str(idx / "mem/fs.faiss"))
cq = faiss.downcast_index(fs.quantizer)
Q = np.load(data / "queries.npy")[:300].astype(np.float32)
G = np.load(data / "ground_truth.npy")[:300, :10]
for npb, sl in ((192, 2048), (256, 3072)):
    fs.nprobe = npb
    cq.hnsw.efSearch = max(64, int(1.5 * npb))
    agree = 0.0
    t_old = t_new = 0.0
    refine_lut(
        np.zeros(1, np.float32), np.zeros(1, np.int64), Q[0], pq2, codes2, kp16, np.float32(sc)
    )
    for q in Q:
        d1, h = fs.search(q[None], sl)
        pos, d1 = h[0][h[0] >= 0], d1[0][h[0] >= 0]
        lst = np.searchsorted(cs, pos, side="right") - 1
        t0 = time.thread_time()
        e2 = pq2[np.arange(M2), codes2[pos]].reshape(pos.size, -1)
        old = d1 - 2.0 * ((q - cent[lst]) * e2).sum(1) + kappa[pos]
        t1 = time.thread_time()
        new = refine_lut(d1, pos, q, pq2, codes2, kp16, np.float32(sc))
        t2 = time.thread_time()
        t_new += t2 - t1
        a, b = np.argsort(old)[:320], np.argsort(new)[:320]
        agree += len(np.intersect1d(pos[a], pos[b])) / 320
    print(
        f"np{npb} sl{sl}: top-320 overlap old vs LUT {agree / len(Q):.4f}, LUT refine {t_new / len(Q) * 1e3:.3f} ms"
    )
    # fast-scan implementations
    fs.implem = 0
    _, h0 = fs.search(Q[:20], sl)
    for impl in (0, 10, 12, 13, 14, 15):
        try:
            fs.implem = impl
            t0 = time.thread_time()
            for q in Q[:200]:
                fs.search(q[None], sl)
            t1 = time.thread_time()
            _, h = fs.search(Q[:20], sl)
            same = np.mean([len(np.intersect1d(h[i], h0[i])) / sl for i in range(20)])
            print(
                f"  implem {impl}: {(t1 - t0) / 200 * 1e3:.3f} ms/query (incl. probe), same ids {same:.4f}"
            )
        except Exception as ex:  # noqa: BLE001
            print(f"  implem {impl}: {ex}")
    fs.implem = 0
    for ef in (npb, int(1.2 * npb), int(1.5 * npb)):
        cq.hnsw.efSearch = ef
        t0 = time.thread_time()
        cov = 0.0
        for q in Q[:200]:
            cq.search(q[None], npb)
        t1 = time.thread_time()
        print(f"  probe ef {ef}: {(t1 - t0) / 200 * 1e3:.3f} ms")
