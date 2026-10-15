"""Where is recall lost at high recall? Diagnostics on a built abl_fastscan_g40 index.

For each (nprobe, shortlist): probe coverage (true top-10 in probed lists), shortlist
coverage, and recall at page budgets B when pages are picked in order of
  d1    fast-scan estimate only,
  est   fast-scan + 8-bit refinement (the deployed design),
  exact exact distances of the shortlist (oracle refinement),
and with the same locality order packed at pp vectors per page (pp=10 is float32
on DEEP; 20 ~ float16 records, 40 ~ int8 records). Recall counts a true neighbour
as found when its page is read (pages are scored exactly).
  python diag_hr.py INDEX_DIR DATA_DIR [nq]
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

idx, data = Path(sys.argv[1]), Path(sys.argv[2])
nq = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
faiss.omp_set_num_threads(16)
lay = json.loads((idx / "mem/layout.json").read_text())
dim, rec, per_page = lay["dim"], lay["rec"], lay["per_page"]
vb = rec - 4
cstart = np.load(idx / "mem/cstart.npy").astype(np.int64)
page_start = np.load(idx / "mem/page_start.npy").astype(np.int64)
nlist = len(cstart) - 1
N = int(cstart[-1])
pages = np.memmap(idx / "disk/lists.pages", dtype=np.uint8, mode="r").reshape(-1, 4096)
t0 = time.time()
vec = np.empty((N, dim), np.float32)
pid = np.empty(N, np.int64)
for c in range(nlist):
    p0, p1 = page_start[c], page_start[c + 1]
    r = np.asarray(pages[p0:p1, : per_page * rec]).reshape(-1, rec)[: cstart[c + 1] - cstart[c]]
    vec[cstart[c] : cstart[c + 1]] = r[:, :vb].view(np.dtype(lay["dtype"])).astype(np.float32)
    pid[cstart[c] : cstart[c + 1]] = r[:, vb:].copy().view(np.uint32)[:, 0]
inv = np.empty(N, np.int64)
inv[pid] = np.arange(N)
print(f"loaded {N} vectors in {time.time() - t0:.0f}s", flush=True)
lst_of = np.repeat(np.arange(nlist), np.diff(cstart))
off_of = np.arange(N) - cstart[lst_of]

fs = faiss.read_index(str(idx / "mem/fs.faiss"))
quant = faiss.downcast_index(fs.quantizer)
codes2 = np.load(idx / "mem/codes2.npy")
pq2 = np.load(idx / "mem/pq2.npy").astype(np.float32)
M2, _, ds2 = pq2.shape
kappa = np.load(idx / "mem/kappa.npy").astype(np.float32) * float(
    json.loads((idx / "mem/kscale.json").read_text())
)
cent = np.load(idx / "mem/centroids.npy").astype(np.float32)
Q = np.load(data / "queries.npy")[:nq].astype(np.float32)
G = np.load(data / "ground_truth.npy")[:nq, :10]
GP = inv[G]
PPS = (per_page, 2 * per_page, 4 * per_page)
POINTS = [(128, 1024), (192, 1536), (192, 2048), (256, 2048), (256, 3072), (384, 4096)]
BUD = (24, 32, 48, 64, 80, 96, 128)


def first_pages(pg, order, b):
    u, first = np.unique(pg[order], return_index=True)
    return set(u[np.argsort(first)][:b].tolist())


for nprobe, sl in POINTS:
    fs.nprobe = nprobe
    quant.hnsw.efSearch = max(64, int(1.5 * nprobe))
    _, probed = quant.search(Q, nprobe)
    D1, H = fs.search(Q, sl)
    cov_p = cov_s = 0.0
    lb = dict.fromkeys(PPS, 0.0)
    rec_ = {(o, pp, b): 0.0 for o in ("d1", "est", "exact") for pp in PPS for b in BUD}
    for i in range(nq):
        ok = H[i] >= 0
        pos, d1 = H[i][ok], D1[i][ok]
        g = GP[i]
        cov_p += np.isin(lst_of[g], probed[i]).mean()
        cov_s += np.isin(g, pos).mean()
        lst = lst_of[pos]
        e2 = pq2[np.arange(M2), codes2[pos]].reshape(pos.size, dim)
        est = d1 - 2.0 * ((Q[i] - cent[lst]) * e2).sum(1) + kappa[pos]
        ex = ((vec[pos] - Q[i]) ** 2).sum(1)
        orders = {"d1": np.argsort(d1), "est": np.argsort(est), "exact": np.argsort(ex)}
        for pp in PPS:
            pg = lst * 100000 + off_of[pos] // pp
            gpg = lst_of[g] * 100000 + off_of[g] // pp
            lb[pp] += len(set(gpg.tolist()))
            for o, od in orders.items():
                u, first = np.unique(pg[od], return_index=True)
                ranked = u[np.argsort(first)]
                for b in BUD:
                    rec_[(o, pp, b)] += np.isin(gpg, ranked[:b]).mean()
    print(
        f"\n== nprobe {nprobe} shortlist {sl}: probe cov {cov_p / nq:.4f}  shortlist cov {cov_s / nq:.4f}"
    )
    for pp in PPS:
        print(f"  {pp:2d}/page  (pages holding the true top-10: {lb[pp] / nq:.2f})")
        for o in ("d1", "est", "exact"):
            print(
                f"    {o:5}",
                "  ".join(f"B{b}:{rec_[(o, pp, b)] / nq:.3f}" for b in BUD),
                flush=True,
            )
