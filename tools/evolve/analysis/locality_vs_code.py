"""Does page locality matter more when the in-DRAM code is weaker? On a built abl_v2 index
(locality-ordered lists): pages needed for a recall target when pages are taken in the
order of the fast-scan estimate only (8 B), the refined estimate (40 B) or exact distances,
with the real locality layout and with a cluster-order proxy (positions permuted at
random inside each list, then packed into pages). A true neighbour counts as found when
its page is read.
  python locality_vs_code.py INDEX DATA L2|IP NPROBE SHORTLIST [nq]
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

idx, data = Path(sys.argv[1]), Path(sys.argv[2])
ip = sys.argv[3] == "IP"
nprobe, sl = int(sys.argv[4]), int(sys.argv[5])
nq = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
faiss.omp_set_num_threads(8)
lay = json.loads((idx / "mem/layout.json").read_text())
dim, rec, per = lay["dim"], lay["rec"], lay["per_page"]
cs = np.load(idx / "mem/cstart.npy").astype(np.int64)
ps = np.load(idx / "mem/page_start.npy").astype(np.int64)
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
# cluster-order proxy: a random permutation of the offsets inside each list
rng = np.random.default_rng(0)
off_rand = np.empty(N, np.int64)
for c in range(nlist):
    off_rand[cs[c] : cs[c + 1]] = rng.permutation(cs[c + 1] - cs[c])
layouts = {"local": off_of // per, "cluster": off_rand // per}

fs = faiss.read_index(str(idx / "mem/fs.faiss"))
fs.nprobe = nprobe
faiss.downcast_index(fs.quantizer).hnsw.efSearch = max(16, nprobe)
rot = np.load(idx / "mem/rot.npy").astype(np.float32)
codes2 = np.load(idx / "mem/codes2.npy")
pq2 = np.load(idx / "mem/pq2.npy").astype(np.float32)
M2 = pq2.shape[0]
kappa = np.load(idx / "mem/kappa.npy").astype(np.float32) * float(
    json.loads((idx / "mem/kscale.json").read_text())
)
Q = np.load(data / "queries.npy")[:nq].astype(np.float32)
GP = inv[np.load(data / "ground_truth.npy")[:nq, :10]]
QR = Q @ rot.T
D1, H = fs.search(QR, sl)
BUD = (8, 12, 16, 24, 32, 48, 64, 96, 128)
hit = {(o, lay_, b): 0.0 for o in ("fastscan", "refined", "exact") for lay_ in layouts for b in BUD}
for i in range(nq):
    ok = H[i] >= 0
    pos, d1 = H[i][ok], D1[i][ok]
    lut = np.einsum("mjt,mt->mj", pq2, QR[i].reshape(M2, -1))
    a = lut[np.arange(M2), codes2[pos]].sum(1)
    if ip:
        orders = {
            "fastscan": np.argsort(-d1),
            "refined": np.argsort(-(d1 + a)),
            "exact": np.argsort(-(vec[pos] @ Q[i])),
        }
    else:
        orders = {
            "fastscan": np.argsort(d1),
            "refined": np.argsort(d1 - 2 * a + kappa[pos]),
            "exact": np.argsort(((vec[pos] - Q[i]) ** 2).sum(1)),
        }
    g = GP[i]
    for lname, pg_of in layouts.items():
        pg = lst_of[pos] * 1_000_000 + pg_of[pos]
        gpg = lst_of[g] * 1_000_000 + pg_of[g]
        for o, od in orders.items():
            u, first = np.unique(pg[od], return_index=True)
            ranked = u[np.argsort(first)]
            for b in BUD:
                hit[o, lname, b] += np.isin(gpg, ranked[:b]).mean()
print(
    f"{idx.name} nprobe {nprobe} shortlist {sl}, {per} vectors/page, recall@10 when reading B pages"
)
for o in ("fastscan", "refined", "exact"):
    for lname in layouts:
        print(f"  {o:8} {lname:7}", "  ".join(f"B{b}:{hit[o, lname, b] / nq:.3f}" for b in BUD))
