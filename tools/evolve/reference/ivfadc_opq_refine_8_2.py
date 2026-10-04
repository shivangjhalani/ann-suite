"""Reference baseline: OPQ-rotated IVFADC with a refinement code, then exact re-ranking
from SSD: the refinement-code design (Jegou et al. 2011, below) with the residuals
rotated by an OPQ matrix first (Ge, He, Ke, Sun, "Optimized Product Quantization",
CVPR 2013 / TPAMI 2014), as in faiss's "OPQ..,IVF..,PQ..+R" indexes. Same layout and
sweep as ivfadc_refine_8_2.py; added 2026-10-04 after evolved programs used OPQ.

Base design (ivfadc_refine_8_2.py):
(Jegou, Tavenard, Douze, Amsaleg, "Searching in one billion vectors: re-rank with
source coding", ICASSP 2011 -- the paper that introduced BIGANN/SIFT1B).

Coarse k-means (n/512 lists). Every vector's residual from its list centroid is
coded with PQ_M bytes (first level), and the residual left by that code with
PQ2_M more bytes (the refinement code); both are held in DRAM in list order (10 B
per vector, ~100 MB at 10M: the 128 MB budget cells; 32 + 16 B is
ivfadc_refine_32_16.py). Full vectors are packed on
SSD in list order. A query scans the first-level codes of its nprobe nearest lists
with a per-list ADC table, re-estimates the `shortlist` best by the two-level
reconstruction (centroid + level-1 + level-2 decoded residuals), then reads only
the pages holding the `rerank` best re-estimated candidates (one I/O round) and
ranks them exactly. Added 2026-10-04: evolved programs reproduced the refinement
code, which the single-level references lack.
Sweep (2026-10-04): points sit around the scored recall targets (0.90, 0.95) and
vary nprobe and the re-rank depth separately, so the reference is measured near
its own best pages/query at each target rather than along one fixed ratio.
Per-query buffers grow with nprobe (~17 MB per 256 lists), so the sweep stays at
nprobe <= 256 to fit the 128 MB cell (codes + centroids are ~110 MB).
"""

import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 128, "shortlist": 384, "rerank": 96},
    {"nprobe": 128, "shortlist": 512, "rerank": 128},
    {"nprobe": 160, "shortlist": 640, "rerank": 128},
    {"nprobe": 160, "shortlist": 640, "rerank": 160},
    {"nprobe": 192, "shortlist": 768, "rerank": 160},
    {"nprobe": 192, "shortlist": 768, "rerank": 192},
    {"nprobe": 256, "shortlist": 1024, "rerank": 160},
    {"nprobe": 256, "shortlist": 1024, "rerank": 224},
]

PQ_M = 8  # first-level PQ bytes per vector (DRAM)
PQ2_M = 2  # refinement-code bytes per vector (DRAM)

PAGE = 4096
OPQ_NITER = 50  # faiss default


def build(ctx):
    data = ctx.data
    n, dim = data.shape
    rec = dim * data.dtype.itemsize + 4  # vector + uint32 id
    per_page = PAGE // rec
    nlist = max(16, n // 512)
    faiss.omp_set_num_threads(ctx.threads)

    rng = np.random.default_rng(0)
    sample_ids = np.sort(rng.choice(n, size=min(n, 20 * nlist), replace=False))
    sample = np.asarray(data[sample_ids], dtype=np.float32)
    km = faiss.Kmeans(dim, nlist, niter=8, seed=1, verbose=False, spherical=ctx.metric == "IP")
    km.train(sample)
    centroids = km.centroids.astype(np.float32)

    # HNSW over the centroids makes assigning 10M vectors a few minutes, not an hour.
    metric = faiss.METRIC_INNER_PRODUCT if ctx.metric == "IP" else faiss.METRIC_L2
    quant = faiss.IndexHNSWFlat(dim, 32, metric)
    quant.hnsw.efSearch = 64
    quant.add(centroids)
    assign = np.empty(n, dtype=np.int64)
    step = 1_000_000
    for s in range(0, n, step):
        _, a = quant.search(np.asarray(data[s : s + step], dtype=np.float32), 1)
        assign[s : s + step] = a[:, 0]

    raw = np.asarray(data).view(np.uint8).reshape(n, -1)  # the build is not DRAM-capped
    order = np.argsort(assign, kind="stable")  # ids ascending within each cluster
    sizes = np.bincount(assign, minlength=nlist)
    list_start = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=list_start[1:])

    # PQ codes of residuals x - c(x) (DRAM-resident shortlist filter), in cluster order.
    M = PQ_M if dim % PQ_M == 0 else max(m for m in range(1, PQ_M + 1) if dim % m == 0)
    pq = faiss.ProductQuantizer(dim, M, 8)
    train = sample if sample.shape[0] >= 256 * 40 else np.asarray(data, dtype=np.float32)
    train_assign = assign[sample_ids] if train is sample else assign
    tres = np.ascontiguousarray(train - centroids[train_assign])
    # OPQ: an orthogonal rotation of the residuals that balances their variance over
    # the PQ sub-spaces; both code levels live in the rotated space.
    opq = faiss.OPQMatrix(dim, M)
    opq.niter = OPQ_NITER
    opq.train(tres)
    rot = faiss.vector_to_array(opq.A).reshape(dim, dim).astype(np.float32)
    tres = np.ascontiguousarray(tres @ rot.T)
    pq.train(tres)
    # Refinement quantizer, trained on the residuals the first level leaves.
    M2 = PQ2_M if dim % PQ2_M == 0 else max(m for m in range(1, PQ2_M + 1) if dim % m == 0)
    pq2 = faiss.ProductQuantizer(dim, M2, 8)
    pq2.train(np.ascontiguousarray(tres - pq.decode(pq.compute_codes(tres))))
    del tres
    codes = np.empty((n, M), dtype=np.uint8)
    codes2 = np.empty((n, M2), dtype=np.uint8)
    for s in range(0, n, step):
        o = order[s : s + step]
        so = np.sort(o)
        back = np.argsort(np.argsort(o))
        resid = np.asarray(data[so], dtype=np.float32) - centroids[assign[so]]
        resid = np.ascontiguousarray(resid @ rot.T)
        c1 = pq.compute_codes(resid)
        codes[s : s + o.size] = c1[back]
        codes2[s : s + o.size] = pq2.compute_codes(np.ascontiguousarray(resid - pq.decode(c1)))[
            back
        ]
    pq_cent = faiss.vector_to_array(pq.centroids).reshape(M, 256, dim // M).astype(np.float32)
    pq2_cent = faiss.vector_to_array(pq2.centroids).reshape(M2, 256, dim // M2)

    # Full vectors packed contiguously in cluster order: position p -> page p // per_page.
    w = ctx.disk_writer("lists")
    vbytes = rec - 4
    chunk = per_page * 32768
    for s in range(0, n, chunk):
        ids = order[s : s + chunk]
        npg = (ids.size + per_page - 1) // per_page
        recs = np.zeros((npg * per_page, rec), dtype=np.uint8)
        recs[:, vbytes:] = 0xFF
        recs[: ids.size, :vbytes] = raw[ids]
        recs[: ids.size, vbytes:] = ids.astype(np.uint32).view(np.uint8).reshape(-1, 4)
        pages = np.zeros((npg, PAGE), dtype=np.uint8)
        pages[:, : per_page * rec] = recs.reshape(npg, per_page * rec)
        w.write_pages(pages)

    ctx.save_mem("centroids", centroids)
    ctx.save_mem("rot", rot)
    ctx.save_mem("list_start", list_start)
    ctx.save_mem("codes", codes)
    ctx.save_mem("pq_cent", pq_cent)
    ctx.save_mem("codes2", codes2)
    ctx.save_mem("pq2_cent", pq2_cent.astype(np.float32))
    ctx.save_json(
        "layout", {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page, "M": M}
    )


class Searcher:
    def __init__(self, ctx, params):
        self.centroids = ctx.mem("centroids")
        self.cnorm = (self.centroids**2).sum(1)
        self.rot = ctx.mem("rot")  # orthogonal: rotated distances are exact
        self.list_start = ctx.mem("list_start")
        self.codes = ctx.mem("codes")
        self.pq_cent = ctx.mem("pq_cent")
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.M = lay["M"]
        self.dsub = self.dim // self.M
        self.lut_off = (np.arange(self.M) * 256).astype(np.int32)
        self.pq_norm = (self.pq_cent**2).sum(2)  # (M, 256)
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        self.nprobe = int(params["nprobe"])
        self.rerank = int(params["rerank"])
        self.shortlist = int(params["shortlist"])
        self.codes2 = ctx.mem("codes2")
        self.pq2_cent = ctx.mem("pq2_cent")
        self.m1 = np.arange(self.M)
        self.m2 = np.arange(self.pq2_cent.shape[0])

    def search(self, query, k, io):
        q = query.astype(np.float32)
        qrot = q @ self.rot.T
        d = -(self.centroids @ q) if self.ip else self.cnorm - 2.0 * (self.centroids @ q)
        npb = min(self.nprobe, d.size - 1)
        probe = np.argpartition(d, npb)[:npb]
        ls = self.list_start
        pos = np.concatenate([np.arange(ls[c], ls[c + 1]) for c in probe])
        if pos.size == 0:
            return np.zeros(0, dtype=np.int64)
        lid = np.repeat(np.arange(probe.size, dtype=np.int32), ls[probe + 1] - ls[probe])

        # IVFADC over DRAM-resident residual codes: no I/O. One table per probed
        # list, (P, M, 256); for IP the residual term is shared and <q, c> is added.
        M, dsub = self.M, self.dsub
        if self.ip:
            lut = -np.einsum("md,mjd->mj", qrot.reshape(M, dsub), self.pq_cent)
            approx = lut.ravel()[self.codes[pos].astype(np.int32) + self.lut_off].sum(1)
            approx -= (self.centroids[probe] @ q)[lid]
        else:
            qr = (qrot - self.centroids[probe] @ self.rot.T).reshape(-1, M, dsub)
            luts = (
                (qr**2).sum(2)[:, :, None]
                - 2.0 * np.einsum("pmd,mjd->pmj", qr, self.pq_cent)
                + self.pq_norm[None]
            )
            flat = self.codes[pos].astype(np.int32) + self.lut_off + (lid * (M * 256))[:, None]
            approx = luts.ravel()[flat].sum(1)

        # Refinement: re-estimate the shortlist from the two-level reconstruction.
        s = min(self.shortlist, pos.size)
        sel = np.argpartition(approx, s - 1)[:s] if s < pos.size else np.arange(pos.size)
        p = pos[sel]
        xh = self.centroids[probe[lid[sel]]] @ self.rot.T
        xh = xh + self.pq_cent[self.m1, self.codes[p]].reshape(s, -1)
        xh += self.pq2_cent[self.m2, self.codes2[p]].reshape(s, -1)
        est = -(xh @ qrot) if self.ip else ((xh - qrot) ** 2).sum(1)
        r = min(self.rerank, s)
        cand = p[np.argpartition(est, r - 1)[:r]] if r < s else p

        # One I/O round: fetch only the pages holding the shortlisted vectors.
        upages, inv = np.unique(cand // self.per_page, return_inverse=True)
        pages = io.read("lists", upages)
        off = (cand % self.per_page) * self.rec
        recs = pages[inv.reshape(-1)[:, None], off[:, None] + np.arange(self.rec)]
        vbytes = self.rec - 4
        vecs = recs[:, :vbytes].copy().view(self.dtype).astype(np.float32)
        ids = recs[:, vbytes:].copy().view(np.uint32).ravel()
        dist = -(vecs @ q) if self.ip else ((vecs - q) ** 2).sum(1)
        kk = min(k, dist.size)
        top = np.argpartition(dist, kk - 1)[:kk] if kk < dist.size else np.arange(dist.size)
        return ids[top[np.argsort(dist[top])]].astype(np.int64)
