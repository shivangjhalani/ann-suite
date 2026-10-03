"""Reference baseline: IVFADC with exact re-ranking from SSD (Jegou et al.,
"Searching in one billion vectors: re-rank with source coding", ICASSP 2011 --
the paper that introduced the BIGANN/SIFT1B set).

Coarse k-means (n/512 lists); PQ codes (PQ_M bytes/vector) of every vector's
residual from its list centroid held in DRAM in list order, as in IVFADC (Jegou
et al., TPAMI 2011, Sec. IV); full vectors packed on SSD in list order. A query
scans the codes of its nprobe nearest lists in memory with a per-list distance
table ||(q - c)_m - y_mj||^2, then reads only the pages holding the `rerank` best
candidates (one I/O round) and ranks them exactly. Run on the evolved-program
harness and added to the frontier with `evolve_bench.py add-reference`, so
evolved programs earn no credit for rediscovering it. (Started from the first
mutation of the 2026-10-03 OpenEvolve wiring test; PQ_M parameterized. Residual
encoding added 2026-10-03: the earlier version coded raw vectors, weaker than
the published design, and an evolved program scored on that gap.)
The 8-byte variant (80 MB of codes) covers the 128 MB budget cell; 8-byte codes
are coarse on SIFT, so the sweep re-ranks more candidates.
"""

import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 32, "rerank": 48},
    {"nprobe": 48, "rerank": 64},
    {"nprobe": 64, "rerank": 96},
    {"nprobe": 96, "rerank": 128},
    {"nprobe": 128, "rerank": 192},
    {"nprobe": 192, "rerank": 256},
    {"nprobe": 256, "rerank": 384},
    {"nprobe": 384, "rerank": 512},
]

PQ_M = 8  # PQ subquantizers x 8 bits = bytes per vector held in DRAM

PAGE = 4096


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
    pq.train(np.ascontiguousarray(train - centroids[train_assign]))
    codes = np.empty((n, M), dtype=np.uint8)
    for s in range(0, n, step):
        o = order[s : s + step]
        so = np.sort(o)
        resid = np.asarray(data[so], dtype=np.float32) - centroids[assign[so]]
        codes[s : s + o.size] = pq.compute_codes(np.ascontiguousarray(resid))[
            np.argsort(np.argsort(o))
        ]
    pq_cent = faiss.vector_to_array(pq.centroids).reshape(M, 256, dim // M).astype(np.float32)

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
    ctx.save_mem("list_start", list_start)
    ctx.save_mem("codes", codes)
    ctx.save_mem("pq_cent", pq_cent)
    ctx.save_json(
        "layout", {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page, "M": M}
    )


class Searcher:
    def __init__(self, ctx, params):
        self.centroids = ctx.mem("centroids")
        self.cnorm = (self.centroids**2).sum(1)
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

    def search(self, query, k, io):
        q = query.astype(np.float32)
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
            lut = -np.einsum("md,mjd->mj", q.reshape(M, dsub), self.pq_cent)
            approx = lut.ravel()[self.codes[pos].astype(np.int32) + self.lut_off].sum(1)
            approx -= (self.centroids[probe] @ q)[lid]
        else:
            qr = (q - self.centroids[probe]).reshape(-1, M, dsub)
            luts = (
                (qr**2).sum(2)[:, :, None]
                - 2.0 * np.einsum("pmd,mjd->pmj", qr, self.pq_cent)
                + self.pq_norm[None]
            )
            flat = self.codes[pos].astype(np.int32) + self.lut_off + (lid * (M * 256))[:, None]
            approx = luts.ravel()[flat].sum(1)

        r = min(self.rerank, pos.size)
        cand = pos[np.argpartition(approx, r - 1)[:r]] if r < pos.size else pos

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
