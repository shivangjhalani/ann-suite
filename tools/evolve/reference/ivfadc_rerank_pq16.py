"""Reference baseline: IVFADC with exact re-ranking from SSD (Jegou et al.,
"Searching in one billion vectors: re-rank with source coding", ICASSP 2011 --
the paper that introduced the BIGANN/SIFT1B set).

Coarse k-means (n/512 lists); PQ codes (PQ_M bytes/vector) of every vector held
in DRAM in list order; full vectors packed on SSD in list order. A query scans
the PQ codes of its nprobe nearest lists in memory (asymmetric distance), then
reads only the pages holding the `rerank` best candidates (one I/O round) and
ranks them exactly. Run on the evolved-program harness and added to the frontier
with `evolve_bench.py add-reference`, so evolved programs earn no credit for
rediscovering it. (This implementation is the first mutation of the 2026-10-03
OpenEvolve wiring test, which reproduced the design; PQ_M parameterized.)
"""

import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 8, "rerank": 12},
    {"nprobe": 16, "rerank": 16},
    {"nprobe": 32, "rerank": 24},
    {"nprobe": 48, "rerank": 32},
    {"nprobe": 64, "rerank": 48},
    {"nprobe": 96, "rerank": 64},
    {"nprobe": 128, "rerank": 96},
    {"nprobe": 192, "rerank": 160},
]

PQ_M = 16  # PQ subquantizers x 8 bits = bytes per vector held in DRAM

PAGE = 4096


def build(ctx):
    data = ctx.data
    n, dim = data.shape
    rec = dim * data.dtype.itemsize + 4  # vector + uint32 id
    per_page = PAGE // rec
    nlist = max(16, n // 512)
    faiss.omp_set_num_threads(ctx.threads)

    rng = np.random.default_rng(0)
    sample = np.asarray(
        data[np.sort(rng.choice(n, size=min(n, 20 * nlist), replace=False))], dtype=np.float32
    )
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

    # PQ codes (DRAM-resident shortlist filter), stored in cluster order.
    M = PQ_M if dim % PQ_M == 0 else max(m for m in range(1, PQ_M + 1) if dim % m == 0)
    pq = faiss.ProductQuantizer(dim, M, 8)
    pq.train(sample if sample.shape[0] >= 256 * 40 else np.asarray(data, dtype=np.float32))
    codes = np.empty((n, M), dtype=np.uint8)
    step = 1_000_000
    for s in range(0, n, step):
        o = order[s : s + step]
        codes[s : s + o.size] = pq.compute_codes(np.asarray(data[np.sort(o)], dtype=np.float32))[
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

        # ADC over DRAM-resident PQ codes: no I/O.
        qs = q.reshape(self.M, 1, self.dsub)
        lut = -(self.pq_cent * qs).sum(2) if self.ip else ((self.pq_cent - qs) ** 2).sum(2)
        approx = lut.ravel()[self.codes[pos].astype(np.int32) + self.lut_off].sum(1)

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
