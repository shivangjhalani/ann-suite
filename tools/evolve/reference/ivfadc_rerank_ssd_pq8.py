"""Reference baseline: IVFADC with exact re-ranking, 8 B PQ codes on SSD (Jegou et al.,
"Product quantization for nearest neighbor search", TPAMI 2011, and "Searching in
one billion vectors: re-rank with source coding", ICASSP 2011), with the inverted
lists of codes kept on SSD instead of in DRAM.

Coarse k-means (n/512 lists) is the only large DRAM structure (~10 MB at 10M), so
the design fits the 32 MB budget cell. Codes encode each vector's residual from its
list centroid, as in IVFADC. A query reads the PQ codes of its nprobe nearest lists
(packed contiguously in list order; round 1), scores them by asymmetric distance
with a per-list table ||(q - c)_m - y_mj||^2, then reads the pages holding the
`rerank` best candidates (round 2) and ranks them exactly. The first OpenEvolve
pilot (2026-10-03) found this design (plus adaptive re-ranking) and scored it on
DRAM alone, hence this reference: evolved programs earn no credit for
rediscovering it. (Residual encoding added 2026-10-03; the earlier version coded
raw vectors, weaker than the published design.)
Sweep (2026-10-04): points sit around the scored recall targets (0.90, 0.95) and
vary nprobe and the re-rank depth separately, so the reference is measured near
its own best pages/query at each target rather than along one fixed ratio.
8 B codes halve the code pages per probed list against the 16 B variant but
shortlist worse, so the sweep re-ranks deeper.
"""

import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 96, "rerank": 128},
    {"nprobe": 128, "rerank": 160},
    {"nprobe": 160, "rerank": 192},
    {"nprobe": 192, "rerank": 192},
    {"nprobe": 192, "rerank": 256},
    {"nprobe": 256, "rerank": 256},
    {"nprobe": 256, "rerank": 384},
    {"nprobe": 320, "rerank": 384},
]

PQ_M = 8  # PQ subquantizers x 8 bits = bytes per vector, stored on SSD

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

    # PQ codes in cluster order, packed contiguously on SSD: list c occupies bytes
    # [list_start[c] * M, list_start[c + 1] * M) of the "codes" file.
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
    flat = codes.ravel()
    npg = (flat.size + PAGE - 1) // PAGE
    code_pages = np.zeros(npg * PAGE, dtype=np.uint8)
    code_pages[: flat.size] = flat
    ctx.disk_writer("codes").write_pages(code_pages.reshape(npg, PAGE))
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
    ctx.save_mem("pq_cent", pq_cent)
    ctx.save_json(
        "layout", {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page, "M": M}
    )


class Searcher:
    def __init__(self, ctx, params):
        self.centroids = ctx.mem("centroids")
        self.cnorm = (self.centroids**2).sum(1)
        self.list_start = ctx.mem("list_start")
        self.pq_cent = ctx.mem("pq_cent")
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.M = lay["M"]
        self.dsub = self.dim // self.M
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        self.nprobe = int(params["nprobe"])
        self.rerank = int(params["rerank"])

    def search(self, query, k, io):
        q = query.astype(np.float32)
        d = -(self.centroids @ q) if self.ip else self.cnorm - 2.0 * (self.centroids @ q)
        npb = min(self.nprobe, d.size - 1)
        probe = np.sort(np.argpartition(d, npb)[:npb])
        ls = self.list_start
        probe = probe[ls[probe + 1] > ls[probe]]
        if probe.size == 0:
            return np.zeros(0, dtype=np.int64)

        # Round 1: the code pages of the probed lists.
        M = self.M
        b0, b1 = ls[probe] * M, ls[probe + 1] * M
        p0, p1 = b0 // 4096, (b1 - 1) // 4096
        cpages = np.unique(
            np.concatenate([np.arange(a, b + 1) for a, b in zip(p0, p1, strict=True)])
        )
        buf = io.read("codes", cpages).ravel()
        row = np.searchsorted(cpages, p0)  # a list's pages are consecutive rows of buf
        start = row * 4096 + (b0 - p0 * 4096)
        codes = np.concatenate(
            [buf[s : s + (e - b)] for s, b, e in zip(start, b0, b1, strict=True)]
        )
        pos = np.concatenate([np.arange(ls[c], ls[c + 1]) for c in probe])
        lid = np.repeat(np.arange(probe.size, dtype=np.int32), ls[probe + 1] - ls[probe])

        # IVFADC over the codes just read, one subquantizer at a time so per-query
        # temporaries stay small (this design targets the 32 MB budget): table
        # (P, 256) per subquantizer for the residual query q - c of each list.
        dsub = self.dsub
        codes = codes.reshape(-1, M)
        approx = np.zeros(codes.shape[0], dtype=np.float32)
        if self.ip:
            approx -= (self.centroids[probe] @ q)[lid]
            qr = np.broadcast_to(q, (probe.size, q.size))
        else:
            qr = q - self.centroids[probe]
        for m in range(M):
            y = self.pq_cent[m]  # (256, dsub)
            qm = qr[:, m * dsub : (m + 1) * dsub]
            if self.ip:
                lut = -(qm @ y.T)
            else:
                lut = (qm**2).sum(1)[:, None] - 2.0 * (qm @ y.T) + (y**2).sum(1)[None]
            approx += lut.ravel()[lid * 256 + codes[:, m]]

        r = min(self.rerank, pos.size)
        cand = pos[np.argpartition(approx, r - 1)[:r]] if r < pos.size else pos

        # Round 2: only the pages holding the shortlisted vectors.
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
