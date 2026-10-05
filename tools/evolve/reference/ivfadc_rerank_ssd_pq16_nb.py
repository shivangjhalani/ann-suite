"""Reference baseline: IVFADC with exact re-ranking, PQ codes on SSD (Jegou et al.,
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
"""

import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 64, "rerank": 96},
    {"nprobe": 80, "rerank": 80},
    {"nprobe": 80, "rerank": 128},
    {"nprobe": 96, "rerank": 96},
    {"nprobe": 112, "rerank": 128},
    {"nprobe": 128, "rerank": 128},
    {"nprobe": 144, "rerank": 112},
    {"nprobe": 160, "rerank": 160},
]

PQ_M = 16  # PQ subquantizers x 8 bits = bytes per vector, stored on SSD

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
    """Numba port of ivfadc_rerank_ssd_pq16.py's search (2026-10-05): the same algorithm,
    sweep and reads, with the ADC tables, the list scan (a bounded heap, as
    faiss does) and the refinement compiled by numba, so this reference's CPU
    per query is that of a native implementation (score v2 charges CPU).
    Everything outside this class is the original's, so the index is shared.
    """

    _K = None

    @staticmethod
    def _kernels():
        """(scan, refine), compiled on first call (the harness's untimed warm-up)."""
        import numba as nb

        @nb.njit(fastmath=True, nogil=True)
        def scan(qr, bias, pq_t, codes, starts, ends, base, s, ip):
            # Lists p = 0..P-1 occupy rows starts[p]:ends[p] of codes; row i of list p
            # is vector position base[p] + i - starts[p]. L2: qr[p] = q - c_p
            # (residual query); IP: qr[0] = q and bias[p] = -<q, c_p>. pq_t is the
            # codebook as (M, dsub, 256), so table rows vectorise over the 256 codes.
            # Returns the s smallest ADC distances as (dist, position, list), unordered.
            M, dsub, K = pq_t.shape
            lut = np.empty((M, K), np.float32)
            hd = np.empty(s, np.float32)
            hp = np.empty(s, np.int64)
            hl = np.empty(s, np.int32)
            n = 0
            for p in range(starts.size):
                if p == 0 or not ip:
                    qv = qr[0] if ip else qr[p]
                    lut[:] = 0.0
                    for m in range(M):
                        row = lut[m]
                        for d in range(dsub):
                            qd = qv[m * dsub + d]
                            y = pq_t[m, d]
                            if ip:
                                for j in range(K):
                                    row[j] -= qd * y[j]
                            else:
                                for j in range(K):
                                    x = qd - y[j]
                                    row[j] += x * x
                b = bias[p]
                for i in range(starts[p], ends[p]):
                    dist = b
                    for m in range(M):
                        dist += lut[m, codes[i, m]]
                    if n < s:  # sift up into the max-heap
                        c = n
                        n += 1
                        while c > 0:
                            par = (c - 1) >> 1
                            if hd[par] >= dist:
                                break
                            hd[c], hp[c], hl[c] = hd[par], hp[par], hl[par]
                            c = par
                    elif dist < hd[0]:  # replace the root, sift down
                        c = 0
                        while True:
                            ch = 2 * c + 1
                            if ch >= n:
                                break
                            if ch + 1 < n and hd[ch + 1] > hd[ch]:
                                ch += 1
                            if hd[ch] <= dist:
                                break
                            hd[c], hp[c], hl[c] = hd[ch], hp[ch], hl[ch]
                            c = ch
                    else:
                        continue
                    hd[c], hp[c], hl[c] = dist, base[p] + i - starts[p], p
            return hd[:n], hp[:n], hl[:n]

        @nb.njit(fastmath=True, nogil=True)
        def refine(q, cen, pq_cent, codes, pq2_cent, codes2, pos, lst, ip):
            # Distance from q to centroid + level-1 + level-2 reconstruction.
            M1, _, d1 = pq_cent.shape
            M2, _, d2 = pq2_cent.shape
            dim = q.size
            xh = np.empty(dim, np.float32)
            est = np.empty(pos.size, np.float32)
            for i in range(pos.size):
                r = pos[i]
                xh[:] = cen[lst[i]]
                for m in range(M1):
                    y = pq_cent[m, codes[r, m]]
                    for d in range(d1):
                        xh[m * d1 + d] += y[d]
                for m in range(M2):
                    y = pq2_cent[m, codes2[r, m]]
                    for d in range(d2):
                        xh[m * d2 + d] += y[d]
                acc = np.float32(0.0)
                for d in range(dim):
                    if ip:
                        acc -= xh[d] * q[d]
                    else:
                        x = xh[d] - q[d]
                        acc += x * x
                est[i] = acc
            return est

        return scan, refine

    def __init__(self, ctx, params):
        if Searcher._K is None:
            Searcher._K = Searcher._kernels()
        self.scan, self.refine_est = Searcher._K
        self.centroids = ctx.mem("centroids")
        self.cnorm = (self.centroids**2).sum(1)
        self.list_start = ctx.mem("list_start")
        self.pq_cent = np.ascontiguousarray(ctx.mem("pq_cent"), dtype=np.float32)
        self.pq_t = np.ascontiguousarray(self.pq_cent.transpose(0, 2, 1))
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.M = lay["M"]
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        self.nprobe = int(params["nprobe"])
        self.rerank = int(params["rerank"])
        self.zero = np.zeros(self.centroids.shape[0], dtype=np.float32)

    def search(self, query, k, io):
        q = query.astype(np.float32)
        probe = self._probe(q)
        if probe.size == 0:
            return np.zeros(0, dtype=np.int64)
        ls = self.list_start
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
        ).reshape(-1, M)
        sizes = ls[probe + 1] - ls[probe]
        ends = np.cumsum(sizes)
        qr, bias = self._scan_args(q, self.centroids[probe])
        _, cand, _ = self.scan(
            qr, bias, self.pq_t, codes, ends - sizes, ends, ls[probe], self.rerank, self.ip
        )
        # Round 2: only the pages holding the shortlisted vectors.
        return self._rerank(q, cand, k, io)

    def _rerank(self, q, cand, k, io):
        # One I/O round: the pages holding the candidates, ranked exactly.
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

    def _probe(self, q):
        d = -(self.centroids @ q) if self.ip else self.cnorm - 2.0 * (self.centroids @ q)
        npb = min(self.nprobe, d.size - 1)
        probe = np.sort(np.argpartition(d, npb)[:npb])
        ls = self.list_start
        return probe[ls[probe + 1] > ls[probe]]

    def _scan_args(self, q, cen):
        # Residual queries (L2) or the query and per-list bias (IP).
        if self.ip:
            return q[None], -(cen @ q).astype(np.float32)
        return np.ascontiguousarray(q - cen, dtype=np.float32), self.zero[: cen.shape[0]]
