"""Reference baseline: IVFADC with a refinement code, then exact re-ranking from SSD
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
    # Refinement quantizer, trained on the residuals the first level leaves.
    M2 = PQ2_M if dim % PQ2_M == 0 else max(m for m in range(1, PQ2_M + 1) if dim % m == 0)
    pq2 = faiss.ProductQuantizer(dim, M2, 8)
    tres = np.ascontiguousarray(train - centroids[train_assign])
    pq2.train(np.ascontiguousarray(tres - pq.decode(pq.compute_codes(tres))))
    del tres
    codes = np.empty((n, M), dtype=np.uint8)
    codes2 = np.empty((n, M2), dtype=np.uint8)
    for s in range(0, n, step):
        o = order[s : s + step]
        so = np.sort(o)
        back = np.argsort(np.argsort(o))
        resid = np.ascontiguousarray(np.asarray(data[so], dtype=np.float32) - centroids[assign[so]])
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
    ctx.save_mem("list_start", list_start)
    ctx.save_mem("codes", codes)
    ctx.save_mem("pq_cent", pq_cent)
    ctx.save_mem("codes2", codes2)
    ctx.save_mem("pq2_cent", pq2_cent.astype(np.float32))
    ctx.save_json(
        "layout", {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page, "M": M}
    )


class Searcher:
    """Numba port of ivfadc_refine_8_2.py's search (2026-10-05): the same algorithm,
    sweep and reads, with the ADC tables, the list scan (a bounded heap, as
    faiss does) and the refinement compiled by numba, so this reference's CPU
    per query is that of a native implementation (score v2 charges CPU).
    Everything outside this class is the original's, so the index is shared.
    """

    OPQ = False

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
        self.codes = ctx.mem("codes")
        self.codes2 = ctx.mem("codes2")
        self.pq2_cent = np.ascontiguousarray(ctx.mem("pq2_cent"), dtype=np.float32)
        self.shortlist = int(params["shortlist"])
        self.rot = ctx.mem("rot") if self.OPQ else None  # orthogonal: rotated distances are exact

    def search(self, query, k, io):
        q = query.astype(np.float32)
        probe = self._probe(q)
        if probe.size == 0:
            return np.zeros(0, dtype=np.int64)
        ls = self.list_start
        cen = self.centroids[probe]
        if self.rot is not None:  # codes live in the rotated space
            qs, cen = q @ self.rot.T, np.ascontiguousarray(cen @ self.rot.T)
        else:
            qs = q
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        qr, bias = self._scan_args(qs, cen)
        _, pos, lst = self.scan(
            qr, bias, self.pq_t, self.codes, ls[probe], ls[probe + 1], ls[probe],
            self.shortlist, self.ip,
        )  # fmt: skip
        # Refinement: re-estimate the shortlist from the two-level reconstruction.
        est = self.refine_est(
            qs, cen, self.pq_cent, self.codes, self.pq2_cent, self.codes2, pos, lst, self.ip
        )
        r = min(self.rerank, pos.size)
        cand = pos[np.argpartition(est, r - 1)[:r]] if r < pos.size else pos
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
