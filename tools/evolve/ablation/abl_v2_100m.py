"""Design v2 with two-level k-means for the IVF centroids (100M-scale build). Otherwise identical to abl_v2.py.
IVF (n/512 lists, HNSW over the centroids); each list is reordered by recursive
bisection so a 4 KB page holds near neighbours. In DRAM, 40 B per vector whatever the
dimension: a 16 x 4-bit PQ of the residual scanned in SIMD (faiss IndexIVFPQFastScan,
ids = disk positions) and a 32 x 8-bit PQ of what it leaves, plus one float16 constant
so the second level costs a single 32 x 256 table per query. Vectors are zero-padded
to a multiple of 32 dimensions and, with OPQ, rotated. A query takes the `shortlist`
best positions of the SIMD scan of its nprobe lists, re-estimates them, reads the
`pages` pages that hold the best-ranked of them (one I/O round) and ranks the vectors
there exactly.
"""

# search: table-form refinement, page choice and scoring (numba)
import faiss
import numpy as np

SEARCH_POINTS = [
    {"nprobe": 64, "shortlist": 512, "pages": 16},
    {"nprobe": 96, "shortlist": 768, "pages": 24},
    {"nprobe": 128, "shortlist": 1024, "pages": 24},
    {"nprobe": 192, "shortlist": 1536, "pages": 24},
    {"nprobe": 192, "shortlist": 1536, "pages": 32},
    {"nprobe": 256, "shortlist": 2048, "pages": 32},
    {"nprobe": 256, "shortlist": 3072, "pages": 48},
    {"nprobe": 384, "shortlist": 4096, "pages": 64},
]
LOCAL = True
OPQ = True
M1, M2 = 16, 32  # fast-scan / refinement sub-quantizers (4 and 8 bits)
PAGE = 4096


def _local_order(vecs, per_page):
    """Balanced recursive bisection: every leaf is one full page of nearby vectors."""
    n = len(vecs)
    out = np.empty(n, dtype=np.int64)
    rng = np.random.default_rng(1)

    def split(idx, lo):
        m = idx.size
        if m <= per_page:
            out[lo : lo + m] = idx
            return
        x = vecs[idx]
        xc = x - x.mean(0)
        v = rng.standard_normal(x.shape[1]).astype(np.float32)
        for _ in range(5):  # power iteration for the principal axis
            v = xc.T @ (xc @ v)
            v /= np.linalg.norm(v) + 1e-12
        left = ((-(-m // per_page)) // 2) * per_page
        o = np.argsort(xc @ v, kind="stable")
        for _ in range(3):  # balanced 2-means refinement of the split direction
            w = xc[o[left:]].mean(0) - xc[o[:left]].mean(0)
            o = np.argsort(xc @ w, kind="stable")
        split(idx[o[:left]], lo)
        split(idx[o[left:]], lo + left)

    split(np.arange(n), 0)
    return out


def _two_level_centroids(sample, nlist, spherical):
    """nlist centroids by two-level k-means: ~sqrt(nlist) coarse clusters, then each
    gets a share of the centroids proportional to its size and its own k-means. Flat
    k-means on nlist ~ n/512 centroids grows as n^2 (6 h and unfinished at 100M)."""
    ncoarse = max(2, min(1024, nlist // 64))
    coarse = faiss.Kmeans(sample.shape[1], ncoarse, niter=8, seed=1, spherical=spherical)
    coarse.train(sample[:: max(1, len(sample) // (64 * ncoarse))])
    _, lab = coarse.index.search(sample, 1)
    lab = lab[:, 0]
    sizes = np.bincount(lab, minlength=ncoarse)
    share = nlist * sizes / sizes.sum()
    kc = np.maximum(1, np.floor(share).astype(np.int64))
    for i in np.argsort(-(share - kc))[: max(0, nlist - int(kc.sum()))]:
        kc[i] += 1
    while kc.sum() > nlist:  # the floor(...) >= 1 clamp can overshoot
        kc[np.argmax(kc)] -= 1
    out = []
    for i in range(ncoarse):
        pts = sample[lab == i]
        if kc[i] == 1 or len(pts) <= kc[i]:
            c = (
                pts[: kc[i]]
                if len(pts) >= kc[i]
                else np.resize(
                    pts if len(pts) else sample[:1],
                    (kc[i], pts.shape[1] if len(pts) else sample.shape[1]),
                )
            )
            out.append(pts.mean(0, keepdims=True) if kc[i] == 1 and len(pts) else c)
            continue
        km = faiss.Kmeans(pts.shape[1], int(kc[i]), niter=8, seed=1, spherical=spherical)
        km.train(pts)
        out.append(km.centroids)
    return np.concatenate(out).astype(np.float32)


def build(ctx):
    data = ctx.data
    n, dim = data.shape
    rec = dim * data.dtype.itemsize + 4  # vector + uint32 id
    per_page = PAGE // rec
    nlist = max(16, n // 512)
    faiss.omp_set_num_threads(ctx.threads)

    rng = np.random.default_rng(0)
    sample = np.asarray(
        data[np.sort(rng.choice(n, size=min(n, 20 * nlist), replace=False))],
        dtype=np.float32,
    )
    centroids = _two_level_centroids(sample, nlist, ctx.metric == "IP")

    # Zero-pad to a multiple of 32 dims (any dimension gets the same 16 + 32 sub-quantizers);
    # optionally rotate by OPQ. rot (dp x dim) has orthonormal columns, so distances and
    # inner products are unchanged: lists are assigned and coded in the rotated space.
    dp = -(-dim // 32) * 32
    rot = np.eye(dp, dtype=np.float32)[:, :dim]
    tr_idx = np.sort(rng.choice(n, size=min(n, 400_000), replace=False))
    trx = np.ascontiguousarray(np.asarray(data[tr_idx], dtype=np.float32))
    metric = faiss.METRIC_INNER_PRODUCT if ctx.metric == "IP" else faiss.METRIC_L2
    if OPQ:
        fl = faiss.IndexFlat(dim, metric)
        fl.add(centroids)
        _, a0 = fl.search(trx[:100_000], 1)
        opq = faiss.OPQMatrix(dp, M1)
        opq.niter = 20
        opq.train(np.ascontiguousarray((trx[:100_000] - centroids[a0[:, 0]]) @ rot.T))
        rot = faiss.vector_to_array(opq.A).reshape(dp, dp).astype(np.float32) @ rot
        del fl
    rcent = np.ascontiguousarray(centroids @ rot.T)

    # HNSW over the centroids makes assigning 10M vectors a few minutes, not an hour.
    quant = faiss.IndexHNSWFlat(dp, 32, metric)
    quant.hnsw.efSearch = 64
    quant.add(rcent)
    assign = np.empty(n, dtype=np.int64)
    step = 1_000_000
    for s in range(0, n, step):
        xs = np.ascontiguousarray(np.asarray(data[s : s + step], dtype=np.float32) @ rot.T)
        _, a = quant.search(xs, 1)
        assign[s : s + step] = a[:, 0]

    raw = np.asarray(data).view(np.uint8).reshape(n, -1)  # the build is not DRAM-capped
    order = np.argsort(assign, kind="stable")  # ids ascending within each cluster
    sizes = np.bincount(assign, minlength=nlist)

    # Page-local layout: each 4 KB page becomes a tight micro-cluster.
    from concurrent.futures import ThreadPoolExecutor

    starts = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=starts[1:])

    def _reorder(c):
        ids = order[starts[c] : starts[c + 1]]
        if ids.size <= per_page:
            return ids
        v = raw[ids].view(data.dtype).astype(np.float32)
        return ids[_local_order(v, per_page)]

    if LOCAL:
        with ThreadPoolExecutor(max(1, int(ctx.threads))) as ex:
            order = np.concatenate(list(ex.map(_reorder, range(nlist))))
    faiss.omp_set_num_threads(ctx.threads)
    pages_per = (sizes + per_page - 1) // per_page
    page_start = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(pages_per, out=page_start[1:])

    w = ctx.disk_writer("lists")
    vbytes = rec - 4
    pos = 0
    for c in range(nlist):
        ids = order[pos : pos + sizes[c]]
        pos += sizes[c]
        recs = np.zeros((pages_per[c] * per_page, rec), dtype=np.uint8)
        recs[:, vbytes:] = 0xFF  # padding records carry id 0xFFFFFFFF
        recs[: ids.size, :vbytes] = raw[ids]
        recs[: ids.size, vbytes:] = ids.astype(np.uint32).view(np.uint8).reshape(-1, 4)
        pages = np.zeros((pages_per[c], PAGE), dtype=np.uint8)
        pages[:, : per_page * rec] = recs.reshape(pages_per[c], per_page * rec)
        w.write_pages(pages)

    # Level 1: 4-bit fast-scan PQ (M1 sub-quantizers); level 2: 8-bit PQ (M2) of the
    # level-1 error e2. Vector ids are disk positions; codes2 / kappa rows likewise.
    # L2: |q-x|^2 ~ d1 - 2<q,e2> + kappa, kappa = 2<c+x1,e2> + |e2|^2 (c: centroid,
    # x1: level-1 reconstruction), so the query side of level 2 is one table <q, .>.
    cstart = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=cstart[1:])
    fs = faiss.IndexIVFPQFastScan(quant, dp, nlist, M1, 4, metric)
    fs.by_residual = True
    trr = np.ascontiguousarray(trx @ rot.T)
    fs.train(trr)
    _, tra = quant.search(trr, 1)
    tr1 = np.ascontiguousarray(trr - rcent[tra[:, 0]])
    tr2 = np.ascontiguousarray(tr1 - fs.pq.decode(fs.pq.compute_codes(tr1)))
    pq2 = faiss.ProductQuantizer(dp, M2, 8)
    pq2.train(tr2)
    codes2 = np.empty((n, pq2.code_size), dtype=np.uint8)
    kappa = np.empty(n, dtype=np.float32)
    for s in range(0, n, step):
        ids = order[s : s + step]
        xs = np.ascontiguousarray(raw[ids].view(data.dtype).astype(np.float32) @ rot.T)
        fs.add_with_ids(xs, np.arange(s, s + ids.size, dtype=np.int64))
        r1 = np.ascontiguousarray(xs - rcent[assign[ids]])
        x1 = fs.pq.decode(fs.pq.compute_codes(r1))
        c2 = pq2.compute_codes(np.ascontiguousarray(r1 - x1))
        e2 = pq2.decode(c2)
        codes2[s : s + ids.size] = c2
        kappa[s : s + ids.size] = 2.0 * ((rcent[assign[ids]] + x1) * e2).sum(1) + (e2 * e2).sum(1)
    got = np.array([fs.invlists.list_size(c) for c in range(nlist)])
    assert np.array_equal(got, sizes), "faiss list assignment differs from the page layout"
    # written straight to the index directory: serialize_index + deserialize_index would
    # hold three copies of the index at once (the search container is capped at 1 GB)
    faiss.write_index(fs, str(ctx._dir / "mem" / "fs.faiss"))
    ctx.save_mem("codes2", codes2)
    ctx.save_mem("pq2", faiss.vector_to_array(pq2.centroids).reshape(pq2.M, 256, pq2.dsub))
    kscale = max(1.0, float(np.abs(kappa).max()) / 60000.0)  # float16 range
    ctx.save_mem("kappa", (kappa / kscale).astype(np.float16))
    ctx.save_json("kscale", kscale)
    ctx.save_mem("rot", rot)
    ctx.save_mem("page_start", page_start)
    ctx.save_mem("cstart", cstart)
    ctx.save_json(
        "layout",
        {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page},
    )


class Searcher:
    _K = None

    @staticmethod
    def _kernels():
        """(refine, pick, score), compiled on first call (the harness's untimed warm-up)."""
        import numba as nb

        @nb.njit(nogil=True)
        def _h2f(h):  # IEEE half (as uint16 bits) -> float32; numba has no float16
            e = (h >> 10) & 0x1F
            m = np.float32(h & 0x3FF)
            v = (
                m * np.float32(2.0**-24)
                if e == 0
                else (np.float32(1.0) + m / np.float32(1024.0))
                * np.float32(2.0) ** (np.int32(e) - 15)
            )
            return -v if h >> 15 else v

        @nb.njit(fastmath=True, nogil=True)
        def refine(d1, pos, q, pq2, codes2, kappa, ks, ip):
            # est = d1 - 2<q,e2> + kappa (L2) or -(d1 + <q,e2>) (IP), <q,e2> by table
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
                est[i] = -(d1[i] + a) if ip else d1[i] - 2.0 * a + _h2f(kappa[r]) * ks

            return est

        @nb.njit(nogil=True)
        def pick(order, pg, budget):
            out = np.empty(budget, np.int64)
            n = 0
            for i in range(order.size):
                p = pg[order[i]]
                dup = False
                for j in range(n):
                    if out[j] == p:
                        dup = True
                        break
                if not dup:
                    out[n] = p
                    n += 1
                    if n == budget:
                        break
            return np.sort(out[:n])

        @nb.njit(fastmath=True, nogil=True)
        def score(recs, dim, isz, q, ip, k):
            # exact distances of uint8 / float32 records; padding (id 0xFFFFFFFF) skipped
            rec = recs.shape[1]
            n = recs.shape[0]
            dist = np.full(n, np.inf, np.float32)
            for i in range(n):
                if (
                    recs[i, rec - 1] == 255
                    and recs[i, rec - 2] == 255
                    and recs[i, rec - 3] == 255
                    and recs[i, rec - 4] == 255
                ):
                    continue
                acc = np.float32(0.0)
                if isz == 1:
                    for d in range(dim):
                        v = np.float32(recs[i, d])
                        acc += -v * q[d] if ip else (v - q[d]) * (v - q[d])
                else:
                    f = recs[i, : dim * 4].view(np.float32)
                    for d in range(dim):
                        acc += -f[d] * q[d] if ip else (f[d] - q[d]) * (f[d] - q[d])
                dist[i] = acc
            kk = min(k, n)
            top = np.argsort(dist)[:kk]
            ids = np.empty(kk, np.int64)
            for j in range(kk):
                i = top[j]
                ids[j] = (
                    np.int64(recs[i, rec - 4])
                    | (np.int64(recs[i, rec - 3]) << 8)
                    | (np.int64(recs[i, rec - 2]) << 16)
                    | (np.int64(recs[i, rec - 1]) << 24)
                )
            return ids

        return refine, pick, score

    def __init__(self, ctx, params):
        try:  # CPU time counts every thread: keep BLAS single-threaded
            from threadpoolctl import threadpool_limits

            self._tp = threadpool_limits(1)
        except Exception:
            pass
        faiss.omp_set_num_threads(1)
        self.fs = faiss.read_index(str(ctx._dir / "mem" / "fs.faiss"))
        self.codes2 = ctx.mem("codes2")
        self.pq2 = np.ascontiguousarray(ctx.mem("pq2"), dtype=np.float32)
        self.kappa = ctx.mem("kappa")
        self.ks = np.float32(ctx.json("kscale"))
        self.kappa = self.kappa.view(np.uint16)
        self.rot = np.ascontiguousarray(ctx.mem("rot"), dtype=np.float32)
        self.page_start = ctx.mem("page_start").astype(np.int64)
        self.cstart = ctx.mem("cstart").astype(np.int64)
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        nprobe = min(int(params["nprobe"]), self.fs.nlist - 1)
        self.fs.nprobe = nprobe
        faiss.downcast_index(self.fs.quantizer).hnsw.efSearch = max(16, nprobe)
        self.shortlist = int(params["shortlist"])
        self.budget = int(params["pages"])
        if Searcher._K is None:
            Searcher._K = Searcher._kernels()
        self.k_refine, self.k_pick, self.k_score = Searcher._K

    def search(self, query, k, io):
        q = np.ascontiguousarray(query, dtype=np.float32)
        qr = self.rot @ q
        d1, hits = self.fs.search(qr[None, :], self.shortlist)
        ok = hits[0] >= 0
        pos, d1 = hits[0][ok], d1[0][ok]
        if pos.size == 0:
            return np.zeros(k, dtype=np.int64)
        est = self.k_refine(d1, pos, qr, self.pq2, self.codes2, self.kappa, self.ks, self.ip)
        lst = np.searchsorted(self.cstart, pos, side="right") - 1
        pg = self.page_start[lst] + (pos - self.cstart[lst]) // self.per_page
        page_ids = self.k_pick(np.argsort(est), pg, self.budget)
        pages = io.read("lists", page_ids)
        recs = np.ascontiguousarray(pages[:, : self.per_page * self.rec]).reshape(-1, self.rec)
        return self.k_score(recs, self.dim, self.dtype.itemsize, q, self.ip, k)
