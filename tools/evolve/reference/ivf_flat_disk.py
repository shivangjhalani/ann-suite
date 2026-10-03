"""Reference baseline: IVF-Flat with inverted lists on SSD (k-means coarse
quantizer in DRAM, each list's full vectors stored contiguously on SSD; a query
reads all pages of its nprobe nearest lists in one round and ranks exactly).
The classic inverted-file design (Sivic & Zisserman 2003; Jegou et al. 2011),
i.e. SPANN without boundary replication. Identical to the OpenEvolve seed
(evolve/disk_ann/initial_program.py), so the evolution starts at score 0.
"""

import faiss
import numpy as np

SEARCH_POINTS = [{"nprobe": 2}, {"nprobe": 4}, {"nprobe": 8}, {"nprobe": 16}, {"nprobe": 24}]

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

    ctx.save_mem("centroids", centroids)
    ctx.save_mem("page_start", page_start)
    ctx.save_json("layout", {"dim": dim, "dtype": data.dtype.str, "rec": rec, "per_page": per_page})


class Searcher:
    def __init__(self, ctx, params):
        self.centroids = ctx.mem("centroids")
        self.cnorm = (self.centroids**2).sum(1)
        self.page_start = ctx.mem("page_start")
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        self.nprobe = int(params["nprobe"])

    def search(self, query, k, io):
        q = query.astype(np.float32)
        d = -(self.centroids @ q) if self.ip else self.cnorm - 2.0 * (self.centroids @ q)
        probe = np.argpartition(d, self.nprobe)[: self.nprobe]
        page_ids = np.concatenate(
            [np.arange(self.page_start[c], self.page_start[c + 1]) for c in probe]
        )
        pages = io.read("lists", page_ids)
        recs = pages[:, : self.per_page * self.rec].reshape(-1, self.rec)
        vbytes = self.rec - 4
        ids = recs[:, vbytes:].copy().view(np.uint32).ravel()
        valid = ids != 0xFFFFFFFF
        vecs = recs[valid, :vbytes].copy().view(self.dtype).astype(np.float32)
        ids = ids[valid]
        dist = -(vecs @ q) if self.ip else ((vecs - q) ** 2).sum(1)
        top = np.argpartition(dist, min(k, dist.size - 1))[:k]
        return ids[top[np.argsort(dist[top])]].astype(np.int64)
