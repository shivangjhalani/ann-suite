"""Write the four ablation programs of the winner (evolve run main2, iteration 268).

Factors: layout (cluster order vs locality-ordered 4 KB pages) x rank (read the pages
of the best `rerank` vectors vs read the `pages` pages with the best code distance).
Everything is the winner's, with 8 B PQ only (the 128 MB cell); the two programs of a
layout differ only in SEARCH_POINTS and Searcher, so they share one built index.

  python make_ablation.py WINNER.py OUT_DIR
"""

import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text().splitlines(keepends=True)
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
L = lambda a, b: "".join(src[a - 1 : b])  # noqa: E731  1-based inclusive

kernels = L(52, 170)  # numba kernels (try/except block)
local_order = L(171, 199)
build = L(200, 308)
build = build.replace(
    "    with ThreadPoolExecutor(max(1, int(ctx.threads))) as ex:\n"
    "        order = np.concatenate(list(ex.map(_reorder, range(nlist))))\n",
    "    if LOCAL:\n"
    "        with ThreadPoolExecutor(max(1, int(ctx.threads))) as ex:\n"
    "            order = np.concatenate(list(ex.map(_reorder, range(nlist))))\n",
)
assert "if LOCAL:" in build

SEARCHER = """
class Searcher:
    def __init__(self, ctx, params):
        try:  # CPU time counts every thread: keep BLAS single-threaded
            from threadpoolctl import threadpool_limits

            self._tp = threadpool_limits(1)
        except Exception:
            pass
        faiss.omp_set_num_threads(1)
        self.centroids = np.ascontiguousarray(ctx.mem("centroids"), dtype=np.float32)
        self.page_start = ctx.mem("page_start").astype(np.int64)
        self.cstart = ctx.mem("cstart").astype(np.int64)
        lay = ctx.json("layout")
        self.dim, self.rec, self.per_page = lay["dim"], lay["rec"], lay["per_page"]
        self.dtype = np.dtype(lay["dtype"])
        self.ip = ctx.metric == "IP"
        self.nprobe = min(int(params["nprobe"]), len(self.centroids) - 1)
        self.rank = params["rank"]  # "page": best pages by code distance; "vec": pages of best vectors
        self.budget = int(params["budget"])  # pages read ("page") or vectors re-ranked ("vec")
        self.M = 8
        self.codes = np.ascontiguousarray(ctx.mem("codes8"))
        self.pqc = np.ascontiguousarray(ctx.mem("pqc8").astype(np.float32))
        self.pqc_norm = (self.pqc**2).sum(-1)
        self.pqcT = np.ascontiguousarray(self.pqc.transpose(0, 2, 1))  # (M, dsub, 256)
        metric = faiss.METRIC_INNER_PRODUCT if self.ip else faiss.METRIC_L2
        self.cq_index = faiss.IndexHNSWFlat(self.dim, 32, metric)
        self.cq_index.hnsw.efConstruction = 80
        self.cq_index.add(self.centroids)
        self.cq_index.hnsw.efSearch = max(64, int(1.5 * self.nprobe))
        self.pstart_hi = self.page_start[1:]

    def _lut(self, q, probe, cq_probe):
        M, dsub, P = self.M, self.pqc.shape[2], probe.size
        if self.ip:
            lut1 = -np.matmul(q.reshape(M, 1, dsub), self.pqcT)
            lut = np.ascontiguousarray(np.broadcast_to(lut1, (M, P, 256)), dtype=np.float32)
            return lut, (-cq_probe).astype(np.float32)
        qrt = np.ascontiguousarray(
            (q[None, :] - self.centroids[probe]).reshape(P, M, dsub).transpose(1, 0, 2)
        )
        lut = np.matmul(qrt, self.pqcT)
        lut *= -2.0
        lut += (qrt**2).sum(-1)[:, :, None]
        lut += self.pqc_norm[:, None, :]
        return np.ascontiguousarray(lut, dtype=np.float32), np.zeros(P, np.float32)

    def search(self, query, k, io):
        q = np.ascontiguousarray(query, dtype=np.float32)
        Dc, Ic = self.cq_index.search(q[None, :], self.nprobe)
        ok = Ic[0] >= 0
        probe = Ic[0][ok].astype(np.int64)
        if probe.size == 0:
            return np.zeros(k, dtype=np.int64)
        lut, base = self._lut(q, probe, Dc[0][ok].astype(np.float32))
        lo = self.cstart[probe]
        cnt = self.cstart[probe + 1] - lo
        ps = self.page_start[probe]
        if self.rank == "page":
            npg = int((self.pstart_hi[probe] - ps).sum())
            pd = np.empty(npg, np.float32)
            pp = np.empty(npg, np.int64)
            n = _page_min(lut, self.codes, lo, cnt, ps, base, self.per_page, pd, pp)
            pd, pp = pd[:n], pp[:n]
            r = min(n, self.budget)
            page_ids = pp[np.argpartition(pd, r - 1)[:r]] if r < n else pp
        else:
            tot = int(cnt.sum())
            adc = np.empty(tot, np.float32)
            _adc_scan(lut, self.codes, lo, cnt, adc)
            pidx = np.repeat(np.arange(probe.size), cnt)
            offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            adc += base[pidx]
            r = min(tot, self.budget)
            sel = np.argpartition(adc, r - 1)[:r] if r < tot else np.arange(tot)
            page_ids = np.unique(ps[pidx[sel]] + offs[sel] // self.per_page)
        pages = io.read("lists", np.sort(page_ids))
        recs = pages[:, : self.per_page * self.rec].reshape(-1, self.rec)
        vbytes = self.rec - 4
        ids = np.ascontiguousarray(recs[:, vbytes:]).view(np.uint32).ravel()
        if self.dtype == np.uint8:
            dist = np.empty(ids.size, np.float32)
            _score_u8(recs, self.dim, q, self.ip, dist)
        else:
            vecs = np.ascontiguousarray(recs[:, :vbytes]).view(self.dtype).astype(np.float32)
            dist = -(vecs @ q) if self.ip else ((vecs - q) ** 2).sum(1)
            dist[ids == 0xFFFFFFFF] = np.inf
        kk = min(k, dist.size)
        top = np.argpartition(dist, kk - 1)[:kk] if kk < dist.size else np.arange(dist.size)
        return ids[top[np.argsort(dist[top])]].astype(np.int64)
"""

HEAP_VEC = """            if Searcher._HEAP is None:
                Searcher._HEAP = Searcher._heap_scan()
            hp, hi = Searcher._HEAP(lut, self.codes, lo, cnt, base, min(int(cnt.sum()), self.budget))
            page_ids = np.unique(ps[hp] + hi // self.per_page)
"""
HEAP_KERNEL = '''
    _HEAP = None

    @staticmethod
    def _heap_scan():
        """Bounded max-heap ADC scan (as faiss does); built here so the build hash holds."""
        import numba as nb

        @nb.njit(fastmath=True, nogil=True)
        def scan(lut, codes, lo, cnt, base, s):
            M = lut.shape[0]
            hd = np.empty(s, np.float32)
            hp = np.empty(s, np.int32)
            hi = np.empty(s, np.int64)
            n = 0
            for p in range(lo.size):
                b = base[p]
                for j in range(cnt[p]):
                    acc = b
                    for m in range(M):
                        acc += lut[m, p, codes[lo[p] + j, m]]
                    if n < s:
                        c = n
                        n += 1
                        while c > 0:
                            par = (c - 1) >> 1
                            if hd[par] >= acc:
                                break
                            hd[c], hp[c], hi[c] = hd[par], hp[par], hi[par]
                            c = par
                    elif acc < hd[0]:
                        c = 0
                        while True:
                            ch = 2 * c + 1
                            if ch >= n:
                                break
                            if ch + 1 < n and hd[ch + 1] > hd[ch]:
                                ch += 1
                            if hd[ch] <= acc:
                                break
                            hd[c], hp[c], hi[c] = hd[ch], hp[ch], hi[ch]
                            c = ch
                    else:
                        continue
                    hd[c], hp[c], hi[c] = acc, p, j
            return hp[:n], hi[:n]

        return scan
'''

# (nprobe, budget) sweeps; page budgets are pages read, vec budgets vectors re-ranked.
SWEEP = {
    "page": [
        (64, 80),
        (72, 100),
        (96, 140),
        (128, 200),
        (160, 240),
        (200, 300),
        (256, 400),
        (320, 500),
    ],
    "vec": [
        (128, 100),
        (192, 160),
        (256, 200),
        (256, 300),
        (384, 300),
        (384, 450),
        (512, 500),
        (768, 700),
    ],
}
DOC = '''"""Ablation of the evolved winner (OpenEvolve main2 iteration 268), layout={layout}
(the rank mode is set per search point; programs of a layout share one index). IVF (n/512 lists, HNSW over the centroids) + 8 B residual PQ codes in DRAM
(80 MB). layout=local reorders each list by recursive bisection so a 4 KB page holds
near neighbours; layout=cluster keeps ids ascending inside a list (IVFADC+R).
rank=vec reads the pages of the best `budget` vectors by ADC (IVFADC+R); rank=page reads
the `budget` pages whose best code distance is smallest. Generated by make_ablation.py.
"""
'''
VEC_BRANCH_START = "            tot = int(cnt.sum())\n"
VEC_BRANCH_END = "            page_ids = np.unique(ps[pidx[sel]] + offs[sel] // self.per_page)\n"
a = SEARCHER.index(VEC_BRANCH_START)
b = SEARCHER.index(VEC_BRANCH_END) + len(VEC_BRANCH_END)
HEAP_SEARCHER = SEARCHER[:a] + HEAP_VEC + SEARCHER[b:]
HEAP_SEARCHER = HEAP_SEARCHER.replace("    def _lut(self", HEAP_KERNEL + "\n    def _lut(self", 1)
for layout in ("cluster", "local"):
    for rank in ("vec", "page"):
        pts = ",\n".join(
            f'    {{"nprobe": {a}, "rank": "{rank}", "budget": {b}}}' for a, b in SWEEP[rank]
        )
        text = (
            DOC.format(layout=layout)
            + f"# rank: {rank}\n"
            + "import faiss\nimport numpy as np\n\n"
            + f"SEARCH_POINTS = [\n{pts},\n]\n"
            + f"LOCAL = {layout == 'local'}\nPAGE = 4096\nCODE_MS = (8,)\nREFINE = {{}}\n"
            + kernels
            + local_order
            + build
            + SEARCHER
        )
        (out / f"abl_{layout}_{rank}.py").write_text(text)
        print("wrote", f"abl_{layout}_{rank}.py")

# Fifth program: locality layout, rank by vector, fused bounded-heap scan (the
# fair by-vector control; shares the local index).
pts = ",\n".join(
    f'    {{"nprobe": {a}, "rank": "vec", "budget": {b}}}' for a, b in SWEEP["vec"]
)
text = (
    DOC.format(layout="local")
    + "# rank: vec (fused heap scan)\n"
    + "import faiss\nimport numpy as np\n\n"
    + f"SEARCH_POINTS = [\n{pts},\n]\n"
    + "LOCAL = True\nPAGE = 4096\nCODE_MS = (8,)\nREFINE = {}\n"
    + kernels + local_order + build + HEAP_SEARCHER
)
(out / "abl_local_heap.py").write_text(text)
print("wrote abl_local_heap.py")
