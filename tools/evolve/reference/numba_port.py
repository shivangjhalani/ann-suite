"""Write the numba ports of the IVFADC reference designs (score v2 charges CPU).

Each port `<name>_nb.py` is `<name>.py` with only its Searcher class replaced: the
same algorithm, sweep and reads, with the ADC tables, the code scan (a bounded
heap, as faiss does) and the refinement compiled by numba. Everything outside
Searcher is copied byte for byte, so the build hash is unchanged and the port
reuses the original's cached index. The kernels are created inside Searcher for
the same reason (top-level code is part of the build hash).

  python tools/evolve/reference/numba_port.py      # rewrites the *_nb.py files
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent

# name -> kind: "dram" (codes in DRAM, re-rank), "ssd" (codes on SSD, re-rank),
# "refine" (two-level codes in DRAM, re-estimate, re-rank); opq: rotated residuals.
PORTS = {
    "ivfadc_rerank_pq8": ("dram", False),
    "ivfadc_rerank_pq16": ("dram", False),
    "ivfadc_rerank_pq32": ("dram", False),
    "ivfadc_rerank_ssd_pq8": ("ssd", False),
    "ivfadc_rerank_ssd_pq16": ("ssd", False),
    "ivfadc_refine_8_2": ("refine", False),
    "ivfadc_refine_32_16": ("refine", False),
    "ivfadc_opq_refine_8_2": ("refine", True),
    "ivfadc_opq_refine_32_16": ("refine", True),
}

KERNELS = '''
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
'''

RERANK = """
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
"""

INIT_COMMON = """
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
"""

SEARCH = {
    "dram": """        self.codes = ctx.mem("codes")

    def search(self, query, k, io):
        q = query.astype(np.float32)
        probe = self._probe(q)
        if probe.size == 0:
            return np.zeros(0, dtype=np.int64)
        ls = self.list_start
        qr, bias = self._scan_args(q, self.centroids[probe])
        _, cand, _ = self.scan(
            qr, bias, self.pq_t, self.codes, ls[probe], ls[probe + 1], ls[probe],
            self.rerank, self.ip,
        )  # fmt: skip
        return self._rerank(q, cand, k, io)
""",
    "ssd": """
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
""",
    "refine": """        self.codes = ctx.mem("codes")
        self.codes2 = ctx.mem("codes2")
        self.pq2_cent = np.ascontiguousarray(ctx.mem("pq2_cent"), dtype=np.float32)
        self.shortlist = int(params["shortlist"])
        self.rot = ctx.mem("rot") if OPQ else None  # orthogonal: rotated distances are exact

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
""",
}


def port(name: str, kind: str, opq: bool) -> str:
    src = (HERE / f"{name}.py").read_text()
    head = src[: src.index("\nclass Searcher:")]
    doc = (
        f'    """Numba port of {name}.py\'s search (2026-10-05): the same algorithm,\n'
        "    sweep and reads, with the ADC tables, the list scan (a bounded heap, as\n"
        "    faiss does) and the refinement compiled by numba, so this reference's CPU\n"
        "    per query is that of a native implementation (score v2 charges CPU).\n"
        "    Everything outside this class is the original's, so the index is shared.\n"
        '    """\n'
    )
    opq_line = f"\n    OPQ = {opq}\n" if kind == "refine" else ""
    body = SEARCH[kind].replace("if OPQ", "if self.OPQ")
    return head + "\nclass Searcher:\n" + doc + opq_line + KERNELS + INIT_COMMON + body + RERANK


def main() -> None:
    for name, (kind, opq) in PORTS.items():
        (HERE / f"{name}_nb.py").write_text(port(name, kind, opq))
        print(f"wrote {name}_nb.py")


if __name__ == "__main__":
    main()
