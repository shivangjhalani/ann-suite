"""Per-stage CPU of the deployed Searcher (its own numba kernels), single thread, pages
from the page cache. python prof3.py PROGRAM INDEX DATA METRIC 'nprobe,shortlist,pages' ..."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from algorithm.harness import SearchContext  # noqa: E402

prog, idx, data, metric = sys.argv[1:5]
ns: dict = {"__name__": "p"}
exec(compile(Path(prog).read_text(), prog, "exec"), ns)  # noqa: S102
faiss = ns["faiss"]
idx = Path(idx)
pages_n = {p.stem: p.stat().st_size // 4096 for p in (idx / "disk").glob("*.pages")}
mm = np.memmap(idx / "disk/lists.pages", dtype=np.uint8, mode="r").reshape(-1, 4096)
Q = np.load(Path(data) / "queries.npy")[:500]
ctx = SearchContext(idx, 1, metric, pages_n)
for spec in sys.argv[5:]:
    npb, sl, b = map(int, spec.split(","))
    s = ns["Searcher"](ctx, {"nprobe": npb, "shortlist": sl, "pages": b})
    cq = faiss.downcast_index(s.fs.quantizer)
    T = dict.fromkeys(["probe", "fastscan", "refine", "argsort", "pick", "score"], 0.0)
    for rep in range(2):
        for qq in Q:
            q = np.ascontiguousarray(qq, dtype=np.float32)
            t = [time.thread_time()]
            cq.search(q[None], s.fs.nprobe)
            t.append(time.thread_time())
            d1, h = s.fs.search(q[None], s.shortlist)
            t.append(time.thread_time())
            ok = h[0] >= 0
            pos, d1 = h[0][ok], d1[0][ok]
            lst = np.searchsorted(s.cstart, pos, side="right") - 1
            est = s.k_refine(d1, pos, lst, q, s.centroids, s.pq2, s.codes2, s.kappa, s.ks, s.ip)
            t.append(time.thread_time())
            o = np.argsort(est)
            t.append(time.thread_time())
            pg = s.page_start[lst] + (pos - s.cstart[lst]) // s.per_page
            pid = s.k_pick(o, pg, s.budget)
            t.append(time.thread_time())
            recs = np.ascontiguousarray(np.asarray(mm[pid])[:, : s.per_page * s.rec]).reshape(
                -1, s.rec
            )
            s.k_score(recs, s.dim, s.dtype.itemsize, q, s.ip, 10)
            t.append(time.thread_time())
            if rep:
                for k, a, c in zip(T, t, t[1:], strict=False):
                    T[k] += (c - a) * 1e3 / len(Q)
    tot = sum(T.values()) - T["probe"]
    print(
        spec,
        {k: round(v, 3) for k, v in T.items()},
        "total(excl. probe, inside fastscan)",
        round(tot, 3),
        flush=True,
    )
