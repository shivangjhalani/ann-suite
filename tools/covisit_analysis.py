"""Co-visitation layout headroom, with a packer that actually works.

first-touch (placing each node when whichever query first touches it) is a bad
packer: a query's nodes end up scattered across pages opened for other queries.
Validated on synthetic data -- 87% node coverage still gave only 15.00 -> 14.09
pages/query. It is kept below only as a baseline.

The real formulation is hypergraph partitioning: each query is a hyperedge over the
nodes it expands, and the objective is to minimise the number of pages each hyperedge
spans. GREEDY CO-VISIT PACKING approximates it: seed a page with the unplaced node of
highest co-visit weight, then fill it with that node's strongest co-visited unplaced
partners.

Note on the FLOOR: ceil(distinct/NPS) is only achievable if a single static layout can
satisfy every query at once. When queries overlap partially it is unachievable, so the
floor is a loose bound and the greedy number is the meaningful one.
"""
import numpy as np, sys
from collections import defaultdict

NPS = 15
path = sys.argv[1]
raw = np.fromfile(path, dtype=np.uint32)
qs, i = [], 0
while i < len(raw):
    n = int(raw[i]); i += 1
    qs.append(np.unique(raw[i:i+n])); i += n

uniq = np.array([len(q) for q in qs])
print(f"queries {len(qs):,}  distinct/query mean {uniq.mean():.2f}")
print(f"FLOOR (loose)        {np.ceil(uniq/NPS).mean():.2f} pages/query")
today = np.array([len(np.unique(q//NPS)) for q in qs])
print(f"TODAY (build order)  {today.mean():.2f} pages/query")

def greedy_pack(query_list, cap=64):
    """Seed each page with the highest-weight unplaced node, fill with its strongest
    co-visited unplaced partners. cap bounds per-node neighbour lists."""
    co = defaultdict(lambda: defaultdict(int))
    deg = defaultdict(int)
    for q in query_list:
        ql = q.tolist()
        for a in ql:
            deg[a] += len(ql) - 1
            d = co[a]
            for b in ql:
                if a != b and len(d) < 4000:
                    d[b] += 1
    order = sorted(deg, key=lambda x: -deg[x])
    slot, nxt, placed = {}, 0, set()
    for seed in order:
        if seed in placed: continue
        page = [seed]; placed.add(seed)
        for b, _ in sorted(co[seed].items(), key=lambda kv: -kv[1])[:cap]:
            if len(page) >= NPS: break
            if b not in placed:
                page.append(b); placed.add(b)
        for nid in page: slot[nid] = nxt; nxt += 1
        if len(page) < NPS: nxt += (NPS - len(page))
    return slot, nxt

def score(slot, query_list, label, fb):
    per = []
    for q in query_list:
        pages = {(slot[n]//NPS) if n in slot else fb + int(n)//NPS for n in q.tolist()}
        per.append(len(pages))
    m = float(np.mean(per)); print(f"{label:<38} {m:>8.2f} pages/query"); return m

s, nx = greedy_pack(qs); score(s, qs, "GREEDY co-visit (foreknowledge)", nx)
half = len(qs)//2
sa, na = greedy_pack(qs[:half])
cov = sum(1 for q in qs[half:] for n in q.tolist() if n in sa)
tot = sum(len(q) for q in qs[half:])
print(f"held-out coverage {cov:,}/{tot:,} ({cov/tot*100:.1f}%)")
score(sa, qs[half:], "GREEDY held-out (DEPLOYABLE)", na)
score({}, qs[half:], "  control: build order", 0)
