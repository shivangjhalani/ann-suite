"""How few pages could a query touch under an optimal on-disk layout?

Per Knowledge/cache-benefit-ceiling the only lever with real headroom is d, the
useful nodes delivered per page read. d is governed by CO-VISITATION -- whether the
nodes a single query expands share pages -- not by per-node visit frequency, which
is why the frequency-ranking predictors all failed.

Reads the per-query trace from covisit_trace.patch: [uint32 n][n x uint32 id]*.

Measures:
  floor        ceil(distinct nodes per query / nps) -- the best any layout could do
  today        distinct sectors per query under build-order layout
  first-touch  a profile-guided layout: walk queries in order, pack each query's
               not-yet-placed nodes into fresh pages. Cheap, deployable, and exactly
               what a profiling pass would produce.
  held-out     first-touch layout built on the FIRST half of queries, scored on the
               SECOND half -- the only number that is not foreknowledge.
"""
import numpy as np, sys, collections

NPS = 15
path = sys.argv[1] if len(sys.argv) > 1 else "/home/gem/shivang/ann-suite/results/covisit_sift100m.bin"
raw = np.fromfile(path, dtype=np.uint32)

qs, i = [], 0
while i < len(raw):
    n = int(raw[i]); i += 1
    qs.append(raw[i:i+n]); i += n
print(f"queries {len(qs):,}   total ids {sum(len(q) for q in qs):,}")

sizes = np.array([len(q) for q in qs])
uniq = np.array([len(np.unique(q)) for q in qs])
print(f"expanded per query: mean {sizes.mean():.2f}   distinct: mean {uniq.mean():.2f}")
print(f"FLOOR (perfect layout): {np.ceil(uniq/NPS).mean():.2f} pages/query")

today = np.array([len(np.unique(np.unique(q)//NPS)) for q in qs])
print(f"TODAY (build order):    {today.mean():.2f} pages/query")
print(f"  => headroom {today.mean()/np.ceil(uniq/NPS).mean():.2f}x")

def first_touch_layout(query_list):
    """Pack each query's not-yet-placed nodes into fresh pages, in query order."""
    slot = {}
    nxt = 0
    for q in query_list:
        for nid in np.unique(q):
            if nid not in slot:
                slot[nid] = nxt; nxt += 1
    return slot, nxt

def score(slot, query_list, label, fallback_pages):
    per = []
    for q in query_list:
        pages = set()
        for nid in np.unique(q):
            s = slot.get(nid)
            pages.add(s//NPS if s is not None else fallback_pages + int(nid)//NPS)
        per.append(len(pages))
    per = np.array(per)
    print(f"{label:<34} {per.mean():>8.2f} pages/query")
    return per.mean()

slot_all, n_all = first_touch_layout(qs)
score(slot_all, qs, "FIRST-TOUCH (foreknowledge)", n_all)

half = len(qs)//2
slot_a, n_a = first_touch_layout(qs[:half])
placed = sum(1 for q in qs[half:] for nid in np.unique(q) if nid in slot_a)
tot = sum(len(np.unique(q)) for q in qs[half:])
print(f"\nheld-out coverage: {placed:,}/{tot:,} ({placed/tot*100:.2f}%) of second-half "
      f"node touches were placed by the first half's layout")
score(slot_a, qs[half:], "HELD-OUT first-touch (DEPLOYABLE)", n_a)
score({}, qs[half:], "  control: build order, same queries", 0)
