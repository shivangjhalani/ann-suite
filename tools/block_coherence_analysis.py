"""Can a PAGE-NATIVE index reach the floor? Measures block coherence of true results.

Every refuted layout mechanism REORDERS an existing Vamana graph. The alternative is
to build the index over page-sized blocks so traversal routes block-to-block and each
page read delivers nps useful candidates by construction -- the DeepSeek Hierarchical
Sparse Indexer transposed (select blocks by pooled score, refine within).

That can only work if a query's ANSWER set is block-coherent. This measures it on
ground truth rather than on the traversal, so it is a property of the data and the
metric, independent of any graph or layout:

  tightness    mean pairwise distance within true top-k, relative to global scale
  spread       distinct kd-leaves (nps points each) the true top-k occupies

The earlier finding was that the 110-node VISITED set is only 2x tighter than random
(0.526) and spans 107.71 of 110 kd-leaves. If the true top-k is much tighter, the
traversal -- not the data -- is what is incoherent, and a page-native construction has
room. If the true top-k is equally spread, the data has no block structure at this
granularity and the direction is dead too.
"""
import numpy as np, sys
BASE, GT, NPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
X = np.load(BASE, mmap_mode="r")
gt = np.load(GT)
npts = X.shape[0]
rng = np.random.default_rng(0)
print(f"base {X.shape} {X.dtype}   ground truth {gt.shape}")

def pdm(V):
    d = ((V[:, None, :] - V[None, :, :])**2).sum(-1)
    return np.sqrt(d[np.triu_indices(len(V), 1)]).mean()

glob = pdm(np.asarray(X[rng.choice(npts, 200, replace=False)], dtype=np.float32))
print(f"global mean pairwise distance: {glob:.1f}\n")

# kd-tree grouping into leaves of NPS, same construction as the layout test
perm = np.arange(npts, dtype=np.int64)
Xc = np.ascontiguousarray(X[:, :32]).astype(np.float32)
stack = [(0, npts)]
while stack:
    lo, hi = stack.pop()
    if hi - lo <= NPS: continue
    idx = perm[lo:hi]; sub = Xc[idx]
    d = int(np.argmax(sub.var(axis=0))); mid = (hi - lo)//2
    o = np.argpartition(sub[:, d], mid)
    perm[lo:hi] = idx[o]
    stack.append((lo, lo+mid)); stack.append((lo+mid, hi))
leaf = np.empty(npts, dtype=np.int64); leaf[perm] = np.arange(npts)//NPS
print("kd grouping built\n")

print(f"{'top-k':>7} {'tightness vs global':>20} {'distinct groups':>17} {'ideal':>7}")
for k in (10, 50, 100):
    t, sp = [], []
    for row in gt[:200, :k]:
        r = row[row < npts]
        if len(r) < 2: continue
        t.append(pdm(np.asarray(X[np.sort(r)], dtype=np.float32))/glob)
        sp.append(len(np.unique(leaf[r])))
    print(f"{k:>7} {np.mean(t):>20.3f} {np.mean(sp):>17.2f} {int(np.ceil(k/NPS)):>7}")
print("\n(tightness 1.0 = no tighter than random; distinct groups == k means no block structure)")
