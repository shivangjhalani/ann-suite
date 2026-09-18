"""Is there any reuse to cache at all?

A cache can only pay for a node that is read MORE THAN ONCE. This measures the
access-count distribution directly, which every H(N) curve integrates over but
none of them exposes.

reuse = total accesses / distinct nodes touched. If that is ~1, the workload is
almost entirely singleton reads and no selection policy -- oracle or otherwise --
can produce hits, because the hits do not exist. H(N) would then be measuring
foreknowledge of which singletons get touched, not learnable structure.
"""
import numpy as np, sys

for tag, q in (("sift100m", 10000), ("sift10m", 10000)):
    f = f"/home/gem/shivang/ann-suite/results/visit_trace_{tag}.bin"
    try: c = np.fromfile(f, dtype=np.uint32)
    except Exception as e: print(f"{tag}: {e}"); continue
    tot = int(c.sum()); t = int((c > 0).sum())
    print(f"\n================ {tag} ================")
    print(f"accesses {tot:,}   touched {t:,}   A={tot/q:.4f}")
    print(f"REUSE (accesses per touched node): {tot/t:.4f}")

    bc = np.bincount(c[c > 0], minlength=2)
    print(f"\n{'visits':>8} {'nodes':>13} {'% of touched':>13} {'accesses':>13} {'% of accesses':>14} {'cum % acc':>10}")
    cum = 0
    for v in list(range(1, 11)) + [15, 20, 50, 100]:
        if v >= len(bc): break
        k = int(bc[v]); acc = k*v; cum += acc
        print(f"{v:>8} {k:>13,} {k/t*100:>12.3f}% {acc:>13,} {acc/tot*100:>13.3f}% {cum/tot*100:>9.2f}%")
    hi = int(bc[101:].sum()) if len(bc) > 101 else 0
    hi_acc = int((c[c > 100]).sum())
    print(f"{'>100':>8} {hi:>13,} {hi/t*100:>12.3f}% {hi_acc:>13,} {hi_acc/tot*100:>13.3f}%")

    once = int(bc[1]) if len(bc) > 1 else 0
    print(f"\nsingletons: {once:,} nodes ({once/t*100:.2f}% of touched) carrying "
          f"{once/tot*100:.2f}% of all accesses -- UNCACHEABLE BY CONSTRUCTION")
    multi = t - once
    print(f"nodes with >=2 visits: {multi:,} ({multi/t*100:.2f}%), "
          f"carrying {(tot-once)/tot*100:.2f}% of accesses")
    print(f"=> ceiling on ANY cache's hit rate at zero foreknowledge, if only "
          f"repeat-visited nodes are learnable: H_max = {(tot-once)/tot:.4f}")
    # pages to hold just the repeat-visited set, ideally co-located
    print(f"   that set is {multi:,} nodes = {int(np.ceil(multi/15)):,} pages "
          f"= {np.ceil(multi/15)*4096/1e6:.0f} MB co-located")
