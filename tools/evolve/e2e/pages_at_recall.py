"""Pages read per query at a recall target for our e2e result files (recall and pages
do not depend on timing, so one repetition suffices): log-miss interpolation along the
recall-vs-pages Pareto front of the parameter grid, and the CPU ms of those points.
  python tools/evolve/e2e/pages_at_recall.py TARGETS NAME [NAME ...]
  e.g. pages_at_recall.py 0.9,0.95,0.97 v2-deep-t1-r1 v2-cluster-deep-t1-r1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_v2 import at, load, pareto  # noqa: E402

targets = [float(t) for t in sys.argv[1].split(",")]
print(f"{'file':30}" + "".join(f"  pages@{t}  cpu@{t}" for t in targets))
for name in sys.argv[2:]:
    rows = load(f"{name}.jsonl")
    front = pareto(rows, "pages", True)
    vals = []
    for t in targets:
        p, c = at(front, t, "pages"), at(front, t, "cpu_ms")
        vals.append(f"  {p:10.1f} {c:8.3f}" if p is not None else f"  {'n/a':>10} {'':8}")
    print(f"{name:30}" + "".join(vals))
