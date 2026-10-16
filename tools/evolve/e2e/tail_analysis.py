"""Tail-latency breakdown from run_tail.sh dumps: for each run and point, latency
percentiles, and for the slowest 0.5% of queries how much of their excess over the
median is CPU (compute or GC pauses; thread_time) versus wall-only (I/O wait or
scheduling), plus their pages.
  python tools/evolve/e2e/tail_analysis.py results/evolve/e2e/tail
"""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(sys.argv[1])
    print(
        f"{'run':22} {'point':38} {'p50':>6} {'p99':>6} {'p99.9':>6} {'max':>6} | slowest 0.5%: excess cpu% pages(med)"
    )
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.glob("p*.npz")):
            z = np.load(f)
            lat, cpu, pg = z["lat_ms"], z["cpu_ms"], z["pages"]
            med_l, med_c = np.median(lat), np.median(cpu)
            slow = np.argsort(lat)[-max(1, len(lat) // 200) :]
            excess = (lat[slow] - med_l).sum()
            cpu_ex = np.clip(cpu[slow] - med_c, 0, None).sum()
            pt = json.loads(str(z["params"]))
            print(
                f"{d.name:22} {json.dumps(pt):38} {med_l:6.3f} {np.percentile(lat, 99):6.3f} "
                f"{np.percentile(lat, 99.9):6.3f} {lat.max():6.3f} | {excess / len(slow):6.3f} ms "
                f"{100 * cpu_ex / max(excess, 1e-9):5.1f}% {np.median(pg[slow]):5.1f} ({np.median(pg):.1f})"
            )


if __name__ == "__main__":
    main()
