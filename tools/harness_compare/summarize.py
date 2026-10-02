"""Median-per-(irn, arm) table of a spann_harness_compare.py run.

uv run python tools/harness_compare/summarize.py results/harness_compare/<run>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

COLS = [
    ("recall10", "recall", "{:.4f}"),
    ("ids_identical_to_first_arm", "sameIDs", "{}"),
    ("suite_qps", "suiteQPS", "{:.0f}"),
    ("tool_qps", "binQPS", "{:.0f}"),
    ("tool_qps_cv", "binQPS_cv%", "{:.1f}"),
    ("qps_process_wall", "wallQPS", "{:.0f}"),
    ("suite_mean_latency_ms", "suiteLat", "{:.3f}"),
    ("tool_lat_mean_ms", "binLat", "{:.3f}"),
    ("tool_lat_p99_ms", "binP99", "{:.3f}"),
    ("cg_duration_s", "cgDur_s", "{:.2f}"),
    ("cg_cpu_s", "cpu_s", "{:.2f}"),
    ("cg_avg_cpu_pct", "avgCPU%", "{:.0f}"),
    ("cg_peak_mem_mb", "cgPeakMB", "{:.0f}"),
    ("cg_peak_file_mb", "fileMB", "{:.0f}"),
    ("maxrss_mb", "RSS_MB", "{:.0f}"),
    ("cg_read_mb", "readMB", "{:.0f}"),
    ("reads_per_q", "rdOps/q", "{:.2f}"),
    ("tool_disk_ios_per_q", "binIO/q", "{:.2f}"),
    ("tool_pages_per_q", "binPg/q", "{:.1f}"),
    ("probe_qd64_kiops", "probeK", "{:.0f}"),
]


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "raw.jsonl").read_text().splitlines()]
    for r in rows:
        if r.get("cg_read_ops") is not None:
            r["reads_per_q"] = r["cg_read_ops"] / 10_000
    out = ["| irn | arm | n | " + " | ".join(c[1] for c in COLS) + " |",
           "|" + "---|" * (len(COLS) + 3)]  # fmt: skip
    for irn in sorted({r["irn"] for r in rows}):
        for arm in dict.fromkeys(r["arm"] for r in rows):
            sel = [r for r in rows if r["irn"] == irn and r["arm"] == arm]
            if not sel:
                continue
            cells = []
            for key, _, fmt in COLS:
                if key == "tool_qps_cv":
                    v = [r["tool_qps"] for r in sel if r.get("tool_qps")]
                    val = 100 * statistics.pstdev(v) / statistics.mean(v) if len(v) > 1 else None
                elif key == "ids_identical_to_first_arm":
                    val = all(r[key] for r in sel)
                else:
                    v = [r[key] for r in sel if r.get(key) is not None]
                    val = statistics.median(v) if v else None
                cells.append("-" if val is None else fmt.format(val))
            out.append(f"| {irn} | {arm} | {len(sel)} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    (run / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
