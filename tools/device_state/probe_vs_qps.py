"""Does search QPS follow the device_state probe? (analysis-paper protocol section 2)

Copies a prebuilt DiskANN index's files to a scratch dir, then alternates ann-suite search
points on the copy and on the untouched original for --duration seconds: copy/original at
1 thread, then copy/original at --threads threads, repeated. Every point records the B1
probes before and after the search, so each row pairs a device state with a QPS. The copy
drifts through fresh, intermediate and folded states on its own (calibration 2026-10-22);
the original is the folded control. Repeats --trials times with --gap seconds idle between.

Writes about the size of the index files per trial. Run nothing else on the machine.

    uv run python tools/device_state/probe_vs_qps.py --out results/probe_vs_qps/2026-10-22
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ann_suite.core.config import load_config
from ann_suite.evaluator import BenchmarkEvaluator

COND_KEYS = (
    "probe_qd1_p50_us",
    "probe_qd64_kiops",
    "post_probe_qd1_p50_us",
    "post_probe_qd64_kiops",
    "dev_smart_written_gb",
    "dev_temp_c",
    "index_file_age_h",
)


def emit(out: Path, **row: Any) -> None:
    row["t_wall"] = round(time.time(), 1)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def point(base: Path, index_dir: Path, threads: int, ls: int) -> dict[str, Any]:
    config = load_config(base)
    config.name = f"probe-vs-qps-T{threads}"
    algo = config.algorithms[0]
    algo.name = f"DiskANN-T{threads}"
    algo.build.prebuilt_path = index_dir
    algo.search.sweep = [{"Ls": ls}]
    algo.search.args["num_threads"] = threads
    config.algorithms = [algo]
    (result,) = BenchmarkEvaluator(config).run()
    cond = result.run_conditions
    return {
        "qps": result.qps,
        "recall": result.recall,
        "mean_latency_ms": result.latency.mean_ms,
        **{k: cond.get(k) for k in COND_KEYS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/docker_overhead_diskann_10m.yaml"))
    ap.add_argument("--original", type=Path, default=Path("/home/isfcr/diskann/datasets/sift10m"))
    ap.add_argument("--files", default="disk_index_*", help="glob of index files to copy")
    ap.add_argument("--scratch", type=Path, default=Path("indices/_probe_vs_qps_scratch"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ls", type=int, default=50)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--gap", type=float, default=600)
    ap.add_argument("--no-copy", action="store_true", help="smoke test: original only, 1 round")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "probe_vs_qps.jsonl"
    scratch = args.scratch.resolve()
    if args.no_copy:
        for threads in (1, args.threads):
            row = point(args.config, args.original, threads, args.ls)
            emit(out, phase="smoke", arm="original", threads=threads, **row)
        return

    for trial in range(1, args.trials + 1):
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        t0 = time.monotonic()
        for f in sorted(args.original.glob(args.files)):
            shutil.copyfile(f, scratch / f.name)
        os.sync()
        emit(out, phase="copied", trial=trial, copy_s=round(time.monotonic() - t0, 1))
        t0 = time.monotonic()
        rnd = 0
        try:
            while time.monotonic() - t0 < args.duration:
                rnd += 1
                for threads in (1, args.threads):
                    for arm, d in (("copy", scratch), ("original", args.original)):
                        t_s = round(time.monotonic() - t0, 1)
                        row = point(args.config, d, threads, args.ls)
                        emit(out, phase="point", trial=trial, round=rnd, arm=arm,
                             threads=threads, t_s=t_s, **row)  # fmt: skip
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            os.sync()
        if trial < args.trials:
            time.sleep(args.gap)
    emit(out, phase="done")


if __name__ == "__main__":
    main()
