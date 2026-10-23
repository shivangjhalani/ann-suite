"""Does search QPS follow the device_state probe? (analysis-paper protocol section 2)

Copies a prebuilt DiskANN index's files to a scratch dir, then alternates ann-suite search
points on the copy and on the untouched original for --duration seconds: copy/original at
each of --threads in turn, repeated. One algorithm (--algo) at one search point (--params). Every point records the B1
probes before and after the search, so each row pairs a device state with a QPS. The copy
drifts through fresh, intermediate and folded states on its own (calibration 2026-10-22);
the original is the folded control. Repeats --trials times with --gap seconds idle between.

Writes about the size of the index files per trial. Run nothing else on the machine.

    uv run python tools/device_state/probe_vs_qps.py --out results/probe_vs_qps/2026-10-22
    uv run python tools/device_state/probe_vs_qps.py --out results/probe_vs_qps/pipeann \
        --config configs/evolve/baselines/pipeann.yaml --algo PipeANN-pq32 --files '*' \
        --original /home/isfcr/evolve_indices/baselines/PipeANN-pq32/bigann10m-q2k/L-128_R-64_... \
        --params '{"mode": 2, "beam_width": 32, "Ls": 25}'
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


def point(
    base: Path, algo_name: str | None, index_dir: Path, threads: int, params: dict[str, Any]
) -> dict[str, Any]:
    config = load_config(base)
    config.name = f"probe-vs-qps-T{threads}"
    algo = next(a for a in config.algorithms if algo_name in (None, a.name))
    algo.name = f"{algo_name or algo.name}-T{threads}"
    algo.build.prebuilt_path = index_dir
    algo.search.sweep = [params]
    algo.search.args["num_threads"] = threads
    if threads > 1:
        # In serial mode the DiskANN and HNSW runners ignore num_threads (queries run one at
        # a time), so multi-thread throughput needs batch mode; other runners ignore the flag.
        algo.search.batch_mode = True
    config.algorithms = [algo]
    (result,) = BenchmarkEvaluator(config).run()
    cond = result.run_conditions
    return {
        "qps": result.qps,
        "recall": result.recall,
        "mean_latency_ms": result.latency.mean_ms,
        "batch_mode": algo.search.batch_mode,
        **{k: cond.get(k) for k in COND_KEYS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/docker_overhead_diskann_10m.yaml"))
    ap.add_argument("--original", type=Path, default=Path("/home/isfcr/diskann/datasets/sift10m"))
    ap.add_argument("--algo", help="algorithm name in --config (default: the first)")
    ap.add_argument("--files", default="disk_index_*", help="glob of index files to copy")
    ap.add_argument("--scratch", type=Path, default=Path("indices/_probe_vs_qps_scratch"))
    ap.add_argument("--threads", default="1,8", help="thread counts, run in this order")
    ap.add_argument("--params", default='{"Ls": 50}', help="search point, JSON")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--gap", type=float, default=600)
    ap.add_argument("--no-copy", action="store_true", help="smoke test: original only, 1 round")
    ap.add_argument(
        "--max-failures", type=int, default=3, help="stop after this many failed points in a row"
    )
    args = ap.parse_args()
    if not [f for f in args.original.glob(args.files) if f.is_file()]:
        raise SystemExit(f"no index files matching {args.files!r} in {args.original}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "probe_vs_qps.jsonl"
    scratch = args.scratch.resolve()
    threads_list = [int(t) for t in args.threads.split(",")]
    params = json.loads(args.params)

    failures = 0

    def run(index_dir: Path, threads: int) -> dict[str, Any]:
        nonlocal failures
        row = point(args.config, args.algo, index_dir, threads, params)
        failures = 0 if row["qps"] else failures + 1
        if failures >= args.max_failures:
            emit(out, phase="aborted", reason=f"{failures} failed points in a row", **row)
            raise SystemExit(f"{failures} failed points in a row; last on {index_dir}")
        return row

    emit(out, phase="start", config=str(args.config), algo=args.algo, params=params,
         threads=threads_list, original=str(args.original))  # fmt: skip
    if args.no_copy:
        for threads in threads_list:
            row = run(args.original, threads)
            emit(out, phase="smoke", arm="original", threads=threads, **row)
        return

    for trial in range(1, args.trials + 1):
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        t0 = time.monotonic()
        for f in sorted(args.original.glob(args.files)):
            if f.is_file() and not f.is_symlink():
                shutil.copyfile(f, scratch / f.name)
        os.sync()
        emit(out, phase="copied", trial=trial, copy_s=round(time.monotonic() - t0, 1))
        t0 = time.monotonic()
        rnd = 0
        try:
            while time.monotonic() - t0 < args.duration:
                rnd += 1
                for threads in threads_list:
                    for arm, d in (("copy", scratch), ("original", args.original)):
                        t_s = round(time.monotonic() - t0, 1)
                        row = run(d, threads)
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
