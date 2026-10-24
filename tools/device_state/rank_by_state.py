"""Do system rankings depend on device state? (analysis paper hypothesis A2, protocol sections 2-3)

For every system in a spec file (prebuilt index, search points bracketing a recall target) and
for each of --rounds rounds, in a random order drawn per round (RMIT, Abedi & Brecht 2017):
  * "fresh" arm: the index's large files are copied to a scratch dir (small files, and large files
    not matching the algorithm's build.probe_file glob, are hard-linked),
    --settle seconds pass, then every (search point, thread count) is run on the copy;
  * "folded" arm: the same points on the untouched original.
With --copy-per-point the fresh arm re-copies the index before every point instead, and with
--fresh-max-us a point counts as fresh ("gated") only if both its probes read at most that QD1;
a point that misses is re-copied and re-run, up to --attempts times. Every attempt is recorded.
A spec system may set "threads" to run only those thread counts ("fresh_threads": in the fresh
arm), and "memory_limit" / "cpu_affinity" to override its config. --max-written-gb stops the run
before any copy once the drive's SMART host writes since the start exceed that many GB.
The arm order is random per system and round. Every point goes through ann-suite (fresh container,
page cache dropped) and records the B1 device probes before and after, so each row pairs a QPS
with the drive state it ran in. Analyse with analyze_rank_by_state.py.
Run nothing else on the machine.
  uv run python tools/device_state/rank_by_state.py --spec configs/device_state/rank_10m.json \
      --out results/rank_by_state/2026-10-23 --rounds 6
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from ann_suite.core.config import load_config
from ann_suite.evaluator import BenchmarkEvaluator
from ann_suite.monitoring.device_state import drive_record

COND_KEYS = (
    "probe_file",
    "probe_qd1_p50_us",
    "probe_qd64_kiops",
    "post_probe_qd1_p50_us",
    "post_probe_qd64_kiops",
    "dev_smart_written_gb",
    "dev_temp_c",
    "index_file_age_h",
)
BIG = 512 << 20  # files at least this large are copied; the rest are hard-linked


def emit(out: Path, **row: Any) -> None:
    row["t_wall"] = round(time.time(), 1)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def point(system: dict, index_dir: Path, threads: int, params: dict) -> dict[str, Any]:
    config = load_config(Path(system["config"]))
    config.name = f"rank-by-state-{system['algo']}-T{threads}"
    algo = next(a for a in config.algorithms if a.name == system["algo"])
    algo.name = f"{system['algo']}-T{threads}"
    algo.build.prebuilt_path = index_dir
    algo.search.sweep = [params]
    algo.search.args["num_threads"] = threads
    if "memory_limit" in system:  # spec override, e.g. a cap that fits every system compared
        algo.memory_limit = system["memory_limit"]
    if "cpu_affinity" in system:  # e.g. all cores for a thread sweep
        algo.cpu_affinity = system["cpu_affinity"]
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
        "reads_per_query": result.disk_io.search_reads_per_query,
        "bytes_read_per_query": result.disk_io.search_bytes_read_per_query,
        "steady_reads_per_query": result.disk_io.search_steady_reads_per_query,
        "steady_bytes_read_per_query": result.disk_io.search_steady_bytes_read_per_query,
        "bytes_per_read_op": result.disk_io.search_avg_bytes_per_read_op,
        "device_read_iops": result.disk_io.search_device_read_iops,
        "avg_queue_depth": result.disk_io.search_avg_queue_depth,
        "cpu_time_per_query_ms": result.cpu.search_cpu_time_per_query_ms,
        "avg_cpu_percent": result.cpu.search_avg_cpu_percent,
        **{k: cond.get(k) for k in COND_KEYS},
    }


def copy_glob(system: dict) -> str:
    """The large files the search reads, as the algorithm's build.probe_file glob (default all)."""
    config = load_config(Path(system["config"]))
    algo = next(a for a in config.algorithms if a.name == system["algo"])
    return algo.build.probe_file or "*"


def copy_index(src: Path, dst: Path, pattern: str = "*") -> float:
    shutil.rmtree(dst, ignore_errors=True)
    t0 = time.monotonic()
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        elif f.stat().st_size >= BIG and fnmatch.fnmatch(f.name, pattern):
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst / rel)
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(f, dst / rel)
    os.sync()
    return time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument(
        "--settle",
        type=float,
        default=45,
        help="seconds between a copy and its first point",
    )
    ap.add_argument("--scratch", type=Path, default=Path("indices/_rank_by_state_scratch"))
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--only", nargs="*", help="system names to keep (smoke test)")
    ap.add_argument(
        "--fresh-max-us",
        type=float,
        help="fresh copy must read at most this QD1 (us): first point's pre-probe, or with "
        "--copy-per-point both probes of every point (see --attempts)",
    )
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument(
        "--copy-per-point",
        action="store_true",
        help="fresh arm: copy before every point, gate each point on its own probes",
    )
    ap.add_argument("--max-written-gb", type=float, help="stop before a copy past this many GB")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "rank_by_state.jsonl"
    systems = json.loads(args.spec.read_text())["systems"]
    if args.only:
        systems = [s for s in systems if s["name"] in args.only]
    for s in systems:
        (s["index_path"],) = [Path(p) for p in glob.glob(s["index"])]
        s["copy_glob"] = copy_glob(s)
    threads = [int(t) for t in args.threads.split(",")]
    rng = random.Random(args.seed)
    scratch = args.scratch.resolve()
    scratch.parent.mkdir(parents=True, exist_ok=True)

    def written() -> float | None:
        w = drive_record(scratch.parent).get("dev_smart_written_gb")
        return None if w is None else float(w)

    written0 = written()
    if args.max_written_gb is not None and written0 is None:
        raise SystemExit("--max-written-gb needs SMART host writes for the scratch drive")
    emit(out, phase="start", spec=str(args.spec), rounds=args.rounds, threads=threads,
         seed=args.seed, systems=[s["name"] for s in systems], settle=args.settle,
         fresh_max_us=args.fresh_max_us, attempts=args.attempts,
         copy_per_point=args.copy_per_point, max_written_gb=args.max_written_gb,
         smart_written_gb=written0)  # fmt: skip

    def copy(s: dict, rnd: int, attempt: int) -> None:
        w = written() if args.max_written_gb is not None else None
        if w is not None and written0 is not None and w - written0 > args.max_written_gb:
            shutil.rmtree(scratch, ignore_errors=True)
            emit(out, phase="budget_stop", round=rnd, system=s["name"],
                 written_gb=round(w - written0, 1))  # fmt: skip
            raise SystemExit(f"write budget of {args.max_written_gb} GB reached")
        secs = copy_index(s["index_path"], scratch, s["copy_glob"])
        emit(out, phase="copied", round=rnd, system=s["name"], copy_s=round(secs, 1),
             attempt=attempt)  # fmt: skip

    def run_arm(s: dict, arm: str, d: Path, rnd: int) -> None:
        for th in s.get("threads", threads):
            for params in s["params"]:
                row = point(s, d, th, params)
                emit(out, phase="point", round=rnd, system=s["name"], arm=arm, threads=th,
                     params=params, **row)  # fmt: skip

    def gated(row: dict) -> bool:
        qd1 = (row["probe_qd1_p50_us"], row["post_probe_qd1_p50_us"])
        return all(v is not None and v <= args.fresh_max_us for v in qd1)

    def run_fresh_per_point(s: dict, rnd: int) -> None:
        for th in s.get("fresh_threads", s.get("threads", threads)):
            for params in s["params"]:
                for attempt in range(1, args.attempts + 1):
                    copy(s, rnd, attempt)
                    time.sleep(args.settle)
                    row = point(s, scratch, th, params)
                    ok = args.fresh_max_us is not None and gated(row)
                    final = ok or args.fresh_max_us is None or attempt == args.attempts
                    emit(out, phase="point", round=rnd, system=s["name"], arm="fresh",
                         threads=th, params=params, attempt=attempt, gated=ok, final=final,
                         **row)  # fmt: skip
                    if final:
                        break
        shutil.rmtree(scratch, ignore_errors=True)

    for rnd in range(1, args.rounds + 1):
        order = rng.sample(systems, len(systems))
        for s in order:
            arms = rng.sample(["fresh", "folded"], 2)
            for arm in arms:
                if arm == "fresh" and args.copy_per_point:
                    run_fresh_per_point(s, rnd)
                elif arm == "fresh":
                    for attempt in range(1, args.attempts + 1):
                        copy(s, rnd, attempt)
                        time.sleep(args.settle)
                        if args.fresh_max_us is None or attempt == args.attempts:
                            break
                        row = point(s, scratch, s.get("threads", threads)[0], s["params"][0])
                        emit(out, phase="gate_check", round=rnd, system=s["name"],
                             attempt=attempt, **row)  # fmt: skip
                        if (row["probe_qd1_p50_us"] or 1e9) <= args.fresh_max_us:
                            break
                    run_arm(s, "fresh", scratch, rnd)
                    shutil.rmtree(scratch, ignore_errors=True)
                else:
                    run_arm(s, "folded", s["index_path"], rnd)
            time.sleep(args.settle)  # let the drive recover from the copy before the next system
    emit(out, phase="done")


if __name__ == "__main__":
    main()
