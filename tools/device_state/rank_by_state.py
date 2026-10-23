"""Do system rankings depend on device state? (analysis paper hypothesis A2, protocol sections 2-3)

For every system in a spec file (prebuilt index, search points bracketing a recall target) and
for each of --rounds rounds, in a random order drawn per round (RMIT, Abedi & Brecht 2017):
  * "fresh" arm: the index's large files are copied to a scratch dir (small files hard-linked),
    --settle seconds pass, then every (search point, thread count) is run on the copy;
  * "folded" arm: the same points on the untouched original.
The arm order is random per system and round. Every point goes through ann-suite (fresh container,
page cache dropped) and records the B1 device probes before and after, so each row pairs a QPS
with the drive state it ran in. Nothing is gated here; analyse with analyze_rank_by_state.py.
Run nothing else on the machine.
  uv run python tools/device_state/rank_by_state.py --spec configs/device_state/rank_10m.json \
      --out results/rank_by_state/2026-10-23 --rounds 6
"""

from __future__ import annotations

import argparse
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

COND_KEYS = (
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
    config.algorithms = [algo]
    (result,) = BenchmarkEvaluator(config).run()
    cond = result.run_conditions
    return {
        "qps": result.qps,
        "recall": result.recall,
        "mean_latency_ms": result.latency.mean_ms,
        **{k: cond.get(k) for k in COND_KEYS},
    }


def copy_index(src: Path, dst: Path) -> float:
    shutil.rmtree(dst, ignore_errors=True)
    t0 = time.monotonic()
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        elif f.stat().st_size >= BIG:
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
    ap.add_argument(
        "--scratch", type=Path, default=Path("indices/_rank_by_state_scratch")
    )
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--only", nargs="*", help="system names to keep (smoke test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "rank_by_state.jsonl"
    systems = json.loads(args.spec.read_text())["systems"]
    if args.only:
        systems = [s for s in systems if s["name"] in args.only]
    for s in systems:
        (s["index_path"],) = [Path(p) for p in glob.glob(s["index"])]
    threads = [int(t) for t in args.threads.split(",")]
    rng = random.Random(args.seed)
    scratch = args.scratch.resolve()
    emit(out, phase="start", spec=str(args.spec), rounds=args.rounds, threads=threads,
         seed=args.seed, systems=[s["name"] for s in systems])  # fmt: skip

    def run_arm(s: dict, arm: str, d: Path, rnd: int) -> None:
        for th in threads:
            for params in s["params"]:
                row = point(s, d, th, params)
                emit(out, phase="point", round=rnd, system=s["name"], arm=arm, threads=th,
                     params=params, **row)  # fmt: skip

    for rnd in range(1, args.rounds + 1):
        order = rng.sample(systems, len(systems))
        for s in order:
            arms = rng.sample(["fresh", "folded"], 2)
            for arm in arms:
                if arm == "fresh":
                    secs = copy_index(s["index_path"], scratch)
                    emit(
                        out,
                        phase="copied",
                        round=rnd,
                        system=s["name"],
                        copy_s=round(secs, 1),
                    )
                    time.sleep(args.settle)
                    run_arm(s, "fresh", scratch, rnd)
                    shutil.rmtree(scratch, ignore_errors=True)
                else:
                    run_arm(s, "folded", s["index_path"], rnd)
            time.sleep(
                args.settle
            )  # let the drive recover from the copy before the next system
    emit(out, phase="done")


if __name__ == "__main__":
    main()
