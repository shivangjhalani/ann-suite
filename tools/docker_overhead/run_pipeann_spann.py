"""PipeANN and SPANN arms of the Docker-overhead check, on BIGANN-10M.

Both runners are thin subprocess wrappers around C++ binaries resolved via
PIPEANN_BIN/SPTAG_BIN env vars, so "native" is: same runner.py, same binaries
(built at the image's pinned commit by setup_native_pipeann.sh /
setup_native_spann.sh), pointed at the index the container build produced
(so both arms search byte-identical index files).

PipeANN is the most Docker-sensitive case here: its default search mode uses
io_uring (optionally SQPOLL, a kernel-side polling thread per search thread),
which runs under Docker's seccomp=unconfined + cgroups. It also has a real
C++ open-loop driver (search_openloop), unlike DiskANN's Python-timed one.

Usage (from the repo root; builds the index once per algorithm, then runs
closed-loop, and for PipeANN also open-loop):
    uv run python tools/docker_overhead/run_pipeann_spann.py pipeann --repeats 3
    uv run python tools/docker_overhead/run_pipeann_spann.py spann --repeats 3
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cpuset import native_cmd_prefix

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/home/isfcr/data")
CPUS = "0-7"

ALGOS = {
    "pipeann": {
        "image": "ann-suite/pipeann:latest",
        "runner_dir": REPO / "library/algorithms/pipeann",
        "bin_env": "PIPEANN_BIN",
        "native_bin": REPO / ".native/pipeann/src/build/tests",
        "build_args": {"R": 64, "L": 128, "pq_bytes": 32, "build_memory_gb": 8, "num_threads": 8},
        "search_args": {"num_threads": 8, "beam_width": 32, "mode": 2, "nbr_type": "pq"},
        "ls_values": [20, 50, 100, 150],
    },
    "spann": {
        "image": "ann-suite/spann:latest",
        "runner_dir": REPO / "library/algorithms/spann",
        "bin_env": "SPTAG_BIN",
        "native_bin": REPO / ".native/spann/src/Release",
        "build_args": {
            "num_threads": 16,
            "max_check": 4096,
            "posting_page_limit": 3,
            "internal_result_num": 64,
        },
        "search_args": {"num_threads": 8, "posting_page_limit": 3},
        "ls_values": None,  # SPANN sweeps internal_result_num instead
        "search_sweep": [32, 64, 96, 160],
    },
}


def drop_caches() -> None:
    password = os.environ.get("ANN_SUITE_SUDO_PASSWORD")
    cmd = [
        "sudo",
        "-S" if password else "-n",
        "sh",
        "-c",
        "sync; echo 3 > /proc/sys/vm/drop_caches",
    ]
    subprocess.run(
        cmd, input=f"{password}\n" if password else None, text=True, capture_output=True, check=True
    )


def fix_ownership(path: Path) -> None:
    """Container runs write index/query/gt files as root; chown so the native
    (non-root) arm can also write its own queries.bin/gt.bin into the same dir."""
    user = getpass.getuser()
    subprocess.run(["sudo", "-n", "chown", "-R", f"{user}:{user}", str(path)], check=True)


def build_index(algo: str, index_dir: Path) -> None:
    """Build once via the container (the production path); both arms read this."""
    spec = ALGOS[algo]
    index_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_path": "/data/sift10m/base.npy",
        "index_path": "/data/index",
        "dimension": 128,
        "metric": "L2",
        "build_args": spec["build_args"],
    }
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size",
        "2g",
        "--security-opt",
        "seccomp=unconfined",
        "--cpuset-cpus",
        CPUS,
        "-v",
        f"{DATA_DIR}:/data",
        "-v",
        f"{index_dir}:/data/index",
        spec["image"],
        "--mode",
        "build",
        "--config",
        json.dumps(config),
    ]
    print("building index:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fix_ownership(index_dir)
    print(proc.stdout.strip().splitlines()[-1], file=sys.stderr)


def run_native(algo: str, index_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    spec = ALGOS[algo]
    env = {
        **os.environ,
        spec["bin_env"]: str(spec["native_bin"]),
        "PYTHONPATH": str(REPO / "library/algorithms"),
    }
    cfg = {**config, "index_path": str(index_dir)}
    cmd = [
        *native_cmd_prefix(CPUS),
        sys.executable,
        "-m",
        "algorithm.runner",
        "--mode",
        "search",
        "--config",
        json.dumps(cfg),
    ]
    proc = subprocess.run(
        cmd, cwd=spec["runner_dir"], capture_output=True, text=True, check=True, env=env
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_suite(algo: str, index_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    spec = ALGOS[algo]
    cfg = {**config, "index_path": "/data/index"}
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size",
        "2g",
        "--security-opt",
        "seccomp=unconfined",
        "--cpuset-cpus",
        CPUS,
        "-v",
        f"{DATA_DIR}:/data",
        "-v",
        f"{index_dir}:/data/index",
        spec["image"],
        "--mode",
        "search",
        "--config",
        json.dumps(cfg),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fix_ownership(index_dir)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def closed_loop_points(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec["ls_values"] is not None:
        return [{**spec["search_args"], "Ls": ls} for ls in spec["ls_values"]]
    return [{**spec["search_args"], "internal_result_num": n} for n in spec["search_sweep"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("algo", choices=list(ALGOS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-build", action="store_true", help="reuse an existing index")
    parser.add_argument("--openloop-rates", type=float, nargs="+", default=[200, 500, 1000])
    args = parser.parse_args()

    spec = ALGOS[args.algo]
    index_dir = REPO / ".native/indices" / args.algo / "bigann10m"
    if not args.skip_build:
        build_index(args.algo, index_dir)

    out = REPO / "results/docker_overhead" / f"{args.algo}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []

    base_search: dict[str, Any] = {
        "queries_path": "/data/sift10m/queries.npy",
        "ground_truth_path": "/data/sift10m/ground_truth.npy",
        "k": 10,
    }

    # Both runners read query/gt paths relative to their own /data mount inside
    # the container; the native arm uses the same paths, just resolved on the host.
    def resolve(cfg: dict[str, Any], native: bool) -> dict[str, Any]:
        if not native:
            return cfg
        return {
            **cfg,
            "queries_path": str(DATA_DIR / "sift10m/queries.npy"),
            "ground_truth_path": str(DATA_DIR / "sift10m/ground_truth.npy"),
        }

    for rep in range(args.repeats):
        for search_args in closed_loop_points(spec):
            cfg = {**base_search, "dimension": 128, "metric": "L2", "search_args": search_args}
            for arm, fn in (("suite", run_suite), ("native", run_native)):
                drop_caches()
                out_cfg = resolve(cfg, native=(arm == "native"))
                res = fn(args.algo, index_dir, out_cfg)
                row = {
                    "test": "closed_loop",
                    "algo": args.algo,
                    "arm": arm,
                    "rep": rep,
                    "point": {k: v for k, v in search_args.items() if k not in ("num_threads",)},
                    "qps": res.get("qps"),
                    "recall": res.get("recall"),
                    "mean_latency_ms": res.get("mean_latency_ms"),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

        if args.algo == "pipeann":
            for rate in args.openloop_rates:
                arrival = {
                    "mode": "poisson",
                    "rate_qps": rate,
                    "num_queries": 5000,
                    "warmup_fraction": 0.1,
                    "seed": 12345 + rep,
                }
                cfg = {
                    **base_search,
                    "dimension": 128,
                    "metric": "L2",
                    "search_args": {**spec["search_args"], "Ls": 100},
                    "arrival": arrival,
                }
                for arm, fn in (("suite", run_suite), ("native", run_native)):
                    drop_caches()
                    out_cfg = resolve(cfg, native=(arm == "native"))
                    res = fn(args.algo, index_dir, out_cfg)
                    ol = res.get("open_loop", {})
                    row = {
                        "test": "open_loop",
                        "algo": args.algo,
                        "arm": arm,
                        "rep": rep,
                        "rate_qps": rate,
                        "achieved_qps": ol.get("achieved_qps"),
                        "recall": ol.get("recall"),
                        "p99_ms": ol.get("p99_latency_ms"),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def summarize(test: str, key: str, group: str) -> str:
        groups: dict[Any, dict[str, list[float]]] = {}
        for r in rows:
            if r["test"] != test or r.get(key) is None:
                continue
            g = json.dumps(r.get(group)) if group == "point" else r.get(group)
            groups.setdefault(g, {}).setdefault(r["arm"], []).append(r[key])
        lines = [f"{test} median {key} by {group}"]
        for g, by_arm in sorted(groups.items(), key=str):
            if "suite" not in by_arm or "native" not in by_arm:
                continue
            s, n = statistics.median(by_arm["suite"]), statistics.median(by_arm["native"])
            lines.append(
                f"{g}: suite={s:.3g} native={n:.3g} ratio={s / n:.3f}" if n else f"{g}: n=0"
            )
        return "\n".join(lines)

    summary = "\n\n".join(
        [
            summarize("closed_loop", "qps", "point"),
            summarize("open_loop", "achieved_qps", "rate_qps") if args.algo == "pipeann" else "",
        ]
    ).strip()
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nresults: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
