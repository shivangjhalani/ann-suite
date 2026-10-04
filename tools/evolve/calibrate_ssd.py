"""Measure the benchmark SSD for score v2's latency and throughput model.

Writes results/evolve/ssd_model.json (config score.host_model):

  batch_ms  [[p, ms], ...]: time to complete one batch of p random 4 KB O_DIRECT
            reads issued together and waited for together (one I/O round), from a
            single job pinned to core 0 like the searches; p = 1, 2, 4, ..., 1024
  iops_max  saturated random 4 KB read rate (8 jobs x queue depth 128)

fio reads an existing large file on the index filesystem (default: the largest file
under index_dir, at least 8 GB), read-only and under the evaluation lock, so no
benchmark runs at the same time. Each value is the median of --repeats runs. Do not
calibrate on a freshly written file: the drive (a QLC SSD with an SLC write cache)
reads erratically while it folds recent writes, which made the first calibration
non-monotonic (1 page 324 us, 2 pages 97 us, 4 pages 948 us).

  .venv/bin/python tools/evolve/calibrate_ssd.py [--file F] [--runtime 8] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve_bench import REPO, _cfg, _evaluation_lock  # noqa: E402

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _fio(args: list[str]) -> dict[str, Any]:
    out = subprocess.run(
        ["fio", "--output-format=json", *args], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _common(path: Path, runtime: int) -> list[str]:
    return [
        f"--filename={path}",
        "--rw=randread",
        "--bs=4k",
        "--direct=1",
        "--ioengine=io_uring",
        "--time_based",
        f"--runtime={runtime}",
        "--ramp_time=1",
        "--norandommap",
        "--randrepeat=0",
    ]


def batch_ms(path: Path, p: int, runtime: int) -> tuple[float, float]:
    """(ms per batch, median completion latency in ms) for batches of p reads."""
    job = _fio(
        [
            "--name=batch",
            *_common(path, runtime),
            f"--iodepth={p}",
            f"--iodepth_batch_submit={p}",
            f"--iodepth_batch_complete_min={p}",
            "--cpus_allowed=0",
        ]
    )["jobs"][0]["read"]
    ios, ms = job["total_ios"], job["runtime"]
    clat = job["clat_ns"]["percentile"].get("50.000000", 0) / 1e6
    return ms / (ios / p), clat


def _largest_file(root: Path, min_bytes: int) -> Path:
    files = [f for f in root.rglob("*") if f.is_file() and not f.is_symlink()]
    best = max(files, key=lambda f: f.stat().st_size, default=None)
    if best is None or best.stat().st_size < min_bytes:
        raise SystemExit(f"no file of at least {min_bytes >> 30} GB under {root}; pass --file")
    return best


def iops_max(path: Path, runtime: int) -> float:
    res = _fio(
        [
            "--name=sat",
            *_common(path, runtime),
            "--iodepth=128",
            "--numjobs=8",
            "--group_reporting",
        ]
    )
    return float(res["jobs"][0]["read"]["iops"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/evolve/bigann10m.yaml")
    ap.add_argument("--file", type=Path, help="existing large file on the index device")
    ap.add_argument("--runtime", type=int, default=8, help="seconds per measurement")
    ap.add_argument("--repeats", type=int, default=3)
    ns = ap.parse_args()
    cfg = _cfg(ns.config)
    out = REPO / cfg["score"]["host_model"]
    with _evaluation_lock(cfg):
        scratch = ns.file or _largest_file(Path(cfg["index_dir"]), 8 << 30)
        model: dict[str, Any] = {"batch_ms": [], "runs": {}, "file": str(scratch)}
        for p in BATCHES:
            runs = [batch_ms(scratch, p, ns.runtime) for _ in range(ns.repeats)]
            ms = statistics.median(r[0] for r in runs)
            model["batch_ms"].append([p, ms])
            model["runs"][str(p)] = runs
            print(
                f"p={p:5d}: {ms * 1000:8.1f} us per round "
                f"(runs {', '.join(f'{r[0] * 1000:.0f}' for r in runs)}; "
                f"median read {runs[0][1] * 1000:.0f} us)",
                file=sys.stderr,
            )
        model["iops_max"] = statistics.median(
            iops_max(scratch, ns.runtime) for _ in range(ns.repeats)
        )
        print(f"iops_max: {model['iops_max']:,.0f}", file=sys.stderr)
        model["measured_at"] = time.time()
    ys = [ms for _p, ms in model["batch_ms"]]
    if any(b < a * 0.9 for a, b in zip(ys, ys[1:], strict=False)):
        print("WARNING: batch time is not monotonic in p; check for disk activity", file=sys.stderr)
        model["warning"] = "non-monotonic"
    out.write_text(json.dumps(model, indent=1))
    print(json.dumps(model))


if __name__ == "__main__":
    main()
