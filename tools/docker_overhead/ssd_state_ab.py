"""Show how SSD state moves PipeANN QPS: a fresh copy of the index vs the original, interleaved.

Copies the BIGANN-10M PipeANN index (built by run_pipeann_spann.py) to `fresh_copy/`, then
alternates native searches on the original and the copy (page cache dropped before each) and
prints QPS plus the suite's own device probe. Identical bytes, code, config and CPU pinning; only
the flash state of the files differs. Re-run with --no-copy after a while, or after other heavy
writes, to watch the relation change. Finding (2026-09-29, isfcr): 2000-6850 QPS, tracking
fio 4 KiB QD64 IOPS of the file; Docker plays no part (see docs/METRICS.md, "Device State").

    ANN_SUITE_SUDO_PASSWORD=... uv run python tools/docker_overhead/ssd_state_ab.py out.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import run_pipeann_spann as h

from ann_suite.monitoring.host_state import probe_device

SPEC = h.ALGOS["pipeann"]
ORIGINAL = h.REPO / ".native/indices/pipeann/bigann10m"
FRESH = h.REPO / ".native/indices/pipeann/fresh_copy"
CONFIG = {
    "dimension": 128,
    "metric": "L2",
    "k": 10,
    "queries_path": str(h.DATA_DIR / "sift10m/queries.npy"),
    "ground_truth_path": str(h.DATA_DIR / "sift10m/ground_truth.npy"),
    "search_args": {**SPEC["search_args"], "Ls": 100},
}


def measure(tag: str, index_dir: Path, out: Path) -> None:
    h.drop_caches()
    probe = probe_device(index_dir)
    qps = h.run_native("pipeann", index_dir, CONFIG)["qps"]
    row = {"tag": tag, "qps": qps, "t": time.strftime("%H:%M:%S"), **probe}
    print(json.dumps(row), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=240, help="seconds between rounds")
    parser.add_argument("--no-copy", action="store_true", help="reuse an existing fresh_copy/")
    args = parser.parse_args()

    if not args.no_copy:
        shutil.rmtree(FRESH, ignore_errors=True)
        shutil.copytree(ORIGINAL, FRESH)
        subprocess.run(["sync"], check=True)
    for _ in range(args.rounds):
        for tag, index_dir in (("original", ORIGINAL), ("fresh", FRESH)) * 2:
            measure(tag, index_dir, args.out)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
