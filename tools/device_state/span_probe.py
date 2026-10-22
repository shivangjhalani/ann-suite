"""Does the folded read level depend on how many bytes a probe spans? (protocol section 2)

The calibration survey found old index files read slower the larger they are (77 us at
1-2 GB, 91-94 us at 3-12 GB, 126-161 us at >= 58 GB), but size was mixed with file age and
origin. This tool holds the file fixed and varies the span: for each old file it probes a
window of --spans GB at a random 1 GiB-aligned offset (QD1 p50 over 2,000 O_DIRECT 4 KiB
reads, then fio QD64 for 4 s), plus the whole file, in random order, --reps times.
Read-only: it writes nothing to the drive.

    uv run python tools/device_state/span_probe.py --out results/span_probe/2026-10-23 \
        indices/starling/sift1b indices/diskann/sift100m ...
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from ann_suite.monitoring.host_state import FIO_SECONDS, PAGE, QD1_READS, largest_file

GIB = 1 << 30


def qd1_p50_us(path: Path, offset: int, size: int) -> float:
    buf = mmap.mmap(-1, PAGE)  # page-aligned, as O_DIRECT requires
    pages = size // PAGE
    lat = []
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        for _ in range(QD1_READS):
            start = time.perf_counter()
            os.preadv(fd, [buf], offset + random.randrange(pages) * PAGE)
            lat.append(time.perf_counter() - start)
    finally:
        os.close(fd)
    return round(statistics.median(lat) * 1e6, 1)


def qd64(path: Path, offset: int, size: int) -> dict[str, Any]:
    cmd = [
        "fio", "--name=span", f"--filename={path}", "--rw=randread", "--bs=4k",
        "--iodepth=64", "--ioengine=io_uring", "--direct=1", "--time_based",
        f"--runtime={FIO_SECONDS}", "--ramp_time=1", f"--offset={offset}", f"--size={size}",
        "--output-format=json",
    ]  # fmt: skip
    read = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    read = read["jobs"][0]["read"]
    return {
        "qd64_kiops": round(read["iops"] / 1000, 1),
        "qd64_lat_us": round(read["clat_ns"]["mean"] / 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("index_dirs", type=Path, nargs="+", help="probe each dir's largest file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spans", default="1,2,4,8,16,32,64,128,256", help="window sizes, GiB")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert shutil.which("fio"), "fio is required"

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "span_probe.jsonl"
    rng = random.Random(args.seed)
    jobs = []
    for d in args.index_dirs:
        f = largest_file(d)
        assert f is not None, d
        size = f.stat().st_size // PAGE * PAGE
        spans = [int(float(s) * GIB) for s in args.spans.split(",")]
        for span in [s for s in spans if s < size] + [size]:
            jobs += [(d, f, size, span)] * args.reps
    rng.shuffle(jobs)
    for d, f, size, span in jobs:
        offset = rng.randrange((size - span) // GIB + 1) * GIB if span < size else 0
        row = {
            "index": str(d), "file": f.name, "file_gib": round(size / GIB, 2),
            "span_gib": round(span / GIB, 2), "offset_gib": offset // GIB, "whole": span == size,
            "age_h": round((time.time() - f.stat().st_mtime) / 3600, 1),
            "qd1_p50_us": qd1_p50_us(f, offset, span), **qd64(f, offset, span),
            "t_wall": round(time.time(), 1),
        }  # fmt: skip
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
