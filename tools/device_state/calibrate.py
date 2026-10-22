"""Calibrate device_state bands on one drive (analysis-paper protocol section 2).

Two phases, both writing JSON lines to --out:

  survey   probe the largest file of every index dir under --survey-root (read-only); old
           files show what "folded" looks like on this drive.
  trials   copy --source to a scratch dir on the same drive, then probe the copy and the
           untouched source in turns for --duration seconds (dense at first, then sparse);
           delete the copy and idle --gap seconds before the next trial. Probing the source
           too separates a drive-wide stall from the state of the new file.

Writes about size(--source) per trial. Run nothing else on the machine meanwhile.

    uv run python tools/device_state/calibrate.py --source indices/diskann/sift10m \
        --out results/device_state_calibration/2026-10-22
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ann_suite.monitoring.device_state import drive_record
from ann_suite.monitoring.host_state import largest_file, probe_device


def emit(out: Path, **row: Any) -> None:
    row["t_wall"] = round(time.time(), 1)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def survey(root: Path, out: Path) -> None:
    for d in sorted(p for p in root.glob("*/*") if p.is_dir()):
        f = largest_file(d)
        if f is None or f.stat().st_size < 1 << 30:
            continue
        rec = {**probe_device(d), **drive_record(f)}
        emit(out, phase="survey", index=str(d), size_gb=round(f.stat().st_size / 1e9, 2), **rec)


def schedule(duration: float, dense_s: float = 300, dense: float = 15, sparse: float = 60):
    t = 0.0
    while t <= duration:
        yield t
        t += dense if t < dense_s else sparse


def trial(n: int, source: Path, scratch: Path, duration: float, out: Path) -> None:
    src_file = largest_file(source)
    assert src_file is not None
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / src_file.name
    emit(out, phase="trial_start", trial=n, **drive_record(src_file))
    t0 = time.monotonic()
    shutil.copyfile(src_file, copy)
    with copy.open("rb+") as f:
        os.fsync(f.fileno())
    emit(out, phase="copied", trial=n, copy_s=round(time.monotonic() - t0, 1))
    t0 = time.monotonic()
    try:
        for at in schedule(duration):
            if (wait := at - (time.monotonic() - t0)) > 0:
                time.sleep(wait)
            for which, d in (("copy", scratch), ("source", source)):
                p = probe_device(d)
                emit(out, phase="trial", trial=n, which=which,
                     t_s=round(time.monotonic() - t0, 1), **p)  # fmt: skip
    finally:
        copy.unlink(missing_ok=True)
        os.sync()
    emit(out, phase="trial_end", trial=n, **drive_record(src_file))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True, help="index dir to copy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--survey-root", type=Path, default=Path("indices"))
    ap.add_argument("--scratch", type=Path, default=Path("indices/_calibration_scratch"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--gap", type=float, default=600)
    ap.add_argument("--skip-survey", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "calibration.jsonl"
    if not args.skip_survey:
        survey(args.survey_root, out)
    for n in range(1, args.trials + 1):
        trial(n, args.source, args.scratch, args.duration, out)
        if n < args.trials:
            time.sleep(args.gap)
    shutil.rmtree(args.scratch, ignore_errors=True)
    emit(out, phase="done")


if __name__ == "__main__":
    main()
