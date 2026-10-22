"""SPANN searches under TapeANN's native benchmark conditions ("equal grounds").

Imports /home/isfcr/tapeann/engine/bench/run_native.py (not modified), so SPANN runs exactly the way
TapeANN, DiskANN and Starling were measured there: one search thread on one pinned core (cgroup
cpuset in a user systemd scope), MemoryMax, page cache dropped before every process, the first 2000
BIGANN queries, recall@10 recomputed from the returned ids against the same ground truth.

What this driver adds on top of that harness:
  * one memory ceiling for every system at a scale (the harness uses a different cap per system);
  * repeats, interleaved, with the order reversed on odd repeats;
  * an SSD-state probe (fio QD64 + Python QD1, ann-suite's host_state) before every point, on one
    fixed file per scale, because flash state alone moves absolute QPS 3-5x;
  * TapeANN anchor points run in the same session, so cross-system numbers share one SSD state;
  * a load-only baseline (1 query) per system and repeat. rbytes/rios in the harness include loading
    the index (SPANN's head index is ~1/3 of its bytes at 10M); the summary subtracts the baseline.
Results go to results/equal_grounds/<scale>_<time>/; the TapeANN repo's native.jsonl is untouched.

    /home/isfcr/tapeann/engine/.venv/bin/python tools/harness_compare/equal_grounds.py run --scale 10m
    /home/isfcr/tapeann/engine/.venv/bin/python tools/harness_compare/equal_grounds.py summarize <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

from ssd_gate import SsdGate

REPO = Path(__file__).resolve().parents[2]
RUN_NATIVE = Path("/home/isfcr/tapeann/engine/bench/run_native.py")
HOST_STATE = REPO / "src/ann_suite/monitoring/host_state.py"
SPANN_ROOT = Path("/home/isfcr/spann/indexes")
PRIOR = RUN_NATIVE.parents[1] / "results/native.jsonl"
NVME = "259:0"

# One ceiling for every system at a scale: the tightest one all included systems fit under.
DEFAULT_MEM = {"10m": "2G", "100m": "4G"}
# (mode, T): TapeANN anchors. Covering uses J=16, the best-recall setting in their sweep.
ANCHORS = {
    "10m": [("ivf", 40), ("ivf", 140), ("covering", 140)],
    "100m": [("ivf", 80), ("ivf", 140), ("covering", 140)],
}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_rios(io_stat: Path) -> int | None:
    if not io_stat.exists():
        return None
    for line in io_stat.read_text().splitlines():
        if line.startswith(NVME + " "):
            match = re.search(r"rios=(\d+)", line)
            return int(match.group(1)) if match else None
    return None


def run(args: argparse.Namespace) -> None:
    rn = load_module(RUN_NATIVE, "run_native")
    host = load_module(HOST_STATE, "host_state")
    scale = args.scale
    mem = args.mem or DEFAULT_MEM[scale]
    out_root = Path(args.out_root) if args.out_root else REPO / "results/equal_grounds"
    out = out_root / f"{scale}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)
    os.chdir(out)  # indexsearcher appends Recall-result.out to its cwd

    rn.RESULTS, rn.JSONL, rn.WORK = out, out / "rows.jsonl", out / "work"
    for system in ("spann", "tapeann"):
        rn.INDEXES[(system, scale)]["mem"] = mem
    spann = rn.INDEXES[("spann", scale)]
    if args.spann_variant:
        root = SPANN_ROOT / scale / args.spann_variant
        spann.update(
            index=root / "index",
            variant=args.spann_variant,
            posting_page_limit=json.loads((root / "manifest.json").read_text())[
                "posting_page_limit"
            ],
        )
    internals = args.internal or rn.SPANN_INTERNAL

    ctx: dict[str, object] = {}
    counter = itertools.count()
    original_append = rn.append_row
    pending: list[dict] = []
    gate = SsdGate(host, ratio=args.gate_ratio, wait_s=args.wait_s)
    probe_target = Path(spann["index"])  # one fixed file per scale for every system

    def tagged_append(row: dict) -> None:
        row.update(ctx)
        # indexsearcher logs mean latency to 0.1 ms; Recall-result.out (in its cwd) has full
        # precision: "mean std min max" in seconds per batch.
        precise = Path.cwd() / "Recall-result.out"
        if row["system"] == "SPANN" and precise.exists():
            row["lat_mean_ms_precise"] = 1000 * float(precise.read_text().split()[0])
        newest = max(rn.WORK.iterdir(), key=lambda p: p.stat().st_mtime)
        row["iostat_rios"] = parse_rios(newest / "io.stat")
        pending.append(row)

    rn.append_row = tagged_append

    def run_point(kind: str, arg: object, repeat: int) -> None:
        # a fresh cwd per process keeps each Recall-result.out separate
        cwd = out / "cwd" / str(next(counter))
        cwd.mkdir(parents=True)
        os.chdir(cwd)
        if kind == "spann":
            rn.run_spann(scale, int(arg), repeat)  # type: ignore[call-overload]
        else:
            mode, clusters = arg  # type: ignore[misc]
            rn.run_tapeann(scale, 16 if mode == "covering" else 0, clusters, mode, False, repeat)

    def flush(**extra: object) -> None:
        for row in pending:
            row.update(extra)
            original_append(row)
        pending.clear()

    def measure(kind: str, arg: object, repeat: int) -> None:
        """One point: gated pre-probe, run, post-probe; re-run it if the drive dipped."""
        for attempt in range(1 + args.max_retries):
            pre = gate.pre(probe_target)
            ctx.update(
                phase="search", repeat_group=repeat, probe=pre, attempt=attempt,
                time=datetime.now().isoformat(),
            )  # fmt: skip
            run_point(kind, arg, repeat)
            post = gate.post(probe_target)
            suspect = not (pre["gate_ok"] and post["ok"])
            flush(probe_after=post, suspect=suspect)
            if not suspect:
                return
            print(f"  suspect point (pre {pre.get('probe_qd64_kiops')}, post "
                  f"{post.get('probe_qd64_kiops')} vs ref {pre['ref_kiops']}); re-running", flush=True)  # fmt: skip
            time.sleep(args.wait_s)

    def load_baseline(repeat: int) -> None:
        rn.N_QUERIES = 1
        ctx.update(phase="load_baseline", repeat_group=repeat, probe={})
        try:
            run_point("spann", min(internals), repeat)
            run_point("tapeann", ANCHORS[scale][0], repeat)
            flush(suspect=False)
        finally:
            rn.N_QUERIES = 2000

    rn.check_prereqs()
    meta = {
        "scale": scale,
        "mem_cap": mem,
        "spann_variant": spann["variant"],
        "spann_index": str(spann["index"]),
        "internals": internals,
        "anchors": ANCHORS[scale],
        "repeats": args.repeats,
        "n_queries": rn.N_QUERIES,
        "core": rn.CORE,
        "gate": {"ratio": args.gate_ratio, "wait_s": args.wait_s, "max_retries": args.max_retries},
        "started": datetime.now().isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"equal grounds: {meta}", flush=True)

    points = [("spann", i) for i in internals] + [("tapeann", a) for a in ANCHORS[scale]]
    for repeat in range(args.repeats):
        order = points if repeat % 2 == 0 else points[::-1]
        load_baseline(repeat)
        for kind, arg in order:
            measure(kind, arg, repeat)
    print(f"results in {out}", flush=True)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(args: argparse.Namespace) -> None:
    run_dir = Path(args.dir)
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines()]
    meta = json.loads((run_dir / "meta.json").read_text())
    scale = meta["scale"]
    base = {}  # system -> (median rbytes, median rios) of the 1-query load-only runs
    for system in ("SPANN", "TapeANN"):
        loads = [r for r in rows if r["system"] == system and r.get("phase") == "load_baseline"]
        base[system] = (
            median([r["iostat_rbytes"] for r in loads if r["exitcode"] == 0]) or 0,
            median([r["iostat_rios"] for r in loads if r["exitcode"] == 0 and r["iostat_rios"]])
            or 0,
        )
    prior = {}
    if PRIOR.exists():
        for line in PRIOR.read_text().splitlines():
            r = json.loads(line)
            if r["scale"] == scale and r.get("exitcode") == 0 and r.get("recall") is not None:
                prior[(r["system"], r["knobs"])] = r

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("phase") == "search" and not r.get("suspect"):
            groups.setdefault((r["system"], r["knobs"]), []).append(r)
    suspects = sum(1 for r in rows if r.get("phase") == "search" and r.get("suspect"))
    probes = [
        r["probe"]["probe_qd64_kiops"]
        for r in rows
        if r.get("phase") == "search" and r.get("probe", {}).get("probe_qd64_kiops")
    ]
    lines = [
        f"# Equal grounds, BIGANN-{scale.upper()}: 1 thread on core {meta['core']}, "
        f"MemoryMax {meta['mem_cap']}, cold cache, {meta['n_queries']} queries",
        f"SPANN index: {meta['spann_variant']}. Medians over {meta['repeats']} repeats; "
        f"{suspects} attempts were discarded and re-run because the SSD probe dipped. "
        "rd MB/q and rd ops/q are net of index load (1-query baseline subtracted).",
        f"SSD probe QD64 (kIOPS) over the session: min {min(probes):.0f}, "
        f"median {statistics.median(probes):.0f}, max {max(probes):.0f}"
        if probes
        else "no probe",
        "",
        "| system | knobs | n | recall@10 | mean ms | p99 ms | peak MB | anon MB | rd MB/q "
        "| rd ops/q | earlier mean ms (other session) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (system, knobs), group in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], median([r.get("lat_mean_ms") or 0 for r in kv[1]]) or 0),
    ):
        ok = [r for r in group if r["exitcode"] == 0 and r.get("recall") is not None]
        for r in ok:  # SPANN: full-precision mean from Recall-result.out when present
            r["lat_mean_ms"] = r.get("lat_mean_ms_precise", r["lat_mean_ms"])
        if not ok:
            lines.append(f"| {system} | {knobs} | 0/{len(group)} | FAILED | | | | | | | |")
            continue
        n = ok[0]["n_queries"]
        rbytes0, rios0 = base[system]
        old = prior.get((system, knobs))
        p99 = median([r["lat_p99_ms"] for r in ok if r.get("lat_p99_ms") is not None])
        ios = median([r["iostat_rios"] for r in ok if r.get("iostat_rios")])
        cells = [
            system, knobs, f"{len(ok)}/{len(group)}",
            f"{median([r['recall'] for r in ok]):.4f}",
            f"{median([r['lat_mean_ms'] for r in ok]):.3f}",
            f"{p99:.2f}" if p99 is not None else "-",
            f"{median([r['dram_mb'] for r in ok]):.0f}",
            f"{median([r['anon_mb'] for r in ok]):.0f}",
            f"{(median([r['iostat_rbytes'] for r in ok]) - rbytes0) / n / 2**20:.2f}",
            f"{(ios - rios0) / n:.1f}" if ios else "-",
            f"{old['lat_mean_ms']:.2f}" if old else "-",
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scale", required=True, choices=["10m", "100m"])
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--mem", help="MemoryMax for every system (default 2G at 10m, 4G at 100m)")
    r.add_argument(
        "--out-root", help="results parent dir (default ann-suite/results/equal_grounds)"
    )
    r.add_argument("--gate-ratio", type=float, default=0.7, help="min probe/reference to accept")
    r.add_argument("--wait-s", type=float, default=20.0, help="wait before re-probing a dip")
    r.add_argument("--max-retries", type=int, default=2, help="re-runs of a suspect point")
    r.add_argument("--internal", type=int, nargs="+", help="SPANN SearchInternalResultNum values")
    r.add_argument("--spann-variant", help="SPANN build dir name under indexes/<scale>/")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    args = parser.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
