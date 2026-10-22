"""Config-driven search-hyperparameter sweep over the natively built SPANN indexes.

Runs SPTAG's `indexsearcher` under TapeANN's native benchmark conditions (its run_native.run_scope,
imported unchanged): a user systemd scope with a cpuset and MemoryMax, page cache dropped before every
process, recall@10 recomputed from the returned ids against the native workflow's exact truth.
A config file defines the indexes, the memory ceilings and the stages; a stage is a grid over the
search-time knobs:

  irn       BuildSSDIndex.SearchInternalResultNum   head candidates -> postings read per query
  ppl       BuildSSDIndex.SearchPostingPageLimit    pages read per posting ("build" = the index's own)
  mdr       BuildSSDIndex.MaxDistRatio              drop postings whose head is > mdr x the nearest
                                                    head's (squared) distance: query-aware pruning
  maxcheck  BuildHead.MaxCheck                      head-graph search budget
  threads   -t                                      search threads (cpuset from cpus_by_threads)
  mem       MemoryMax                               DRAM ceiling

Each point is bracketed by SSD probes (ssd_gate): a point whose drive state dipped is re-run, and the
attempts are all kept (`suspect`). Per index and thread count a 1-query run measures the cost of loading
the index; the summary subtracts it from read bytes/ops and CPU time.

    /home/isfcr/tapeann/engine/.venv/bin/python tools/harness_compare/spann_sweep.py run \\
        --config configs/spann_sweep.json [--only STAGE ...] [--resume RESULTS_DIR]
    /home/isfcr/tapeann/engine/.venv/bin/python tools/harness_compare/spann_sweep.py summarize RESULTS_DIR
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import itertools
import json
import os
import platform
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from equal_grounds import load_module, parse_rios
from spann_loader import IRN_CAPACITY, ensure_irn_capacity, write_loader
from ssd_gate import SsdGate

REPO = Path(__file__).resolve().parents[2]
PARAM_KEYS = ("irn", "ppl", "mdr", "maxcheck", "threads", "mem")


@dataclass(frozen=True)
class IndexInfo:
    name: str  # "<scale>/<build dir>"
    scale: str
    root: Path
    ppl: int
    replica: int
    heads: float

    @property
    def index_dir(self) -> Path:
        return self.root / "index"


@dataclass(frozen=True)
class Task:
    stage: str
    index: str
    params: dict[str, Any]
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.stage}|{self.index}|{json.dumps(self.params, sort_keys=True)}|r{self.repeat}"


def discover(spann_root: Path) -> dict[str, IndexInfo]:
    found = {}
    for manifest in sorted(spann_root.glob("*/*/manifest.json")):
        root = manifest.parent
        data = json.loads(manifest.read_text())
        if root.is_symlink() or data.get("stage") != "postings" or data.get("returncode") != 0:
            continue
        name = f"{root.parent.name}/{root.name}"
        found[name] = IndexInfo(
            name, data["scale"], root, data["posting_page_limit"], data["replica"],
            data["head_fraction_actual"],
        )  # fmt: skip
    return found


def select(selector: str | list[str], infos: dict[str, IndexInfo]) -> list[IndexInfo]:
    if selector == "all":
        return list(infos.values())
    patterns = [selector] if isinstance(selector, str) else selector
    chosen = [i for i in infos.values() if any(fnmatch.fnmatch(i.name, p) for p in patterns)]
    if not chosen:
        raise SystemExit(f"no index matches {selector!r}; have {sorted(infos)}")
    return chosen


def expand(stage: dict, cfg: dict, infos: dict[str, IndexInfo]) -> list[Task]:
    grid = {**{"irn": [64]}, **stage["grid"]}
    unknown = set(grid) - set(PARAM_KEYS)
    if unknown:
        raise SystemExit(f"stage {stage['name']}: unknown knobs {sorted(unknown)}")
    tasks = []
    for info in select(stage["indexes"], infos):
        repeats = stage.get("repeats_by_scale", {}).get(
            info.scale, stage.get("repeats", cfg["repeats"])
        )
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values, strict=True))
            params.setdefault("threads", 1)
            params.setdefault("ppl", "build")
            params.setdefault("mdr", None)  # None = not passed: the index's default (disabled)
            params.setdefault("maxcheck", None)  # None = the head index's own (4096)
            params["ppl"] = info.ppl if params["ppl"] == "build" else params["ppl"]
            params["mem"] = params.get("mem") or cfg["mem_by_index"].get(
                info.name, cfg["mem_by_scale"][info.scale]
            )
            tasks.extend(Task(stage["name"], info.name, params, r) for r in range(repeats))
    return tasks


def command(cfg: dict, info: IndexInfo, params: dict, n_queries: int, queries: Path, out: Path):
    timeout = cfg["timeout_s"]
    timeout = timeout[info.scale] if isinstance(timeout, dict) else timeout
    cmd = [
        "timeout", str(timeout), "/usr/bin/time", "-v", "-o", str(out / "time.txt"),
        cfg["paths"]["searcher"], "-i", str(queries), "-x", str(info.index_dir),
        "-o", str(out / "search-results.txt"), "-d", "128", "-v", "UInt8", "-f", "DEFAULT",
        "-k", "10", "-b", str(n_queries), "-of", "0", "-t", str(params["threads"]),
        f"BuildSSDIndex.SearchInternalResultNum={params['irn']}",
        f"BuildSSDIndex.SearchPostingPageLimit={params['ppl']}",
    ]  # fmt: skip
    if params.get("mdr") is not None:
        cmd.append(f"BuildSSDIndex.MaxDistRatio={params['mdr']}")
    if params.get("maxcheck") is not None:
        cmd.append(f"BuildHead.MaxCheck={params['maxcheck']}")
    return cmd


def parse_time(path: Path) -> dict[str, float | None]:
    text = path.read_text(errors="replace") if path.exists() else ""

    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    rss = grab(r"Maximum resident set size \(kbytes\): (\d+)")
    return {
        "cpu_user_s": grab(r"User time \(seconds\): ([\d.]+)"),
        "cpu_sys_s": grab(r"System time \(seconds\): ([\d.]+)"),
        "maxrss_mb": rss / 1024 if rss else None,
    }


def parse_search(rn: Any, text: str, out: Path, n: int) -> dict[str, Any]:
    """Latency/QPS from indexsearcher's output, recall ids from its result file."""
    batches = [m for m in (rn.BATCH_LINE.search(line) for line in text.splitlines()) if m]
    stats: dict[str, Any] = {}
    if batches:
        sizes = [int(m.group(2)) - int(m.group(1)) for m in batches]
        total = sum(sizes)
        stats["qps"] = total / sum(
            s / float(m.group(6)) for s, m in zip(sizes, batches, strict=True)
        )
        for name, group in (("lat_p99_ms", 4), ("lat_p95_ms", 5)):
            stats[name] = (
                1000
                * sum(s * float(m.group(group)) for s, m in zip(sizes, batches, strict=True))
                / total
            )
    precise = out / "Recall-result.out"  # full precision: mean std min max (s) per batch
    if precise.exists() and precise.read_text().split():
        mean, std, low, high = (1000 * float(v) for v in precise.read_text().split()[:4])
        stats.update(lat_mean_ms=mean, lat_std_ms=std, lat_min_ms=low, lat_max_ms=high)
    eff = re.search(r"Load index with posting page limit:(\d+)", text)
    stats["eff_ppl"] = int(eff.group(1)) if eff else None
    ids = np.full((n, 10), -1, dtype=np.int64)
    result = out / "search-results.txt"
    if result.exists():
        for line in result.read_text(errors="replace").splitlines():
            if ":" not in line:
                continue
            qid, values = line.split(":", 1)
            if int(qid) >= n:
                continue
            for col, value in enumerate(values.split("|")[:10]):
                if "@" in value and not value.endswith("NULL"):
                    ids[int(qid), col] = int(value.split("@", 1)[1])
        result.unlink()  # 1.7 MB per point; recall is computed and stored
    stats["ids"] = ids
    return stats


class Sweep:
    def __init__(self, cfg: dict, out: Path) -> None:
        self.cfg, self.out = cfg, out
        self.rn = load_module(Path(cfg["paths"]["run_native"]), "run_native")
        host = load_module(REPO / cfg["paths"]["host_state"], "host_state")
        g = cfg["gate"]
        self.gate = SsdGate(host, g["ratio"], g["wait_s"], g["max_waits"], g["fio_seconds"])
        self.max_retries = g["max_retries"]
        taken = (
            [int(d.name[:5]) for d in (out / "work").glob("*") if d.name[:5].isdigit()]
            if (out / "work").exists()
            else []
        )
        self.counter = itertools.count(
            max(taken, default=-1) + 1
        )  # resume without reusing work dirs
        self.truth: dict[str, np.ndarray] = {}
        self.rows_path = out / "rows.jsonl"
        (out / "work").mkdir(exist_ok=True)

    def truth_for(self, scale: str) -> np.ndarray:
        if scale not in self.truth:
            raw = np.fromfile(Path(self.cfg["paths"]["truth_dir"]) / f"truth{scale}.bin", np.int32)
            n, k = int(raw[0]), int(raw[1])
            self.truth[scale] = raw[2 : 2 + n * k].reshape(n, k)
        return self.truth[scale]

    def queries_file(self, n: int) -> Path:
        """The first n BIGANN queries in SPTAG's DEFAULT format (= u8bin: int32 n, int32 dim, rows)."""
        full = Path(self.cfg["paths"]["queries"])
        if n == int(np.fromfile(full, np.int32, count=1)[0]):
            return full
        small = self.out / f"queries_{n}.bin"
        if not small.exists():
            rows = np.fromfile(full, np.uint8, offset=8).reshape(-1, 128)[:n]
            with small.open("wb") as f:
                np.asarray([n, 128], np.int32).tofile(f)
                rows.tofile(f)
        return small

    def execute(self, task: Task, info: IndexInfo, phase: str, n: int, attempt: int) -> dict:
        p = task.params
        out = self.out / "work" / f"{next(self.counter):05d}_{info.scale}_{info.root.name}_{phase}"
        out.mkdir(parents=True)
        os.chdir(out)  # indexsearcher appends Recall-result.out to its cwd
        cmd = command(self.cfg, info, p, n, self.queries_file(n), out)
        cpus = self.cfg["cpus_by_threads"][str(p["threads"])]
        started = datetime.now().isoformat()
        run = self.rn.run_scope(cmd, cpus=cpus, mem=p["mem"], out_dir=out)
        row: dict[str, Any] = {
            "stage": task.stage, "index": info.name, "scale": info.scale, "build_ppl": info.ppl,
            "build_replica": info.replica, "heads_frac": info.heads, "phase": phase,
            "repeat": task.repeat, "attempt": attempt, **p, "cpus": cpus, "n_queries": n,
            "exitcode": run["exitcode"], "wall_s": round(run["wall_s"], 3),
            "dram_mb": round(run["dram_mb"], 1), "anon_mb": round(run["anon_mb"], 1),
            "rbytes": run["rbytes"], "rios": parse_rios(out / "io.stat"), "time": started,
            "governor": self.rn.governor(), "nvme_temp_c": self.rn.nvme_temp_c(),
            "load1": os.getloadavg()[0], "work": out.name,
        }  # fmt: skip
        row.update(parse_time(out / "time.txt"))
        if run["exitcode"] != 0:
            row["error"] = (run["stderr"] or run["stdout"] or run["runner_stderr"])[-400:]
            return row
        stats = parse_search(self.rn, run["stdout"] + "\n" + run["stderr"], out, n)
        ids = stats.pop("ids")
        empty = float((ids < 0).all(axis=1).mean())
        if empty > 0.01:  # indexsearcher exits 0 with empty results when it cannot allocate
            row["error"] = f"empty result rows: {empty:.0%}"
            row["exitcode"] = 1
        else:
            row["recall"] = self.rn.recall_at_10(ids, self.truth_for(info.scale)[:n])
        row.update(stats)
        row.setdefault("lat_mean_ms", None)
        return row

    def append(self, row: dict) -> None:
        with self.rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        flags = " SUSPECT" if row.get("suspect") else ""
        print(
            f"  {row['stage'][:14]:14} {row['index']:38} {row['phase'][:6]:6} irn={row['irn']:<4} "
            f"ppl={row['ppl']} mdr={row.get('mdr')} mc={row.get('maxcheck')} t={row['threads']} "
            f"mem={row['mem']} recall={row.get('recall')} lat={row.get('lat_mean_ms')} "
            f"exit={row['exitcode']}{flags}",
            flush=True,
        )

    def measure(self, task: Task, info: IndexInfo) -> None:
        n = self.cfg["n_queries"]
        for attempt in range(1 + self.max_retries):
            pre = self.gate.pre(info.index_dir)
            row = self.execute(task, info, "search", n, attempt)
            post = self.gate.post(info.index_dir)
            dipped = not (pre["gate_ok"] and post["ok"]) and row["exitcode"] == 0
            # the last attempt is kept either way, flagged `degraded`, so a point is never lost
            suspect = dipped and attempt < self.max_retries
            row.update(
                probe=pre, probe_after=post, suspect=suspect, degraded=dipped and not suspect
            )
            row["task"] = task.key
            self.append(row)
            if not suspect:
                return
            print(f"  suspect: probe {pre.get('probe_qd64_kiops')}/{post.get('probe_qd64_kiops')} "
                  f"kIOPS vs ref {pre['ref_kiops']}; re-running", flush=True)  # fmt: skip
            time.sleep(self.gate.wait_s)

    def load_baseline(self, stage: str, info: IndexInfo, threads: int, mem: str) -> None:
        params = {"irn": 8, "ppl": info.ppl, "threads": threads, "mem": mem}
        row = self.execute(Task(stage, info.name, params, 0), info, "load_baseline", 1, 0)
        row.update(suspect=False, task=f"baseline|{info.name}|{threads}")
        self.append(row)


def run(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text())
    infos = discover(Path(cfg["paths"]["spann_root"]))
    stages = [s for s in cfg["stages"] if not args.only or s["name"] in args.only]
    if args.resume:
        out = Path(args.resume).resolve()
    else:
        out = (
            Path(cfg["paths"]["results_root"]) / f"{cfg['name']}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        )
        out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    sweep = Sweep(cfg, out)
    sweep.rn.check_prereqs()
    done: set[str] = set()
    if (out / "rows.jsonl").exists():
        for line in (out / "rows.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("task") and not r.get("suspect"):
                done.add(r["task"])
    if args.redo:
        done -= set(json.loads(Path(args.redo).read_text()))
    (out / ("meta_resume.json" if args.resume else "meta.json")).write_text(json.dumps({
        "started": datetime.now().isoformat(), "host": platform.node(), "kernel": platform.release(),
        "governor": sweep.rn.governor(), "stages": [s["name"] for s in stages], "resumed": bool(args.resume),
    }, indent=2))  # fmt: skip
    print(f"sweep -> {out}", flush=True)

    rng = random.Random(cfg["seed"])
    for stage in stages:
        tasks = [t for t in expand(stage, cfg, infos) if t.key not in done]
        rng.shuffle(tasks)  # interleave knobs, indexes and repeats so SSD drift is not aliased
        print(f"== stage {stage['name']}: {len(tasks)} points", flush=True)
        for t in tasks:
            write_loader(infos[t.index].root)  # no-op when a loader exists
            if ensure_irn_capacity(
                infos[t.index].root, cfg.get("loader_irn_capacity", IRN_CAPACITY)
            ):
                print(
                    f"  raised AIO capacity in {t.index}/index/indexloader.ini (original kept as .bak)"
                )
        baselines = {(t.index, t.params["threads"]) for t in tasks}
        for index, threads in sorted(baselines):
            key = f"baseline|{index}|{threads}"
            if key not in done:
                info = infos[index]
                mem = cfg["mem_by_index"].get(index, cfg["mem_by_scale"][info.scale])
                sweep.load_baseline(stage["name"], info, threads, mem)
                done.add(key)
        for t in tasks:
            sweep.measure(t, infos[t.index])
    print(f"results in {out}", flush=True)


# ---------------------------------------------------------------------------------------------- summary
def med(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def interpolate(points: list[tuple[float, float]], target: float) -> float | None:
    """Metric at a recall target by linear interpolation between neighbouring sweep points."""
    points = sorted(points)
    if not points or points[-1][0] < target or points[0][0] > target:
        return None
    for (r0, m0), (r1, m1) in itertools.pairwise(points):
        if r0 <= target <= r1:
            return m0 if r1 == r0 else m0 + (m1 - m0) * (target - r0) / (r1 - r0)
    return points[-1][1]


def summarize(args: argparse.Namespace) -> None:
    run_dir = Path(args.dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines()]
    base: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows:
        if r["phase"] == "load_baseline" and r["exitcode"] == 0:
            cpu = (r.get("cpu_user_s") or 0) + (r.get("cpu_sys_s") or 0)
            base[(r["index"], r["threads"])] = {
                "rbytes": r["rbytes"],
                "rios": r["rios"] or 0,
                "cpu": cpu,
            }
    search = [r for r in rows if r["phase"] == "search"]
    latest: dict[str, dict] = {}
    for r in search:
        if not r.get("suspect"):
            latest[r["task"]] = r  # a --redo re-run replaces the original point
    clean = list(latest.values())
    discarded = sum(1 for r in search if r.get("suspect"))
    stalls = {r["task"]: r for r in [*stall_candidates(clean, base), *monotone_candidates(clean)]}
    stalls = list(stalls.values())
    (run_dir / "stalls.json").write_text(json.dumps([r["task"] for r in stalls], indent=1))
    probes = [
        r["probe"]["probe_qd64_kiops"] for r in search if r.get("probe", {}).get("probe_qd64_kiops")
    ]

    groups: dict[tuple, list[dict]] = {}
    for r in clean:
        groups.setdefault((r["stage"], r["index"], *(r[k] for k in PARAM_KEYS)), []).append(r)
    points = []
    for (stage, index, *vals), g in groups.items():
        ok = [r for r in g if r["exitcode"] == 0 and r.get("recall") is not None]
        row = dict(zip(("stage", "index", *PARAM_KEYS), (stage, index, *vals), strict=True))
        failures = [r for r in g if r["exitcode"] != 0 or r.get("recall") is None]
        row.update(
            n=len(ok), failed=len(failures), fail_exit=failures[0]["exitcode"] if failures else None
        )
        if ok:
            b = base.get((index, row["threads"]), {"rbytes": 0, "rios": 0, "cpu": 0})
            nq = ok[0]["n_queries"]
            cpu = [(r.get("cpu_user_s") or 0) + (r.get("cpu_sys_s") or 0) - b["cpu"] for r in ok]
            lats = [r["lat_mean_ms"] for r in ok]
            spread = (
                (max(lats) - min(lats)) / med(lats) if len(ok) > 1 and None not in lats else None
            )
            row.update(
                heads=ok[0]["heads_frac"], degraded=sum(1 for r in ok if r.get("degraded")),
                recall=med([r["recall"] for r in ok]), lat_mean_ms=med([r["lat_mean_ms"] for r in ok]),
                lat_p99_ms=med([r.get("lat_p99_ms") for r in ok]), qps=med([r.get("qps") for r in ok]),
                cpu_ms_q=1000 * statistics.median(cpu) / nq,
                peak_mb=med([r["dram_mb"] for r in ok]), anon_mb=med([r["anon_mb"] for r in ok]),
                rss_mb=med([r["maxrss_mb"] for r in ok]), eff_ppl=ok[0].get("eff_ppl"),
                ops_q=(statistics.median([r["rios"] or 0 for r in ok]) - b["rios"]) / nq,
                mb_q=(statistics.median([r["rbytes"] for r in ok]) - b["rbytes"]) / nq / 2**20,
                lat_spread=spread,
            )  # fmt: skip
        points.append(row)
    cols = ["stage", "index", *PARAM_KEYS, "n", "failed", "fail_exit", "heads", "degraded", "recall", "lat_mean_ms", "lat_p99_ms", "qps",
            "cpu_ms_q", "peak_mb", "anon_mb", "rss_mb", "eff_ppl", "ops_q", "mb_q", "lat_spread"]  # fmt: skip
    with (run_dir / "points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(points)

    lines = [
        f"# SPANN search sweep: {cfg['name']}",
        f"{len(clean)} measured points kept, {discarded} suspect attempts discarded and re-run "
        f"(SSD probe dip); {sum(1 for r in clean if r.get('degraded'))} kept after exhausting "
        f"retries and marked * in n. Cpuset per `cpus_by_threads`, cold cache, {cfg['n_queries']} "
        "queries, recall@10 vs exact truth. Medians over repeats.",
        f"{len(stalls)} point(s) look like mid-run stalls (see the end of this file; stalls.json).",
        "ops/q, MB/q and cpu ms/q are net of loading the index (1-query baseline subtracted).",
        f"SSD probe QD64 kIOPS: min {min(probes):.0f}, median {statistics.median(probes):.0f}, "
        f"max {max(probes):.0f}" if probes else "no probes",
        "",
    ]  # fmt: skip
    for stage in cfg["stages"]:
        pts = [p for p in points if p["stage"] == stage["name"]]
        if not pts:
            continue
        varied = [k for k in PARAM_KEYS if len({p[k] for p in pts}) > 1]
        lines += [f"## {stage['name']}", stage.get("description", ""), ""]
        head = ["index", *varied, "n", "recall", "mean ms", "p99 ms", "qps", "cpu ms/q", "ops/q", "MB/q",
                "peak MB", "anon MB", "spread"]  # fmt: skip
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for p in sorted(
            pts,
            key=lambda p: (p["index"], *(str(p[k]) if k == "mem" else p[k] or 0 for k in varied)),
        ):
            if "recall" not in p:
                lines.append(f"| {p['index']} | " + " | ".join(str(p[k]) for k in varied)
                             + f" | FAILED x{p['failed']} (exit {p['fail_exit']}: 124=timeout, "
                             "134/139=crash, negative or 137=killed by the memory cap) |")  # fmt: skip
                continue
            lines.append("| " + " | ".join([
                p["index"], *(str(p[k]) for k in varied), f"{p['n']}{'*' if p['degraded'] else ''}",
                f"{p['recall']:.4f}",
                fmt(p["lat_mean_ms"], ".3f"), fmt(p["lat_p99_ms"], ".2f"), fmt(p["qps"], ".0f"),
                fmt(p["cpu_ms_q"], ".3f"), fmt(p["ops_q"], ".1f"), fmt(p["mb_q"], ".2f"),
                fmt(p["peak_mb"], ".0f"), fmt(p["anon_mb"], ".0f"), fmt(p.get("lat_spread"), ".0%"),
            ]) + " |")  # fmt: skip
        lines.append("")
        if set(stage["grid"]) == {"irn"}:  # a pure IRN sweep per index
            lines += cost_at_recall(pts)
    if stalls:
        lines += ["## Possible mid-run stalls (mean latency far above the CPU cost seen at the same "
                  "index/threads/mem, with an extreme slowest query); re-run with `run --resume DIR --redo "
                  "stalls.json`:", ""]  # fmt: skip
        lines += [f"- {r['stage']} {r['index']} irn={r['irn']} mdr={r['mdr']} maxcheck={r['maxcheck']} "
                  f"threads={r['threads']} mem={r['mem']}: mean {r['lat_mean_ms']:.2f} ms, "
                  f"slowest query {r['lat_max_ms']:.1f} ms" for r in stalls]  # fmt: skip
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def stall_candidates(rows: list[dict], base: dict) -> list[dict]:
    """Points whose latency is far above the index's usual latency-per-CPU-ms AND that had one
    extreme query: the signature of a drive stall during the run, which pre/post probes can miss.
    Genuinely slow regimes (memory-starved caps, SSD-saturating thread counts) are compared only
    with points of the same index/threads/mem, so they are not flagged."""
    groups: dict[tuple, list[tuple[float, dict]]] = {}
    for r in rows:
        if r["phase"] != "search" or r["exitcode"] != 0 or not r.get("lat_mean_ms"):
            continue
        b = base.get((r["index"], r["threads"]), {"cpu": 0})
        cpu = (
            ((r.get("cpu_user_s") or 0) + (r.get("cpu_sys_s") or 0) - b["cpu"])
            * 1000
            / r["n_queries"]
        )
        if cpu > 0:
            groups.setdefault((r["index"], r["threads"], r["mem"]), []).append(
                (r["lat_mean_ms"] / cpu, r)
            )
    flagged = []
    for members in groups.values():
        if len(members) < 4:
            continue
        usual = statistics.median(ratio for ratio, _ in members)
        flagged += [
            r for ratio, r in members
            if ratio > 1.8 * usual and r["lat_max_ms"] > 8 * r["lat_mean_ms"]
        ]  # fmt: skip
    return flagged


def monotone_candidates(rows: list[dict]) -> list[dict]:
    """Points that are >15% slower than a point with a looser setting of one knob, all else equal.

    Loosening IRN, MaxDistRatio or head MaxCheck only adds work, so latency cannot fall; a point that
    violates this was hit by a stall that the pre/post probes missed."""
    looser = {
        "irn": lambda v: v,
        "mdr": lambda v: float("inf") if v is None else v,  # None = pruning off = loosest
        "maxcheck": lambda v: 4096 if v is None else v,  # None = the head index's own 4096
    }
    flagged: dict[str, dict] = {}
    ok = [r for r in rows if r["phase"] == "search" and r["exitcode"] == 0 and r.get("lat_mean_ms")]
    for knob, order in looser.items():
        groups: dict[tuple, list[dict]] = {}
        for r in ok:
            others = tuple(r.get(k) for k in (*PARAM_KEYS, "stage", "index", "repeat") if k != knob)
            groups.setdefault(others, []).append(r)
        for members in groups.values():
            members.sort(key=lambda r: order(r[knob]))
            for i, a in enumerate(members):
                if any(a["lat_mean_ms"] > 1.15 * b["lat_mean_ms"] for b in members[i + 1 :]):
                    flagged[a["task"]] = a
    return list(flagged.values())


def cost_at_recall(pts: list[dict]) -> list[str]:
    """Per index: the cost of reaching a recall target, interpolated along the IRN sweep."""
    targets = (0.90, 0.95, 0.97)
    lines = ["### Cost to reach a recall target (interpolated along the sweep)", "",
             "| index | heads | " + " | ".join(f"R>={t}: IRN / ms / ops / MB" for t in targets) + " |",
             "|---|---|" + "---|" * len(targets)]  # fmt: skip
    for index in sorted({p["index"] for p in pts}):
        mine = [p for p in pts if p["index"] == index and "recall" in p]
        if not mine:
            continue
        cells = []
        for t in targets:
            vals = [
                interpolate([(p["recall"], p[m]) for p in mine if p[m] is not None], t)
                for m in ("irn", "lat_mean_ms", "ops_q", "mb_q")
            ]
            cells.append("n/a" if vals[0] is None else
                         f"{vals[0]:.0f} / {vals[1]:.2f} / {vals[2]:.0f} / {vals[3]:.2f}")  # fmt: skip
        lines.append(f"| {index} | {mine[0]['heads']:.1%} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--only", nargs="+", help="run only these stage names")
    r.add_argument("--resume", help="continue an interrupted run directory")
    r.add_argument("--redo", help="JSON list of task keys (e.g. stalls.json) to re-run")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    args = parser.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
