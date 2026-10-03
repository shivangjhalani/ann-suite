"""Evaluate evolved disk-ANN programs in ann-suite and score them against baselines.

Subcommands (run from the ann-suite root; see docs/EVOLVE.md):

  floors      measure each image's runtime memory floor (idle Python + imports) and
              the evolved harness floor (a null program through the full pipeline)
  baselines   run configs/evolve/baselines_*.yaml and write the frontier point set
  candidate   build + search one candidate program, score it, print one JSON line
  add-reference  measure a reference program (a known design on the harness) and
              add its points to the frontier as a named baseline

`candidate` is what OpenEvolve's evaluator calls (over SSH). It holds a file lock
for the whole evaluation: every search point drops the OS page cache, so two
evaluations at once would corrupt each other's I/O counts.

Validation: a candidate whose score would beat the record (best validated score
so far, results/evolve/record_<name>.json; at least 0) is measured again on the
same queries and on hidden queries it never saw (stage `hidden`, same base and
build); its score becomes the minimum of the three. Delete the record file to
start a new run from scratch.

Build caching: indices are cached under <index_dir>/cache/<stage>/<hash>, where
the hash covers the program source minus `class Searcher` and SEARCH_POINTS (plus
the stage dataset). A candidate that only changes search code reuses the index,
searched with its own program.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier import (
    Cells,
    Point,
    load_points,
    pareto,
    point_from_result,
    recall_at_k,
    save_points,
    score,
)  # noqa: E402,I001

from ann_suite.core.schemas import BenchmarkConfig  # noqa: E402
from ann_suite.evaluator import BenchmarkEvaluator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SUDO_FILE = Path.home() / ".config" / "ann-suite" / "sudo_password"
FAIL = -10.0


def _cfg(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def _ensure_sudo_env() -> None:
    if not os.environ.get("ANN_SUITE_SUDO_PASSWORD") and SUDO_FILE.exists():
        os.environ["ANN_SUITE_SUDO_PASSWORD"] = SUDO_FILE.read_text().strip()


def _quiet_logs() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def _run_suite(config: dict[str, Any]) -> list[Any]:
    bc = BenchmarkConfig.model_validate(config)
    ev = BenchmarkEvaluator(bc)
    try:
        return list(ev.run())
    finally:
        ev.cleanup()


def _floors(cfg: dict[str, Any]) -> dict[str, float]:
    p = REPO / cfg["floors"]
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------- floors

FLOOR_SNIPPET = (
    "import time; {imports}; time.sleep(2); "
    "print([int(l.split()[1]) for l in open('/sys/fs/cgroup/memory.stat') "
    "if l.startswith('anon ')][0])"
)


def cmd_floors(cfg: dict[str, Any]) -> None:
    floors = _floors(cfg)
    for image, imports in cfg["floor_images"].items():
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
            print(f"{image}: not built, skipped", file=sys.stderr)
            continue
        out = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python3",
                image,
                "-c",
                FLOOR_SNIPPET.format(imports=imports),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        floors[image] = int(out.stdout.strip().splitlines()[-1]) / 2**20
        print(f"{image}: {floors[image]:.1f} MB", file=sys.stderr)
    null = REPO / "tools" / "evolve" / "null_program.py"
    res = _evaluate(cfg, null, "null-floor", stages=["sanity"], use_cache=False, score_it=False)
    anon = [p["peak_anon_mb"] for p in res["stages"]["sanity"]["points"]]
    floors[cfg["image"]] = float(max(anon))
    print(f"{cfg['image']} (null program): {floors[cfg['image']]:.1f} MB", file=sys.stderr)
    out_path = REPO / cfg["floors"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(floors, indent=1))
    print(json.dumps(floors))


# ------------------------------------------------------------------ baselines


def _merge_frontier(out: Path, points: list[Point], names: set[str]) -> list[Point]:
    """Merge points into the frontier file under a lock (configs may run in
    parallel), replacing earlier points of the same configuration name."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ".frontier.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = (
            [p for p in load_points(out) if p.label.split(":")[0].split("@")[0] not in names]
            if out.exists()
            else []
        )
        allp = old + points
        save_points(allp, out)
    return allp


def cmd_baselines(cfg: dict[str, Any], baseline_config: Path) -> None:
    _ensure_sudo_env()
    floors = _floors(cfg)
    bcfg = yaml.safe_load(baseline_config.read_text())
    systems = {a["name"]: a.pop("x-system", a["name"]) for a in bcfg["algorithms"]}
    images = {a["name"]: a["docker_image"] for a in bcfg["algorithms"]}
    for k in [k for k in bcfg if k.startswith("x-")]:
        bcfg.pop(k)
    prebuilt = {
        a["name"]: Path(a["build"]["prebuilt_path"])
        for a in bcfg["algorithms"]
        if a.get("build", {}).get("prebuilt_path")
    }
    results = _run_suite(bcfg)
    points = []
    for r in results:
        if not (r.search_result and r.search_result.success) or r.recall is None:
            continue
        base = r.algorithm.split("@")[0]
        if not r.index_size_bytes and base in prebuilt:
            d = prebuilt[base]
            d = d if d.is_absolute() else Path(bcfg["index_dir"]) / d
            r.index_size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        floor = floors.get(images.get(base, ""), 0.0)
        points.append(point_from_result(r, systems.get(base, base), floor))
    out = REPO / cfg["frontier"]
    out.parent.mkdir(parents=True, exist_ok=True)
    allp = _merge_frontier(out, points, {r.algorithm.split("@")[0] for r in results})
    front = pareto(allp)
    print(
        json.dumps(
            {"points": len(allp), "pareto": len(front), "systems": sorted({p.system for p in allp})}
        )
    )


# ------------------------------------------------------------------ candidate


def _build_hash(source: str, stage: str, dataset: str) -> str:
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Searcher":
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SEARCH_POINTS" for t in node.targets
        ):
            continue
        keep.append(ast.dump(node))
    h = hashlib.sha256(("\n".join(keep) + stage + dataset).encode()).hexdigest()
    return h[:20]


def _search_points(source: str) -> list[dict[str, Any]]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SEARCH_POINTS" for t in node.targets
        ):
            pts = ast.literal_eval(node.value)
            if isinstance(pts, list) and all(isinstance(p, dict) for p in pts):
                return pts
    raise ValueError("program must define SEARCH_POINTS as a literal list of dicts")


def _gc_cache(cache_root: Path, max_gb: float) -> None:
    entries = [d for d in cache_root.glob("*/*") if d.is_dir()]
    sized = []
    for d in entries:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        sized.append((d.stat().st_atime, size, d))
    total = sum(s for _, s, _ in sized)
    for _, size, d in sorted(sized):
        if total <= max_gb * 1e9:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size


def _stage_config(
    cfg: dict[str, Any],
    stage: str,
    cand: str,
    prog_container_path: str,
    n_points: int,
    prebuilt: str | None,
    run_tag: str,
) -> dict[str, Any]:
    st = cfg["stages"][stage]
    ds = st["dataset"]
    build: dict[str, Any] = {
        "timeout_seconds": st["build_timeout_s"],
        "memory_limit": "none",
        "args": {"program": prog_container_path, "threads": cfg["build_threads"]},
    }
    if prebuilt:
        build = {
            "prebuilt_path": prebuilt,
            "required_files": ["evolved_meta.json"],
            "timeout_seconds": 60,
        }
    return {
        "name": f"{cfg['name']}-{cand}-{stage}",
        "data_dir": cfg["data_dir"],
        "index_dir": cfg["index_dir"],
        "results_dir": str(REPO / cfg["results_dir"] / "runs"),
        "monitor_interval_ms": cfg["monitor_interval_ms"],
        "algorithms": [
            {
                "name": f"Evolved-{cand}",
                "docker_image": cfg["image"],
                "algorithm_type": "disk",
                # Candidate code runs sandboxed: the runner starts as root, drops the
                # candidate to `nobody`, and the container has no network.
                "container_user": "root",
                "network": "none",
                "datasets": [ds],
                "memory_limit": cfg["memory_limit"],
                "build": build,
                "search": {
                    "timeout_seconds": st["search_timeout_s"],
                    "k": cfg["k"],
                    "args": {
                        "threads": cfg["search_threads"],
                        "program": prog_container_path,
                        "run_tag": run_tag,
                    },
                    "sweep": [{"point": i} for i in range(n_points)],
                },
            }
        ],
        "datasets": [
            {
                "name": ds,
                "base_path": f"{ds}/base.npy",
                "query_path": f"{ds}/queries.npy",
                "distance_metric": st.get("metric", "L2"),
                "dimension": st.get("dimension", 128),
                "point_type": st.get("point_type", "uint8"),
                "base_count": st["base_count"],
                "query_count": st["query_count"],
            }
        ],
    }


def _index_dir_of(cfg: dict[str, Any], cand: str, ds: str) -> Path | None:
    hits = sorted((Path(cfg["index_dir"]) / f"Evolved-{cand}" / ds).glob("*/evolved_meta.json"))
    return hits[0].parent if hits else None


def _run_stage(
    cfg: dict[str, Any], stage: str, cand: str, source: str, prog_host: Path, use_cache: bool
) -> dict[str, Any]:
    st = cfg["stages"][stage]
    ds = st["dataset"]
    points = _search_points(source)
    gr = cfg["guardrails"]
    if not 1 <= len(points) <= gr["max_search_points"]:
        raise ValueError(f"SEARCH_POINTS must have 1..{gr['max_search_points']} entries")
    index_root = Path(cfg["index_dir"])
    # A stage on the same base vectors (e.g. hidden queries) reuses another
    # stage's build.
    bstage = st.get("build_from", stage)
    bhash = _build_hash(source, bstage, cfg["stages"][bstage]["dataset"])
    cache_dir = index_root / "cache" / bstage / bhash
    cached = use_cache and (cache_dir / "evolved_meta.json").exists()
    run_tag = f"{cand}-{int(time.time())}"
    container_prog = f"/data/{cfg['programs_subdir']}/{prog_host.name}"
    prebuilt = f"cache/{bstage}/{bhash}" if cached else None
    t0 = time.time()
    results = _run_suite(
        _stage_config(cfg, stage, cand, container_prog, len(points), prebuilt, run_tag)
    )
    wall = time.time() - t0
    idx = cache_dir if cached else _index_dir_of(cfg, cand, ds)

    build_info: dict[str, Any] = {"cached": cached, "build_hash": bhash}
    if not cached:
        b = results[0].build_result if results else None
        if b is None or not b.success or idx is None:
            msg = (b.error_message if b else None) or "build failed"
            if b is not None and (b.duration_seconds or 0) >= 0.98 * st["build_timeout_s"]:
                msg = (
                    f"build exceeded the {st['build_timeout_s']} s limit for "
                    f"{st['base_count']:,} vectors (make the build faster)"
                )
            log = _tail(b.stderr_path if b else None)
            shutil.rmtree(index_root / f"Evolved-{cand}", ignore_errors=True)
            return {
                "ok": False,
                "error": f"build failed: {msg}",
                "stderr_tail": log,
                "wall_s": wall,
            }
        build_info["build_seconds"] = b.duration_seconds
    gt = np.load(st["ground_truth"], mmap_mode="r")
    out_points = []
    for r in results:
        sr = r.search_result
        params = (r.hyperparameters or {}).get("search", {})
        pi = int(params.get("point", -1))
        npz_path = idx / "results" / f"{run_tag}_point_{pi}.npz" if idx is not None else None
        # ann-suite's sync wrapper masks the runner's exit code, so a failed search
        # can arrive as success=True: also require no error and a results file.
        if sr is None or not sr.success or sr.error_message or not npz_path.exists():
            out_points.append(
                {
                    "point": pi,
                    "ok": False,
                    "error": (sr.error_message if sr else None) or "no search result",
                    "stderr_tail": _tail(sr.stderr_path if sr else None),
                }
            )
            continue
        ev = (sr.output or {}).get("evolved", {})
        npz = np.load(npz_path)
        rec = recall_at_k(npz["ids"], np.asarray(gt), cfg["k"])
        harness_pages = float(npz["pages"].mean())
        kernel_pages = ev.get("kernel_pages_per_query", r.disk_io.search_pages_per_query)
        # The harness count is kept inside the candidate's process, so it is not
        # trusted on its own: score whichever of it and the kernel's count is larger
        # (less a small slack for metadata reads).
        pages = max(harness_pages, (kernel_pages or 0.0) - gr["io_crosscheck_slack_pages"])
        rounds = float(ev.get("rounds_per_query") or 0.0)
        problems = list(ev.get("integrity", {}).get("violations", []))
        if not ev.get("integrity", {}).get("sandboxed"):
            problems.append("search did not run in the sandbox")
        if kernel_pages is not None and kernel_pages > (
            gr["io_crosscheck_ratio"] * harness_pages + gr["io_crosscheck_slack_pages"]
        ):
            problems.append(
                f"kernel read {kernel_pages:.1f} pages/query but the harness "
                f"counted {harness_pages:.1f}: reads bypassed io.read()"
            )
        if rounds > gr["max_rounds_per_query"]:
            problems.append(
                f"{rounds:.1f} I/O rounds/query exceeds {gr['max_rounds_per_query']} "
                "(each round waits for the SSD; batch reads into fewer io.read() calls)"
            )
        if ev.get("cpu_ms_per_query", 0) > gr["max_cpu_ms_per_query"]:
            problems.append(
                f"cpu {ev['cpu_ms_per_query']:.1f} ms/query exceeds {gr['max_cpu_ms_per_query']}"
            )
        out_points.append(
            {
                "point": pi,
                "ok": not problems,
                "problems": problems,
                "params": ev.get("params"),
                "recall": rec,
                "pages": pages,
                "harness_pages": harness_pages,
                "kernel_pages": kernel_pages,
                "rounds": rounds,
                "cpu_ms_per_query": ev.get("cpu_ms_per_query"),
                "peak_anon_mb": r.memory.search_peak_anon_mb,
                "qps_python": r.qps,
                "_result": r,
            }
        )
    if not cached and use_cache and idx is not None:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        shutil.move(str(idx), cache_dir)
        idx = cache_dir
    index_bytes = (
        sum(f.stat().st_size for d in ("disk", "mem") for f in (idx / d).rglob("*") if f.is_file())
        if idx is not None
        else 0
    )
    shutil.rmtree(index_root / f"Evolved-{cand}", ignore_errors=True)
    if index_bytes > gr["max_index_gb"] * 1e9:
        for p in out_points:
            if "recall" in p:
                p["ok"] = False
                p["problems"] = [
                    *p.get("problems", []),
                    f"index is {index_bytes / 1e9:.2f} GB, over the "
                    f"{gr['max_index_gb']} GB limit (8x the raw vectors)",
                ]
    if idx is not None and not use_cache:
        shutil.rmtree(idx, ignore_errors=True)
    elif idx is not None:
        for f in (idx / "results").glob(f"{run_tag}_*"):
            f.unlink()
    return {
        "ok": True,
        "build": build_info,
        "index_bytes": index_bytes,
        "points": out_points,
        "wall_s": wall,
    }


def _tail(path: Any, n: int = 3000) -> str:
    try:
        return Path(path).read_text()[-n:] if path else ""
    except OSError:
        return ""


def _harness_points(
    good: list[dict[str, Any]], index_bytes: int, floor: float, system: str, prefix: str
) -> list[Point]:
    return [
        Point(
            system=system,
            label=f"{prefix}{p['point']}:{json.dumps(p['params'])}",
            miss=1.0 - p["recall"],
            pages=p["pages"],
            dram_mb=max(0.0, (p["peak_anon_mb"] or 0.0) - floor),
            index_gb=index_bytes / 1e9,
            extra={
                "rounds": p["rounds"],
                "cpu_ms_per_query": p["cpu_ms_per_query"],
                "recall": p["recall"],
            },
        )
        for p in good
    ]


def _evaluate(
    cfg: dict[str, Any],
    program: Path,
    cand: str,
    stages: list[str],
    use_cache: bool = True,
    score_it: bool = True,
) -> dict[str, Any]:
    _ensure_sudo_env()
    source = program.read_text()
    progs = Path(cfg["data_dir"]) / cfg["programs_subdir"]
    progs.mkdir(parents=True, exist_ok=True)
    prog_host = progs / f"{cand}.py"
    shutil.copy2(program, prog_host)
    report: dict[str, Any] = {"candidate": cand, "stages": {}}
    floors = _floors(cfg)
    floor = floors.get(cfg["image"], 0.0)
    for stage in stages:
        res = _run_stage(cfg, stage, cand, source, prog_host, use_cache)
        report["stages"][stage] = {k: v for k, v in res.items() if k != "points"}
        report["stages"][stage]["points"] = [
            {k: v for k, v in p.items() if k != "_result"} for p in res.get("points", [])
        ]
        if not res["ok"]:
            report["combined_score"] = FAIL
            report["error"] = f"{stage}: {res['error']}"
            return report
        good = [p for p in res["points"] if p["ok"]]
        bad = [p for p in res["points"] if not p["ok"]]
        if bad and not good:
            report["combined_score"] = FAIL
            report["error"] = f"{stage}: every search point failed or broke a rule"
            return report
        if stage == "sanity":
            best = max(p["recall"] for p in good)
            report["sanity_best_recall"] = best
            if best < cfg["stages"]["sanity"]["min_recall"] and "full" in stages:
                report["combined_score"] = -5.0 - (1.0 - best)
                report["error"] = (
                    f"sanity gate: best recall@10 at 1M is {best:.3f} "
                    f"< {cfg['stages']['sanity']['min_recall']}"
                )
                return report
        if cfg["stages"][stage].get("scored") and score_it:
            cand_points = _harness_points(good, res["index_bytes"], floor, "candidate", "point")
            baselines = load_points(REPO / cfg["frontier"])
            sc = score(cand_points, baselines, Cells.from_config(cfg["score"]))
            report["stages"][stage]["score"] = sc["combined_score"]
            if stage != "full":
                continue
            report.update({k: sc[k] for k in ("combined_score", "best_cell", "recall_shortfall")})
            report["cells"] = sc["cells"]
            report["uncovered"] = sc["uncovered"]
            report["features"] = {
                "dram_mb": min(cp.dram_mb for cp in cand_points),
                "rounds": min(cp.extra["rounds"] or 0 for cp in cand_points),
                "best_recall": max(1 - cp.miss for cp in cand_points),
                "min_pages": min(cp.pages for cp in cand_points),
            }
    return report


def cmd_candidate(cfg: dict[str, Any], program: Path, cand: str, stages: list[str]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", cand):
        raise SystemExit("candidate id must match [A-Za-z0-9_.-]{1,64}")
    lock_path = Path(cfg["index_dir"]) / ".evolve.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            report = _evaluate(cfg, program, cand, stages)
            if "full" in stages and report.get("combined_score", FAIL) > _record(cfg)["score"]:
                _validate(cfg, program, cand, report)
        except Exception as exc:
            report = {
                "candidate": cand,
                "combined_score": FAIL,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:],
            }
        finally:
            _gc_cache(Path(cfg["index_dir"]) / "cache", cfg["cache"]["max_gb"])
    log_dir = REPO / cfg["results_dir"] / "candidates"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{cand}.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, default=str))


def _record_path(cfg: dict[str, Any]) -> Path:
    return REPO / cfg["results_dir"] / f"record_{cfg['name']}.json"


def _record(cfg: dict[str, Any]) -> dict[str, Any]:
    p = _record_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"score": 0.0, "candidate": None}


def _validate(cfg: dict[str, Any], program: Path, cand: str, report: dict[str, Any]) -> None:
    """Re-measure a would-be record on the same and on hidden queries; keep the
    minimum score, so neither measurement luck nor fitting to the scored queries
    sets a record."""
    raw = report["combined_score"]
    scores = {"first": raw}
    for stage in cfg["validation"]["stages"]:
        rep = _evaluate(cfg, program, f"{cand}-v{stage}", [stage])
        st = rep.get("stages", {}).get(stage, {})
        scores[f"{stage}_rerun" if stage == "full" else stage] = (
            st.get("score") if st.get("score") is not None else rep.get("combined_score", FAIL)
        )
    validated = min(scores.values())
    report["validation"] = {"scores": scores, "validated_score": validated}
    report["combined_score"] = validated
    if validated > _record(cfg)["score"]:
        _record_path(cfg).write_text(
            json.dumps({"score": validated, "candidate": cand, "time": time.time()})
        )


def cmd_reference(cfg: dict[str, Any], program: Path, name: str, system: str) -> None:
    """Measure a reference program (a published design implemented on the harness)
    on the full stage and merge its points into the frontier as baseline `system`.
    Use for classic designs no packaged system covers, so the evolver earns no
    credit for rediscovering them."""
    if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", name):
        raise SystemExit("reference name must match [A-Za-z0-9_.+-]{1,64}")
    lock_path = Path(cfg["index_dir"]) / ".evolve.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = _evaluate(cfg, program, f"ref-{name}", ["full"], score_it=False)
    st = report["stages"].get("full", {})
    if not st.get("ok"):
        raise SystemExit(f"reference failed: {report.get('error')}")
    good = [p for p in st["points"] if p["ok"]]
    floor = _floors(cfg).get(cfg["image"], 0.0)
    # Labels "<name>:p<i>:<params>": the name part keys merge replacement.
    points = _harness_points(good, st["index_bytes"], floor, system, f"{name}:p")
    allp = _merge_frontier(REPO / cfg["frontier"], points, {name})
    print(json.dumps({"added": len(points), "points": len(allp), "pareto": len(pareto(allp))}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/evolve/bigann10m.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("floors")
    b = sub.add_parser("baselines")
    b.add_argument("baseline_config", type=Path)
    c = sub.add_parser("candidate")
    c.add_argument("program", type=Path)
    c.add_argument("--id", required=True)
    c.add_argument("--stages", default="sanity,full")
    r = sub.add_parser("add-reference")
    r.add_argument("program", type=Path)
    r.add_argument("--name", required=True)
    r.add_argument("--system", required=True)
    ns = ap.parse_args()
    _quiet_logs()
    os.chdir(REPO)
    cfg = _cfg(ns.config)
    if ns.cmd == "floors":
        cmd_floors(cfg)
    elif ns.cmd == "baselines":
        cmd_baselines(cfg, ns.baseline_config)
    elif ns.cmd == "add-reference":
        cmd_reference(cfg, ns.program, ns.name, ns.system)
    else:
        cmd_candidate(cfg, ns.program, ns.id, ns.stages.split(","))


if __name__ == "__main__":
    main()
