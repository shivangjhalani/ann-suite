"""Native SPANN (SPTAG ssdserving) vs ann-suite (Docker indexsearcher) on the same natively built index.

Arms, run interleaved per point and repeat, page cache dropped before each:
  suite      real `ann-suite run` (Docker, indexsearcher, CgroupsV2Collector)
  native     SPTAG `ssdserving` from /home/isfcr/spann (the native workflow's search tool)
  native_is  native `indexsearcher` binary with the suite runner's exact command line (diagnostic:
             separates harness/timing effects from Docker/compiler effects)

Both native arms run in a system-level systemd scope with the same cpuset as the container and are
monitored by ann-suite's own CgroupsV2Collector, so CPU/memory/IO come from identical code in
every arm. Result IDs of every arm are saved and recall is recomputed offline against one truth.

    ANN_SUITE_SUDO_PASSWORD=... uv run python tools/harness_compare/spann_harness_compare.py prep <build_root>
    ANN_SUITE_SUDO_PASSWORD=... uv run python tools/harness_compare/spann_harness_compare.py run <build_root> \\
        --irn 32 64 128 --repeats 3
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector

REPO = Path(__file__).resolve().parents[2]
SPTAG_BIN = Path("/home/isfcr/spann/SPTAG/Release")
DATA_DIR = Path("/home/isfcr/data")
CPUS = "0-7"
THREADS = 8
K = 10
SUITE_IMAGE = "ann-suite/spann:latest"
DATASET = {
    "name": "sift10m",
    "base_path": "sift10m/base.npy",
    "query_path": "sift10m/queries.npy",
    "ground_truth_path": "sift10m/ground_truth.npy",
    "distance_metric": "L2",
    "dimension": 128,
    "point_type": "float32",
    "base_count": 10_000_000,
    "query_count": 10_000,
}
QUERY_U8 = DATA_DIR / "bigann/query.u8bin"
TRUTH_DIR = Path("/home/isfcr/spann/data")  # truth<scale>.bin, the native workflow's truth


def set_scale(scale: str) -> None:
    """Point DATASET at the BIGANN prefix matching the index's scale (e.g. "10m", "100m")."""
    name = f"sift{scale}"
    DATASET.update(name=name, base_path=f"{name}/base.npy", query_path=f"{name}/queries.npy",
                   ground_truth_path=f"{name}/ground_truth.npy",
                   base_count=int(scale[:-1]) * 1_000_000)  # fmt: skip


def sudo(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    pw = os.environ["ANN_SUITE_SUDO_PASSWORD"]
    return subprocess.run(["sudo", "-S", "-p", "", *cmd], input=pw + "\n", text=True,
                          capture_output=True, check=True)  # fmt: skip


def drop_caches() -> None:
    sudo(["sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"])


def read_ini(path: Path) -> configparser.ConfigParser:
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    ini.read(path)
    return ini


def write_ini(ini: configparser.ConfigParser, path: Path) -> None:
    with path.open("w") as f:
        ini.write(f, space_around_delimiters=False)


def base_section(root: Path) -> dict[str, str]:
    """[Base] of the build config with nothing that makes a search load extra data.

    ssdserving loads VectorPath (the full base set) into RAM before searching and again after it
    for recall; indexsearcher never does. Blank it so both load only the index itself.
    """
    base = dict(read_ini(root / "build.ini")["Base"])
    base.update(VectorPath="", QueryPath=str(QUERY_U8), TruthPath="", GenerateTruth="false",
                WarmupPath="")  # fmt: skip
    return base


def search_params(irn: int, ppl: int) -> dict[str, str]:
    return {"ResultNum": str(K), "MaxCheck": "4096", "SearchInternalResultNum": str(irn),
            "SearchPostingPageLimit": str(ppl)}  # fmt: skip


def prep(root: Path) -> None:
    """Make a native SPANN index loadable by ann-suite: indexloader.ini."""
    manifest = json.loads((root / "manifest.json").read_text())
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    ini["Index"] = {"IndexAlgoType": "SPANN", "ValueType": base_section(root)["ValueType"]}
    ini["Base"] = base_section(root)
    ini["SelectHead"] = {"isExecute": "false"}
    # SPANN's own SaveConfig writes the head (BKT) index's parameters under [BuildHead]; without
    # them the head loads with BKT defaults (DistCalcMethod=Cosine on an L2 index: recall drops
    # with unchanged IO). ssdserving instead loads head_index/indexloader.ini directly.
    index = root / "index"
    head = dict(read_ini(index / "head_index/indexloader.ini")["Index"])
    ini["BuildHead"] = {"isExecute": "false", **head}
    ini["BuildSSDIndex"] = {"isExecute": "true", "BuildSsdIndex": "false",
                            **search_params(64, manifest["posting_page_limit"])}  # fmt: skip
    write_ini(ini, index / "indexloader.ini")
    print(f"wrote {index / 'indexloader.ini'}")


def parse_time_v(text: str) -> dict[str, Any]:
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text)
    secs = None
    if wall:
        secs = 0.0
        for part in wall.group(1).split(":"):
            secs = secs * 60 + float(part)
    return {"maxrss_mb": int(rss.group(1)) / 1024 if rss else None, "wall_s": secs}


def run_in_scope(cmd: list[str], cwd: Path, log: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run cmd in a system.slice scope pinned to CPUS, monitored by ann-suite's collector."""
    name = "hc" + uuid.uuid4().hex[:10]
    tfile = cwd / f"{name}.time"
    # `sleep` keeps the cgroup alive after the binary exits so the final sample holds the totals.
    inner = f"/usr/bin/time -v -o {tfile} {' '.join(cmd)} > {log} 2>&1; sleep 0.3"
    pw = os.environ["ANN_SUITE_SUDO_PASSWORD"]
    proc = subprocess.Popen(
        ["sudo", "-S", "-p", "", "systemd-run", "--quiet", "--scope", f"--unit={name}",
         "-p", f"AllowedCPUs={CPUS}", "--uid=isfcr", "--gid=isfcr", "--working-directory",
         str(cwd), "sh", "-c", inner],
        stdin=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert proc.stdin is not None
    proc.stdin.write(pw + "\n")
    proc.stdin.close()
    cg = Path(f"/sys/fs/cgroup/system.slice/{name}.scope")
    t0 = time.perf_counter()
    while not cg.exists():
        if proc.poll() is not None or time.perf_counter() - t0 > 30:
            raise RuntimeError(f"scope {name} never appeared")
        time.sleep(0.001)
    collector = CgroupsV2Collector(interval_ms=50)
    collector.start(name)
    proc.wait()
    res = collector.stop()
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} failed rc={proc.returncode}; see {log}")
    cgm = {
        "cg_duration_s": res.duration_seconds,
        "cg_cpu_s": res.cpu_time_total_seconds,
        "cg_avg_cpu_pct": res.avg_cpu_percent,
        "cg_peak_mem_mb": res.peak_memory_mb,
        "cg_peak_file_mb": res.peak_file_bytes / 2**20,
        "cg_read_mb": res.total_read_bytes / 2**20,
        "cg_read_ops": res.total_read_ops,
        "cg_avg_read_iops": res.avg_read_iops,
        "cg_majfault": res.pgmajfault_delta,
        "cg_samples": res.sample_count,
    }
    return cgm, parse_time_v(tfile.read_text())


def recall(ids: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(ids[i, :K]) & set(truth[i, :K])) / K for i in range(len(ids))]))


def load_truth(scale: str) -> np.ndarray:
    t = np.fromfile(TRUTH_DIR / f"truth{scale}.bin", dtype=np.int32)
    n, k = int(t[0]), int(t[1])
    return t[2 : 2 + n * k].reshape(n, k)


def native_arm(
    root: Path, irn: int, ppl: int, out: Path, tag: str, keep_vector_path: bool = False
) -> dict[str, Any]:
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str  # type: ignore[assignment,method-assign]
    ini["Base"] = base_section(root)
    if keep_vector_path:  # as search_check.py does: ssdserving then loads the whole base set
        ini["Base"]["VectorPath"] = read_ini(root / "build.ini")["Base"]["VectorPath"]
    for sec in ("SelectHead", "BuildHead", "BuildSSDIndex"):
        ini[sec] = {"isExecute": "false"}
    res_bin = out / f"{tag}.res.bin"
    ini["SearchSSDIndex"] = {"isExecute": "true", "BuildSsdIndex": "false",
                             "SearchThreadNum": str(THREADS), "NumberOfThreads": str(THREADS),
                             "SearchResult": str(res_bin), **search_params(irn, ppl)}  # fmt: skip
    ini_path = out / f"{tag}.ini"
    write_ini(ini, ini_path)
    log = out / f"{tag}.log"
    cgm, tv = run_in_scope([str(SPTAG_BIN / "ssdserving"), str(ini_path)], out, log)
    text = log.read_text()

    def pct_block(title: str) -> list[float] | None:
        # "<title>:\n[1] Avg\t50tiles...\n[1] v0\tv1...": Avg, p50, p90, p95, p99, p99.9, Max
        m = re.search(re.escape(title) + r":\s*\n.*Avg.*\n(?:\[\d+\] )?([\d.\t ]+)\n", text)
        return [float(x) for x in m.group(1).split()] if m else None

    qps = re.findall(r"actuallQPS is ([0-9.]+)", text)
    lat = pct_block("Total Latency Distribution")
    dio = pct_block("Total Disk IO Distribution")
    pages = pct_block("Total Disk Page Access Distribution")
    raw = np.fromfile(res_bin, dtype=np.int32)
    n, k = int(raw[0]), int(raw[1])
    ids = raw[2:].reshape(n, k, 2)[:, :, 0]  # (vid int32, dist float32) pairs
    np.save(out / f"{tag}.ids.npy", ids)
    return {
        **cgm, **tv, "tool_qps": float(qps[-1]) if qps else None, "ids": ids,
        "tool_lat_mean_ms": lat[0] if lat else None, "tool_lat_p50_ms": lat[1] if lat else None,
        "tool_lat_p99_ms": lat[4] if lat else None,
        "tool_disk_ios_per_q": dio[0] if dio else None,
        "tool_pages_per_q": pages[0] if pages else None,
    }  # fmt: skip


def indexsearcher_cmd(
    index: Path, queries_bin: Path, result: Path, irn: int, ppl: int
) -> list[str]:
    """The exact command line library/algorithms/spann/algorithm/runner.py builds."""
    return [str(SPTAG_BIN / "indexsearcher"), "-i", str(queries_bin), "-x", str(index),
            "-o", str(result), "-d", "128", "-v", "UInt8", "-f", "DEFAULT", "-k", str(K),
            "-b", "10000", "-of", "0", f"BuildSSDIndex.SearchInternalResultNum={irn}",
            f"BuildSSDIndex.SearchPostingPageLimit={ppl}", "-t", str(THREADS)]  # fmt: skip


def parse_txt_results(path: Path) -> np.ndarray:
    ids = np.full((DATASET["query_count"], K), -1, dtype=np.int64)
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        q, vals = line.split(":", 1)
        for j, v in enumerate(vals.split("|")[:K]):
            if "@" in v and not v.endswith("NULL"):
                ids[int(q), j] = int(v.split("@", 1)[1])
    return ids


def indexsearcher_tool_qps(text: str) -> float | None:
    # Final summary line: 0-10000 <maxcheck> avg 99% 95% recall qps
    m = re.findall(r"0-10000\t\S+\t[\d.]+\t[\d.]+\t[\d.]+\t[\d.]+\t([\d.]+)\n", text)
    return float(m[-1]) if m else None


def native_is_arm(root: Path, irn: int, ppl: int, out: Path, tag: str) -> dict[str, Any]:
    queries_bin = out / "queries_u8.bin"
    if not queries_bin.exists():
        q = np.load(DATA_DIR / DATASET["query_path"])
        with queries_bin.open("wb") as f:
            np.asarray(q.shape, dtype=np.int32).tofile(f)
            q.astype(np.uint8).tofile(f)
    result = out / f"{tag}.txt"
    log = out / f"{tag}.log"
    cgm, tv = run_in_scope(
        indexsearcher_cmd(root / "index", queries_bin, result, irn, ppl), out, log
    )
    ids = parse_txt_results(result)
    np.save(out / f"{tag}.ids.npy", ids)
    return {**cgm, **tv, "tool_qps": indexsearcher_tool_qps(log.read_text()), "ids": ids}


def suite_arm(root: Path, irn: int, ppl: int, out: Path, tag: str) -> dict[str, Any]:
    run_dir = out / tag
    run_dir.mkdir()
    cfg = {
        "name": "harness-compare-spann",
        "data_dir": str(DATA_DIR),
        "results_dir": str(run_dir),
        "index_dir": str(REPO / "indices"),
        "monitor_interval_ms": 50,
        "resources": {"cpu_affinity": CPUS},
        "algorithms": [{
            "name": "SPANN-native-index",
            "docker_image": SUITE_IMAGE,
            "algorithm_type": "disk",
            "build": {"prebuilt_path": str(root / "index"),
                      "required_files": ["SPTAGFullList.bin", "indexloader.ini"]},
            "search": {"timeout_seconds": 1800, "k": K, "batch_mode": False,
                       "args": {"num_threads": THREADS, "posting_page_limit": ppl,
                                "internal_result_num": irn, "value_type": "UInt8"}},
        }],
        "datasets": [DATASET],
    }  # fmt: skip
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "ann_suite.cli", "run", "--config", str(cfg_path)],
                       capture_output=True, text=True, cwd=REPO)  # fmt: skip
    wall = time.perf_counter() - t0
    (run_dir / "cli.log").write_text(p.stdout + p.stderr)
    r = json.loads(next(run_dir.rglob("results.json")).read_text())[0]
    if r["search"].get("error") or r["quality"]["recall"] in (None, 0):
        raise RuntimeError(f"suite point failed: {r['search'].get('error')}; see {run_dir}")
    ids = parse_txt_results(root / "index/search-results.txt")
    np.save(out / f"{tag}.ids.npy", ids)
    stderr = "".join(
        f.read_text(errors="replace") for f in run_dir.rglob("*stderr*") if f.is_file()
    )
    s, d = r["search"], r["search"]["disk_io"]
    return {
        "suite_qps": r["quality"]["qps"], "suite_recall": r["quality"]["recall"],
        "suite_mean_latency_ms": r["latency"]["mean_ms"], "suite_p99_ms": r["latency"]["p99_ms"],
        "cg_duration_s": s["duration_seconds"], "cg_cpu_s": s["cpu_time_seconds"],
        "cg_avg_cpu_pct": s["avg_cpu_percent"], "cg_peak_mem_mb": s["peak_rss_mb"], "cg_peak_anon_mb": s.get("peak_anon_mb"),
        "cg_peak_file_mb": s["file_cache_peak_mb"], "cg_read_mb": d["total_read_mb"],
        "cg_read_ops": round(d["reads_per_query"] * DATASET["query_count"])
        if d["reads_per_query"] is not None else None,
        "cg_avg_read_iops": d["avg_read_iops"], "suite_reads_per_query": d["reads_per_query"],
        "suite_pages_per_query": d["pages_per_query"], "suite_sample_count": r["metadata"]["sample_count"],
        "probe_qd64_kiops": r["run_conditions"].get("probe_qd64_kiops"),
        "tool_qps": indexsearcher_tool_qps(stderr), "cli_wall_s": wall, "ids": ids,
    }  # fmt: skip


def native_vp_arm(root: Path, irn: int, ppl: int, out: Path, tag: str) -> dict[str, Any]:
    return native_arm(root, irn, ppl, out, tag, keep_vector_path=True)


ARMS = {
    "suite": suite_arm,
    "native": native_arm,
    "native_is": native_is_arm,
    "native_vp": native_vp_arm,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prep", "run"])
    ap.add_argument("root", type=Path, help="SPANN build root (contains build.ini, index/)")
    ap.add_argument("--irn", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--arms", nargs="+", default=["suite", "native", "native_is"], choices=list(ARMS)
    )
    args = ap.parse_args()
    root = args.root.resolve()
    if args.cmd == "prep":
        prep(root)
        return
    manifest = json.loads((root / "manifest.json").read_text())
    ppl = manifest["posting_page_limit"]
    set_scale(manifest["scale"])
    posting = root / "index/SPTAGFullList.bin"
    mtime = posting.stat().st_mtime
    out = REPO / "results/harness_compare" / f"{root.name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True)
    truth = load_truth(manifest["scale"])
    rows = []
    for rep in range(args.repeats):
        for irn in args.irn:
            ref = None
            for arm in args.arms:
                tag = f"{arm}_irn{irn}_r{rep}"
                drop_caches()
                row = ARMS[arm](root, irn, ppl, out, tag)
                ids = row.pop("ids")
                ref = ids if ref is None else ref
                row.update(rep=rep, irn=irn, arm=arm, recall10=recall(ids, truth),
                           ids_identical_to_first_arm=bool(np.array_equal(ids, ref)),
                           qps_process_wall=DATASET["query_count"] / row["wall_s"]
                           if row.get("wall_s") else None)  # fmt: skip
                rows.append(row)
                print(json.dumps(row), flush=True)
                (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert posting.stat().st_mtime == mtime, "posting file was modified!"
    print(f"results in {out}")


if __name__ == "__main__":
    main()
