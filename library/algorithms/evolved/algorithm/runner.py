"""ANN Suite runner for evolved disk-ANN programs (OpenEvolve candidates).

The candidate is a single Python file (see harness.py for its interface) passed
as build_args.program, a path under /data. Build copies it into the index
directory. Search loads search_args.program when given (a cached index built by
identical build code, searched with a newer candidate) and otherwise the copy.

Integrity measures (the candidate is machine-generated and selected for score,
so it is treated as untrusted):
- The evolution data dir holds base vectors and queries only; ground truth lives
  outside the container's mounts. This runner reports no recall: it writes the
  result ids to <index>/results/<run_tag>_point_<i>.npz and the host scores them.
- Disk reads at query time go through harness.QueryIO (O_DIRECT, counted). The
  host compares the harness page count with the kernel's io.stat for the query
  window; a large excess means reads bypassed the harness.
- Before the first query every file page under the writable/mounted trees is
  synced and evicted (posix_fadvise DONTNEED), and after the run the process
  must not hold file-backed mappings of index/data files or shmem beyond a small
  slack. Violations are reported in "integrity" and the host fails the point.
"""

from __future__ import annotations

import json
import os
import sys


def _pin_search_threads() -> None:
    """Size BLAS/OpenMP pools to search_args.threads before numpy/faiss load.

    Left alone, OpenBLAS spins one thread per core on every small mat-vec, which
    multiplies the CPU time per query by the core count without doing work.
    """
    if "--mode" not in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "search":
        return
    try:
        cfg = json.loads(sys.argv[sys.argv.index("--config") + 1])
        n = str(int(cfg.get("search_args", {}).get("threads", 1)))
    except (ValueError, IndexError):
        n = "1"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = n


_pin_search_threads()

import argparse  # noqa: E402
import ast  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from algorithm.harness import BuildContext, QueryIO, SearchContext  # noqa: E402

# Imported up front in every search so the measured memory floor (a null
# program) includes the libraries candidates are expected to use.
try:  # noqa: SIM105
    import faiss  # noqa: F401
except ImportError:
    pass
try:  # noqa: SIM105
    import numba  # noqa: F401
except ImportError:
    pass

SLACK_MB = 16.0
EVICT_ROOTS = ("/data", "/tmp", "/var/tmp", "/app", "/root", "/home")
LIB_PREFIXES = ("/usr/", "/lib", "/opt/", "/app/algorithm", "/etc/", "/sys/", "/proc/")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _search_points(source: str) -> list[dict[str, Any]]:
    """Read SEARCH_POINTS as a literal without executing the program."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SEARCH_POINTS" for t in node.targets
        ):
            points = ast.literal_eval(node.value)
            if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
                raise ValueError("SEARCH_POINTS must be a literal list of dicts")
            return points
    raise ValueError("program defines no SEARCH_POINTS literal")


def _load_program(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_build(config: dict[str, Any]) -> dict[str, Any]:
    try:
        start = time.perf_counter()
        index_path = Path(config["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        args = dict(config.get("build_args", {}))
        src = Path(args["program"])
        source = src.read_text()
        points = _search_points(source)
        shutil.copy2(src, index_path / "program.py")
        data = np.load(config["dataset_path"], mmap_mode="r")
        threads = int(args.get("threads", os.cpu_count() or 8))
        ctx = BuildContext(data, index_path, threads, str(config.get("metric", "L2")))
        prog = _load_program(index_path / "program.py")
        prog.build(ctx)
        ctx.close()
        meta = {
            "num_points": int(data.shape[0]),
            "dim": int(data.shape[1]),
            "num_search_points": len(points),
            "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        (index_path / "evolved_meta.json").write_text(json.dumps(meta))
        index_bytes = _dir_bytes(index_path / "disk") + _dir_bytes(index_path / "mem")
        return {
            "status": "success",
            "build_time_seconds": time.perf_counter() - start,
            "index_size_bytes": index_bytes,
            **meta,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "build_time_seconds": 0,
            "index_size_bytes": 0,
        }


def _evict_page_cache() -> int:
    """fsync + POSIX_FADV_DONTNEED every regular file under EVICT_ROOTS."""
    n = 0
    for root in EVICT_ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
                except OSError:
                    continue
                try:
                    with contextlib.suppress(OSError):
                        os.fsync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return n


def _suspicious_file_rss_mb() -> float:
    """Rss of file-backed mappings that are not libraries (e.g. mmapped index data)."""
    total_kb = 0
    current = None
    with open("/proc/self/smaps") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if "-" in parts[0] and len(parts) >= 5 and not parts[0].endswith(":"):
                path = parts[5] if len(parts) >= 6 else ""
                ok = (
                    not path.startswith("/")
                    or path.startswith(LIB_PREFIXES)
                    or ".so" in path
                    or path.endswith(".pyc")
                )
                current = None if ok else path
            elif parts[0] == "Rss:" and current is not None:
                total_kb += int(parts[1])
    return total_kb / 1024.0


def _cgroup_mem_stat(key: str) -> float:
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            k, v = line.split()
            if k == key:
                return int(v) / 2**20
    except OSError:
        pass
    return -1.0


def run_search(config: dict[str, Any]) -> dict[str, Any]:
    try:
        index_path = Path(config["index_path"])
        meta = json.loads((index_path / "evolved_meta.json").read_text())
        args = dict(config.get("search_args", {}))
        # A cached index (same build code) is searched with the candidate's own
        # program, passed per search; without one, use the program that built it.
        program_path = Path(args.get("program") or index_path / "program.py")
        source = program_path.read_text()
        points = _search_points(source)
        run_tag = str(args.get("run_tag", "run"))
        point = int(args["point"])
        if point >= len(points):
            raise ValueError(f"point {point} >= len(SEARCH_POINTS)={len(points)}")
        params = points[point]
        k = int(config.get("k", 10))
        threads = int(args.get("threads", 1))
        queries = np.load(config["queries_path"])
        nq = len(queries)

        w_start = _now()
        t0 = time.perf_counter()
        prog = _load_program(program_path)
        ctx = SearchContext(index_path, threads, str(config.get("metric", "L2")))
        searcher = prog.Searcher(ctx, dict(params))
        load_s = time.perf_counter() - t0
        evicted = _evict_page_cache()
        shm_used_mb = 0.0
        try:
            st = os.statvfs("/dev/shm")
            shm_used_mb = (st.f_blocks - st.f_bfree) * st.f_frsize / 2**20
        except OSError:
            pass
        w_end = _now()

        ids = np.full((nq, k), -1, dtype=np.int64)
        pages = np.zeros(nq, dtype=np.int32)
        rounds = np.zeros(nq, dtype=np.int32)
        lat = np.zeros(nq, dtype=np.float64)
        q_start = _now()
        cpu0 = time.process_time()
        t1 = time.perf_counter()
        for i in range(nq):
            io = QueryIO(ctx)
            ts = time.perf_counter()
            res = np.asarray(searcher.search(queries[i], k, io), dtype=np.int64).ravel()[:k]
            lat[i] = time.perf_counter() - ts
            ids[i, : res.size] = res
            pages[i], rounds[i] = io.pages, io.rounds
        total_s = time.perf_counter() - t1
        cpu_s = time.process_time() - cpu0
        q_end = _now()

        bad_ids = int(((ids < -1) | (ids >= meta["num_points"])).sum())
        integrity = {
            "evicted_files": evicted,
            "shm_used_mb": round(shm_used_mb, 2),
            "file_mapped_rss_mb": round(_suspicious_file_rss_mb(), 2),
            "cgroup_shmem_mb": round(_cgroup_mem_stat("shmem"), 2),
            "out_of_range_ids": bad_ids,
        }
        violations = []
        if shm_used_mb > SLACK_MB:
            violations.append(f"/dev/shm holds {shm_used_mb:.0f} MB")
        if integrity["file_mapped_rss_mb"] > SLACK_MB:
            violations.append(f"file-backed mappings hold {integrity['file_mapped_rss_mb']} MB")
        if integrity["cgroup_shmem_mb"] > SLACK_MB:
            violations.append(f"cgroup shmem is {integrity['cgroup_shmem_mb']} MB")
        if bad_ids:
            violations.append(f"{bad_ids} result ids out of range")
        integrity["violations"] = violations

        out_dir = index_path / "results"
        out_dir.mkdir(exist_ok=True)
        np.savez(
            out_dir / f"{run_tag}_point_{point}.npz", ids=ids, pages=pages, rounds=rounds, lat=lat
        )
        lat_ms = lat * 1000.0
        return {
            "status": "success",
            "total_queries": nq,
            "total_time_seconds": total_s,
            "qps": nq / total_s if total_s > 0 else 0.0,
            "mean_latency_ms": float(lat_ms.mean()),
            "p50_latency_ms": float(np.percentile(lat_ms, 50)),
            "p95_latency_ms": float(np.percentile(lat_ms, 95)),
            "p99_latency_ms": float(np.percentile(lat_ms, 99)),
            "warmup_duration_seconds": load_s,
            "warmup_start_timestamp": w_start,
            "warmup_end_timestamp": w_end,
            "load_duration_seconds": load_s,
            "query_start_timestamp": q_start,
            "query_end_timestamp": q_end,
            "stats": {"io_reads": int(pages.sum()), "hops": int(rounds.sum())},
            "evolved": {
                "point": point,
                "params": params,
                "harness_pages_per_query": float(pages.mean()),
                "rounds_per_query": float(rounds.mean()),
                "cpu_ms_per_query": cpu_s * 1000.0 / nq,
                "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "build_program_sha256": meta["program_sha256"],
                "integrity": integrity,
            },
        }
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "total_queries": 0,
            "total_time_seconds": 0,
            "qps": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolved disk-ANN runner")
    parser.add_argument("--mode", choices=["build", "search"], required=True)
    parser.add_argument("--config", required=True)
    ns = parser.parse_args()
    config = json.loads(ns.config)
    result = run_build(config) if ns.mode == "build" else run_search(config)
    print(json.dumps(result))
    results_dir = Path("/results")
    if results_dir.exists():
        with contextlib.suppress(OSError):
            (results_dir / "metrics.json").write_text(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
