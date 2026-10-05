"""ANN Suite runner for evolved disk-ANN programs (OpenEvolve candidates).

The candidate is a single Python file (see harness.py for its interface) passed
as build_args.program, a path under /data. Build copies it into the index
directory. Search loads search_args.program when given (a cached index built by
identical build code, searched with a newer candidate) and otherwise the copy.

Integrity measures (the candidate is machine-generated and selected for score,
so it is treated as untrusted):
- Candidate code never runs in this process. The container runs as root with no
  network (the evolve config sets container_user/network); this runner starts
  algorithm/sandbox.py as `nobody` and hands it file descriptors, not paths, for
  the base vectors and its index directory. /data is 0700 on the host, so the
  candidate cannot read queries, ground truth or other programs, and the runner
  holds the queries and sends them one at a time. Index files are handed back to
  the owner of /data/index afterwards.
- The evolution data dir holds base vectors and queries only; ground truth lives
  outside the container's mounts. This runner reports no recall: it writes the
  result ids to <index>/results/<run_tag>_point_<i>.npz and the host scores them.
- Disk reads at query time are done by this process: the candidate's io.read()
  sends the page ids of one round over the pipe and gets the pages back. The disk
  files are closed to the sandbox user (mode 0700 directory), so rounds and pages
  per round are counted here, and the candidate's CPU time excludes the reads. The
  host still compares pages with the kernel's io.stat for the query window and
  scores the larger of the two (less a small slack).
- Before the first query every file page under the writable/mounted trees is
  synced and evicted (posix_fadvise DONTNEED), and after the run the process
  must not hold file-backed mappings of index/data files or shmem beyond a small
  slack, and the container's page cache must not have grown (io.read() bypasses
  it, so growth means file data cached outside the DRAM measurement). Violations are reported in "integrity" and the host fails the point.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from algorithm import wire
from algorithm.harness import PAGE, page_buffer

NOBODY = 65534
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


class CandidateError(Exception):
    """The candidate program failed inside the sandbox (its traceback is on stderr)."""


class Sandbox:
    """The candidate's process: `nobody`, no inherited fds except the ones given."""

    def __init__(self, pass_fds: tuple[int, ...], threads: int | None) -> None:
        if os.geteuid() != 0:
            raise RuntimeError(
                "the evolved runner must start as root to sandbox candidate code "
                "(set container_user: root on the algorithm)"
            )
        to_child_r, self._w = os.pipe()
        self._r, from_child_w = os.pipe()
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": "/app",
            "PYTHONUNBUFFERED": "1",
            "HOME": "/tmp",
            "NUMBA_CACHE_DIR": "/tmp/numba_cache_sandbox",
        }
        if threads is not None:
            # Left alone, OpenBLAS spins one thread per core on every small
            # mat-vec, multiplying CPU time per query without doing work.
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            ):
                env[var] = str(threads)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "algorithm.sandbox", str(to_child_r), str(from_child_w)],
            pass_fds=(to_child_r, from_child_w, *pass_fds),
            user=NOBODY,
            group=NOBODY,
            extra_groups=[],
            umask=0o022,
            env=env,
            cwd="/tmp",
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr.fileno(),
        )
        os.close(to_child_r)
        os.close(from_child_w)
        self._buf = None

    def send_json(self, obj: Any) -> None:
        wire.send_json(self._w, obj)

    def reply(self) -> None:
        """Wait for the sandbox's acknowledgement; raise its error if it failed."""
        try:
            msg = wire.recv_json(self._r)
        except EOFError:
            self.proc.wait()
            msg = f"candidate process died (exit code {self.proc.returncode})"
            raise CandidateError(msg) from None
        if not msg.get("ok"):
            raise CandidateError(str(msg.get("error", "unknown error")))

    def query(
        self, q: np.ndarray, k: int, disk: dict[str, tuple[int, int]]
    ) -> tuple[np.ndarray, list[int]]:
        """Send one query and serve its page reads until the answer arrives.
        Returns the ids and the number of distinct pages read in each round."""
        wire.send(self._w, b"Q", q.tobytes())
        rounds: list[int] = []
        limit = max(wire.MAX_CONTROL, 8 * k + 8, 2 + 65535 + 8 * wire.MAX_ROUND_PAGES)
        while True:
            try:
                kind, payload = wire.recv(self._r, limit)
            except EOFError:
                self.proc.wait()
                msg = f"candidate process died (exit code {self.proc.returncode})"
                raise CandidateError(msg) from None
            if kind == b"R":
                rounds.append(self._serve_read(payload, disk))
                continue
            if kind == b"J":
                raise CandidateError(str(json.loads(payload).get("error", "unknown error")))
            if kind != b"Q" or len(payload) != 8 * k + 8:
                raise CandidateError("malformed reply from the candidate process")
            return np.frombuffer(payload[: 8 * k], dtype=np.int64).copy(), rounds

    def _serve_read(self, payload: bytes, disk: dict[str, tuple[int, int]]) -> int:
        """One I/O round: read the requested pages (O_DIRECT) and send them back."""
        name, raw = wire.unpack_read(payload)
        if name not in disk or len(raw) % 8:
            raise CandidateError(f"bad read request for {name!r}")
        fd, num_pages = disk[name]
        ids = np.unique(np.frombuffer(raw, dtype="<i8"))
        if ids.size > wire.MAX_ROUND_PAGES:
            raise CandidateError(f"one round asked for {ids.size} pages")
        if ids.size and (ids[0] < 0 or ids[-1] >= num_pages):
            raise CandidateError(f"page id out of range for {name} ({num_pages} pages)")
        need = ids.size * PAGE
        if self._buf is None or len(self._buf) < need:
            if self._buf is not None:
                self._buf.close()
            self._buf = page_buffer(max(need, 64 * PAGE))
        view = memoryview(self._buf)
        for j, pid in enumerate(ids.tolist()):
            if os.preadv(fd, [view[j * PAGE : (j + 1) * PAGE]], pid * PAGE) != PAGE:
                raise OSError(f"short read on {name} page {pid}")
        wire.send(self._w, b"P", view[:need])
        return int(ids.size)

    def cpu_seconds(self) -> float:
        """utime + stime (+ waited-for children) of the candidate process."""
        fields = Path(f"/proc/{self.proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = sum(int(x) for x in fields[11:15])
        return ticks / os.sysconf("SC_CLK_TCK")

    def close(self) -> None:
        with contextlib.suppress(OSError):
            wire.send(self._w, b"X")
        with contextlib.suppress(OSError):
            os.close(self._w)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        with contextlib.suppress(OSError):
            os.close(self._r)


def _host_owner() -> tuple[int, int]:
    st = os.stat("/data/index")
    return st.st_uid, st.st_gid


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    for p in [path, *path.rglob("*")]:
        with contextlib.suppress(OSError):
            os.lchown(p, uid, gid)


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
        uid, gid = _host_owner()
        os.chown(index_path, NOBODY, NOBODY)
        base_fd = os.open(config["dataset_path"], os.O_RDONLY)
        index_fd = os.open(index_path, os.O_RDONLY | os.O_DIRECTORY)
        sandbox = None
        try:
            sandbox = Sandbox((base_fd, index_fd), threads=None)
            sandbox.send_json(
                {
                    "cmd": "build",
                    "source": source,
                    "base_fd": base_fd,
                    "index_fd": index_fd,
                    "threads": threads,
                    "metric": str(config.get("metric", "L2")),
                }
            )
            sandbox.reply()
        finally:
            if sandbox is not None:
                sandbox.close()
            os.close(base_fd)
            os.close(index_fd)
            _chown_tree(index_path, uid, gid)
        meta = {
            "num_points": int(data.shape[0]),
            "dim": int(data.shape[1]),
            "num_search_points": len(points),
            "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        (index_path / "evolved_meta.json").write_text(json.dumps(meta))
        os.chown(index_path / "evolved_meta.json", uid, gid)
        index_bytes = _dir_bytes(index_path / "disk") + _dir_bytes(index_path / "mem")
        return {
            "status": "success",
            "build_time_seconds": time.perf_counter() - start,
            "index_size_bytes": index_bytes,
            **meta,
        }
    except Exception as exc:
        import traceback

        if not isinstance(exc, CandidateError):
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


@contextlib.contextmanager
def _as_sandbox_user():
    """Take the sandbox's uid as effective uid (root stays the saved uid) while
    reading its /proc files: without CAP_SYS_PTRACE, which Docker drops, root may
    not read another user's /proc/<pid>/smaps."""
    os.setegid(NOBODY)
    os.seteuid(NOBODY)
    try:
        yield
    finally:
        os.seteuid(0)
        os.setegid(0)


def _suspicious_file_rss_mb(pid: int) -> float:
    """Rss of file-backed mappings that are not libraries (e.g. mmapped index data)."""
    total_kb = 0
    current = None
    with _as_sandbox_user():
        lines = Path(f"/proc/{pid}/smaps").read_text().splitlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if "-" in parts[0] and len(parts) >= 5 and not parts[0].endswith(":"):
            path = parts[5] if len(parts) >= 6 else ""
            # Only root-owned trees are exempt: the sandbox cannot write there, so
            # a data file cannot pass as a library by its name (x.so, x.pyc).
            ok = not path.startswith("/") or path.startswith(LIB_PREFIXES)
            current = None if ok else path
        elif parts[0] == "Rss:" and current is not None:
            total_kb += int(parts[1])
    return total_kb / 1024.0


def _cgroup_read_bytes() -> int:
    """Bytes the container has read from block devices (cgroup v2 io.stat)."""
    total = 0
    try:
        for line in Path("/sys/fs/cgroup/io.stat").read_text().splitlines():
            for field in line.split()[1:]:
                if field.startswith("rbytes="):
                    total += int(field[7:])
    except OSError:
        return -1
    return total


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
    sandbox = None
    index_fd = -1
    disk_fds: dict[str, tuple[int, int]] = {}
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
        uid, gid = _host_owner()

        w_start = _now()
        t0 = time.perf_counter()
        index_fd = os.open(index_path, os.O_RDONLY | os.O_DIRECTORY)
        # The disk files are read by this process only: close them to the sandbox.
        disk_dir = index_path / "disk"
        os.chmod(disk_dir, 0o700)
        for f in disk_dir.glob("*.pages"):
            fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
            disk_fds[f.stem] = (fd, os.fstat(fd).st_size // PAGE)
        sandbox = Sandbox((index_fd,), threads=threads)
        sandbox.send_json(
            {
                "cmd": "search",
                "source": source,
                "index_fd": index_fd,
                "threads": threads,
                "metric": str(config.get("metric", "L2")),
                "params": params,
                "k": k,
                "dim": int(queries.shape[1]),
                "dtype": queries.dtype.str,
                "disk_pages": {name: n for name, (_fd, n) in disk_fds.items()},
            }
        )
        sandbox.reply()
        # One untimed warm-up query, as a server would run before taking traffic:
        # numba compiles a kernel on its first call, and that must not be charged
        # as search CPU. It is the mean of the first queries, so no scored query
        # is seen early; its reads are served but not counted.
        warm = queries[: min(16, nq)].astype(np.float64).mean(axis=0)
        if np.issubdtype(queries.dtype, np.integer):
            warm = warm.round()
        sandbox.query(np.ascontiguousarray(warm.astype(queries.dtype)), k, disk_fds)
        load_s = time.perf_counter() - t0
        evicted = _evict_page_cache()
        file_mb_start = _cgroup_mem_stat("file")
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
        round_pages: list[int] = []
        lat = np.zeros(nq, dtype=np.float64)
        q_start = _now()
        cpu0 = sandbox.cpu_seconds()
        rbytes0 = _cgroup_read_bytes()
        t1 = time.perf_counter()
        for i in range(nq):
            ts = time.perf_counter()
            ids[i], rp = sandbox.query(np.ascontiguousarray(queries[i]), k, disk_fds)
            pages[i], rounds[i] = sum(rp), len(rp)
            round_pages.extend(rp)
            lat[i] = time.perf_counter() - ts
        total_s = time.perf_counter() - t1
        rbytes1 = _cgroup_read_bytes()
        cpu_s = sandbox.cpu_seconds() - cpu0
        q_end = _now()
        page_cache_growth_mb = _cgroup_mem_stat("file") - file_mb_start

        bad_ids = int(((ids < -1) | (ids >= meta["num_points"])).sum())
        integrity = {
            "sandboxed": True,
            "evicted_files": evicted,
            "shm_used_mb": round(shm_used_mb, 2),
            "file_mapped_rss_mb": round(_suspicious_file_rss_mb(sandbox.proc.pid), 2),
            "cgroup_shmem_mb": round(_cgroup_mem_stat("shmem"), 2),
            "page_cache_growth_mb": round(page_cache_growth_mb, 2),
            "out_of_range_ids": bad_ids,
        }
        sandbox.close()
        sandbox = None
        violations = []
        if shm_used_mb > SLACK_MB:
            violations.append(f"/dev/shm holds {shm_used_mb:.0f} MB")
        if integrity["file_mapped_rss_mb"] > SLACK_MB:
            violations.append(f"file-backed mappings hold {integrity['file_mapped_rss_mb']} MB")
        if integrity["cgroup_shmem_mb"] > SLACK_MB:
            violations.append(f"cgroup shmem is {integrity['cgroup_shmem_mb']} MB")
        if page_cache_growth_mb > SLACK_MB:
            # io.read() is O_DIRECT; page cache filled during the queries means file
            # data read with ordinary reads, i.e. DRAM that anon memory does not show.
            violations.append(
                f"page cache grew by {page_cache_growth_mb:.0f} MB during the queries: "
                "data read outside io.read()"
            )
        if bad_ids:
            violations.append(f"{bad_ids} result ids out of range")
        if (pages < 0).any() or (rounds < 0).any():
            violations.append("negative page or round counts")
        integrity["violations"] = violations

        out_dir = index_path / "results"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{run_tag}_point_{point}.npz"
        np.savez(
            out_file,
            ids=ids,
            pages=pages,
            rounds=rounds,
            round_pages=np.asarray(round_pages, dtype=np.int32),
            lat=lat,
        )
        _chown_tree(out_dir, uid, gid)
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
                # {pages in a round: number of such rounds}, over all queries
                "round_pages_hist": {str(p): n for p, n in sorted(Counter(round_pages).items())},
                # Exactly the query loop (ann-suite's own figure comes from 100 ms
                # monitor samples, which can catch the end of the index load).
                "kernel_pages_per_query": (
                    (rbytes1 - rbytes0) / 4096 / nq if rbytes0 >= 0 and rbytes1 >= 0 else None
                ),
                "cpu_ms_per_query": cpu_s * 1000.0 / nq,
                "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "build_program_sha256": meta["program_sha256"],
                "integrity": integrity,
            },
        }
    except Exception as exc:
        import traceback

        if not isinstance(exc, CandidateError):
            traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "total_queries": 0,
            "total_time_seconds": 0,
            "qps": 0,
        }
    finally:
        if sandbox is not None:
            sandbox.close()
        if index_fd >= 0:
            os.close(index_fd)
        for fd, _n in disk_fds.values():
            os.close(fd)


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
            os.chown(results_dir / "metrics.json", *_host_owner())
    raise SystemExit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
