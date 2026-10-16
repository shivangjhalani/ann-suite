"""End-to-end measurement of an evolved-harness program with real, concurrent reads.

The scoring harness serves io.read() one page at a time (its latency is modelled),
so this measures what a deployment would see: every io.read() round is submitted at
once with Linux AIO (O_DIRECT, so never from the page cache, as Starling/DiskANN
do), each query is timed wall-clock, and N threads share one loaded index for
throughput. Run inside the ann-suite/evolved image (faiss, numba, the harness):

  docker run --rm --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
    -v <ann-suite>:/repo:ro -v <index dir>:/idx:ro -v <data dir>:/data:ro \
    ann-suite/evolved:latest /repo/tools/evolve/e2e/e2e_bench.py \
    --program /repo/tools/evolve/ablation/abl_fastscan_h40.py --index /idx \
    --queries /data/bigann10m-q2k/queries.npy --gt /data/bigann10m-q2k/ground_truth.npy \
    --points '[{"nprobe": 64, "shortlist": 768, "pages": 16}]' --threads 1

Prints one JSON line per point: recall@10, pages and rounds per query, latency
(mean, p50, p99, p99.9 ms) at --threads 1 or QPS at --threads N, CPU ms per query, and
the process's peak resident and anonymous memory.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import mmap
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from algorithm.harness import PAGE, QueryIO, SearchContext  # noqa: E402

_libc = ctypes.CDLL(None, use_errno=True)
_SYS_IO_SETUP, _SYS_IO_DESTROY, _SYS_IO_GETEVENTS, _SYS_IO_SUBMIT = 206, 207, 208, 209  # x86_64
_IOCB = np.dtype(
    [
        ("data", "<u8"),
        ("key", "<u4"),
        ("rw_flags", "<u4"),
        ("opcode", "<u2"),
        ("reqprio", "<i2"),
        ("fildes", "<u4"),
        ("buf", "<u8"),
        ("nbytes", "<u8"),
        ("offset", "<i8"),
        ("reserved2", "<u8"),
        ("flags", "<u4"),
        ("resfd", "<u4"),
    ]
)  # fmt: skip  struct iocb, 64 bytes
_EVENT = np.dtype([("data", "<u8"), ("obj", "<u8"), ("res", "<i8"), ("res2", "<i8")])
MAX_PAGES = 1024


class AioReader:
    """One thread's AIO context: reads a sorted set of pages in one submit + wait."""

    def __init__(self, fds: dict[str, int]) -> None:
        self.fds = fds
        self.ctx = ctypes.c_ulong(0)
        if _libc.syscall(_SYS_IO_SETUP, MAX_PAGES, ctypes.byref(self.ctx)) != 0:
            raise OSError(ctypes.get_errno(), "io_setup")
        self.mm = mmap.mmap(-1, MAX_PAGES * PAGE)  # page aligned, for O_DIRECT
        self.buf = np.frombuffer(self.mm, dtype=np.uint8).reshape(MAX_PAGES, PAGE)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        self.iocbs = np.zeros(MAX_PAGES, _IOCB)
        self.ptrs = self.iocbs.ctypes.data + np.arange(MAX_PAGES, dtype=np.uint64) * 64
        self.events = np.zeros(MAX_PAGES, _EVENT)

    def fetch(self, name: str, ids: np.ndarray) -> np.ndarray:
        n = ids.size
        if n > MAX_PAGES:
            raise ValueError(f"{n} pages in one round (max {MAX_PAGES})")
        cb = self.iocbs[:n]
        cb["opcode"] = 0  # IOCB_CMD_PREAD
        cb["fildes"] = self.fds[name]
        cb["buf"] = self.base + np.arange(n, dtype=np.uint64) * PAGE
        cb["nbytes"] = PAGE
        cb["offset"] = ids.astype(np.int64) * PAGE
        done = 0
        while done < n:  # io_submit may accept fewer than asked
            r = _libc.syscall(_SYS_IO_SUBMIT, self.ctx, ctypes.c_long(n - done),
                              ctypes.c_void_p(self.ptrs.ctypes.data + done * 8))  # fmt: skip
            if r < 0:
                raise OSError(ctypes.get_errno(), "io_submit")
            done += r
        got = 0
        while got < n:
            r = _libc.syscall(_SYS_IO_GETEVENTS, self.ctx, ctypes.c_long(1), ctypes.c_long(n - got),
                              ctypes.c_void_p(self.events.ctypes.data + got * 32), None)  # fmt: skip
            if r < 0:
                err = ctypes.get_errno()
                if err == 4:  # EINTR
                    continue
                raise OSError(err, "io_getevents")
            got += r
        if (self.events["res"][:n] != PAGE).any():
            raise OSError("short or failed AIO read")
        return self.buf[:n]  # rows follow ids (buffers were assigned in id order)

    def close(self) -> None:
        _libc.syscall(_SYS_IO_DESTROY, self.ctx)


def mem_mb() -> dict[str, float]:
    st = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("VmHWM", "VmRSS", "RssAnon"):
            st[k] = round(int(v.split()[0]) / 1024, 1)
    return st


def load_program(path: str):
    ns: dict = {"__name__": "evolved_program"}
    exec(compile(Path(path).read_text(), path, "exec"), ns)  # noqa: S102
    return ns


def run(searcher, ctx, readers, queries, k):
    """Run all queries on len(readers) threads sharing `searcher`; per-query arrays."""
    nq = len(queries)
    res = np.full((nq, k), -1, dtype=np.int64)
    lat, cpu, npages, nrounds = (np.zeros(nq) for _ in range(4))
    counter = itertools.count()

    def worker(rd: AioReader) -> None:
        while (i := next(counter)) < nq:
            io = QueryIO(ctx, rd.fetch)
            c0, w0 = time.thread_time(), time.perf_counter()
            ids = np.asarray(searcher.search(queries[i], k, io), dtype=np.int64)[:k]
            lat[i] = time.perf_counter() - w0
            cpu[i] = time.thread_time() - c0
            res[i, : ids.size] = ids
            npages[i], nrounds[i] = io.pages, io.rounds

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(rd,)) for rd in readers]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return res, lat, cpu, npages, nrounds, time.perf_counter() - t0


def run_procs(searcher, ctx, fds, queries, k, procs):
    """As run(), on `procs` forked single-thread workers sharing the loaded index
    copy-on-write (no GIL between workers); each opens its own AIO context."""
    import multiprocessing as mp

    nq = len(queries)
    mpc = mp.get_context("fork")
    out = mpc.Queue()
    go = mpc.Event()

    def child(rank: int) -> None:
        rd = AioReader(fds)
        mine = np.arange(rank, nq, procs)
        sub = queries[mine]
        go.wait()
        r = run(searcher, ctx, [rd], sub, k)
        out.put((mine, *r[:5]))

    ps = [mpc.Process(target=child, args=(r,)) for r in range(procs)]
    for p in ps:
        p.start()
    time.sleep(1.0)  # let every worker reach the start line
    t0 = time.perf_counter()
    go.set()
    parts = [out.get() for _ in ps]
    wall = time.perf_counter() - t0
    for p in ps:
        p.join()
    res = np.full((nq, k), -1, dtype=np.int64)
    lat, cpu, npages, nrounds = (np.zeros(nq) for _ in range(4))
    for mine, r_res, r_lat, r_cpu, r_pg, r_rd in parts:
        res[mine], lat[mine], cpu[mine], npages[mine], nrounds[mine] = (
            r_res,
            r_lat,
            r_cpu,
            r_pg,
            r_rd,
        )
    return res, lat, cpu, npages, nrounds, wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", required=True)
    ap.add_argument("--index", required=True, help="built index dir (mem/, disk/)")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--points", required=True, help="JSON list of SEARCH_POINTS dicts")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--procs", type=int, default=0, help="forked workers instead of threads")
    ap.add_argument("--metric", default="L2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--dump", help="directory for per-query arrays (pN.npz: latency ms, cpu ms, pages, recall)"
    )
    a = ap.parse_args()

    idx = Path(a.index)
    pages = {p.stem: p.stat().st_size // PAGE for p in (idx / "disk").glob("*.pages")}
    fds = {n: os.open(idx / "disk" / f"{n}.pages", os.O_RDONLY | os.O_DIRECT) for n in pages}
    queries = np.load(a.queries)
    gt = np.load(a.gt)[:, : a.k]
    prog = load_program(a.program)
    mem0 = mem_mb()

    for pi, params in enumerate(json.loads(a.points)):
        ctx = SearchContext(idx, a.threads, a.metric, pages)
        searcher = prog["Searcher"](ctx, params)
        readers = [AioReader(fds) for _ in range(a.threads)]
        # untimed warm-up (compiles numba kernels), as the scoring harness does
        warm = queries[: min(16, len(queries))].astype(np.float64).mean(axis=0)
        if np.issubdtype(queries.dtype, np.integer):
            warm = warm.round()
        searcher.search(np.ascontiguousarray(warm.astype(queries.dtype)), a.k,
                        QueryIO(ctx, readers[0].fetch))  # fmt: skip

        if a.procs:
            res, lat, cpu, npages, nrounds, wall = run_procs(
                searcher, ctx, fds, queries, a.k, a.procs
            )
        else:
            res, lat, cpu, npages, nrounds, wall = run(searcher, ctx, readers, queries, a.k)
        for rd in readers:
            rd.close()
        nq = len(queries)
        rq = np.array([len(set(res[i]) & set(gt[i])) / a.k for i in range(nq)])
        recall = float(rq.mean())
        if a.dump:
            Path(a.dump).mkdir(parents=True, exist_ok=True)
            np.savez(Path(a.dump) / f"p{pi}.npz", params=json.dumps(params), lat_ms=lat * 1e3,
                     cpu_ms=cpu * 1e3, pages=npages, recall=rq)  # fmt: skip
        out = {
            "params": params, "threads": a.threads, "procs": a.procs, "queries": nq, "recall": round(recall, 4),
            "pages": round(float(npages.mean()), 2), "rounds": round(float(nrounds.mean()), 2),
            "qps": round(nq / wall, 1),
            "lat_mean_ms": round(float(lat.mean()) * 1e3, 3),
            "lat_p50_ms": round(float(np.percentile(lat, 50)) * 1e3, 3),
            "lat_p99_ms": round(float(np.percentile(lat, 99)) * 1e3, 3),
            "lat_p999_ms": round(float(np.percentile(lat, 99.9)) * 1e3, 3),
            "cpu_ms": round(float(cpu.mean()) * 1e3, 3),
            "mem_mb_start": mem0, "mem_mb": mem_mb(),
            "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }  # fmt: skip
        print(json.dumps(out), flush=True)
        del searcher


if __name__ == "__main__":
    main()
