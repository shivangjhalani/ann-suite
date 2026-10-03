"""Fixed harness API for evolved disk-ANN programs (not evolved itself).

An evolved program is one Python file that defines:

    SEARCH_POINTS: list[dict]          # literal list; one benchmark point per entry
    def build(ctx: BuildContext) -> None
    class Searcher:
        def __init__(self, ctx: SearchContext, params: dict) -> None
        def search(self, query: np.ndarray, k: int, io: QueryIO) -> np.ndarray  # k ids

Programs must not assume a dimension, dtype or metric: ctx.data is (N, D) uint8
(BIGANN) or float32 (DEEP, T2I held-out sets), ctx.metric is "L2" or "IP", and
queries have the base vectors' dtype.

Rules the harness enforces or measures:
- Disk-resident data is written with ``ctx.disk_writer(name)`` (whole 4 KB pages)
  and read at query time only through ``io.read(name, page_ids)``. Each call is one
  I/O round; every distinct page it returns is one 4 KB O_DIRECT device read. The
  host cross-checks these counts against the kernel's per-container io.stat.
- Everything kept in DRAM is whatever the Searcher holds after __init__ plus what it
  allocates while searching; the suite measures the container's peak anonymous
  memory, so there is no separate DRAM declaration to fill in.
- ``ctx.mem(name)`` loads an array saved at build time with ``ctx.save_mem`` fully
  into RAM (counted as DRAM); it is the only way to read memory-resident index
  files.
- Before the first query the runner evicts every file page from the page cache, so
  data cannot be smuggled into DRAM through the page cache.
"""

from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np

PAGE = 4096


class PageWriter:
    """Append-only writer of whole 4 KB pages to a disk-resident index file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = path.open("wb")
        self.num_pages = 0

    def write_pages(self, pages: np.ndarray) -> int:
        """Append pages (uint8 array of shape (n, 4096)); return the first page id."""
        pages = np.ascontiguousarray(pages, dtype=np.uint8)
        if pages.ndim != 2 or pages.shape[1] != PAGE:
            raise ValueError(f"pages must have shape (n, {PAGE}), got {pages.shape}")
        first = self.num_pages
        self._f.write(pages.tobytes())
        self.num_pages += pages.shape[0]
        return first

    def close(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()


class BuildContext:
    """What build() gets: the base vectors, a thread budget and index writers."""

    def __init__(self, data: np.ndarray, index_dir: Path, threads: int, metric: str) -> None:
        self.data = data  # (N, D) read-only memmap; uint8 or float32 depending on dataset
        self.metric = metric  # "L2" or "IP" (maximize inner product)
        self.threads = threads
        self._dir = index_dir
        self._writers: list[PageWriter] = []
        (index_dir / "disk").mkdir(parents=True, exist_ok=True)
        (index_dir / "mem").mkdir(parents=True, exist_ok=True)

    def disk_writer(self, name: str) -> PageWriter:
        w = PageWriter(self._dir / "disk" / f"{name}.pages")
        self._writers.append(w)
        return w

    def save_mem(self, name: str, array: np.ndarray) -> None:
        np.save(self._dir / "mem" / f"{name}.npy", np.ascontiguousarray(array))

    def save_json(self, name: str, obj: Any) -> None:
        (self._dir / "mem" / f"{name}.json").write_text(json.dumps(obj))

    def close(self) -> None:
        for w in self._writers:
            if not w._f.closed:
                w.close()


class _DiskFile:
    def __init__(self, path: Path) -> None:
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.num_pages = os.fstat(self.fd).st_size // PAGE


class SearchContext:
    """What Searcher.__init__ gets: memory-resident arrays and disk files."""

    def __init__(self, index_dir: Path, threads: int, metric: str) -> None:
        self.threads = threads
        self.metric = metric
        self._dir = index_dir
        self._disk: dict[str, _DiskFile] = {}
        for p in sorted((index_dir / "disk").glob("*.pages")):
            self._disk[p.stem] = _DiskFile(p)

    def mem(self, name: str) -> np.ndarray:
        return np.load(self._dir / "mem" / f"{name}.npy")  # fully in RAM, no mmap

    def json(self, name: str) -> Any:
        return json.loads((self._dir / "mem" / f"{name}.json").read_text())

    def num_pages(self, name: str) -> int:
        return self._disk[name].num_pages


class QueryIO:
    """Per-query I/O handle; the only way to touch disk-resident pages."""

    def __init__(self, ctx: SearchContext) -> None:
        self._ctx = ctx
        self.pages = 0
        self.rounds = 0
        self._buf = mmap.mmap(-1, PAGE * 64)  # page-aligned, grown on demand

    def read(self, name: str, page_ids: Any) -> np.ndarray:
        """Read pages of disk file `name` in one I/O round.

        Returns a uint8 array of shape (len(unique ids), 4096) in the order of
        np.unique(page_ids) (sorted ascending). Duplicate ids are read once.
        """
        f = self._ctx._disk[name]
        ids = np.unique(np.asarray(page_ids, dtype=np.int64).ravel())
        if ids.size == 0:
            return np.empty((0, PAGE), dtype=np.uint8)
        if ids[0] < 0 or ids[-1] >= f.num_pages:
            raise IndexError(f"page id out of range for {name} ({f.num_pages} pages)")
        need = ids.size * PAGE
        if len(self._buf) < need:
            self._buf.close()
            self._buf = mmap.mmap(-1, need)
        view = memoryview(self._buf)
        for j, pid in enumerate(ids.tolist()):
            n = os.preadv(f.fd, [view[j * PAGE : (j + 1) * PAGE]], pid * PAGE)
            if n != PAGE:
                raise OSError(f"short read on {name} page {pid}")
        self.pages += int(ids.size)
        self.rounds += 1
        return np.frombuffer(self._buf, dtype=np.uint8, count=need).reshape(-1, PAGE).copy()
