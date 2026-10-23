"""Concurrent O_DIRECT page reads with Linux AIO, for the trusted runner.

By default the runner serves an io.read() round with one preadv per page, one after
another, so a candidate's wall-clock latency is not what a deployment would see.
With search_args.aio the runner submits every page of a round at once (io_submit)
and waits for all of them, as Starling/DiskANN/PipeANN do, so measured per-query
latency includes real device parallelism. Same structs as tools/evolve/e2e/e2e_bench.py.
"""

from __future__ import annotations

import ctypes
import mmap

import numpy as np

from algorithm.harness import PAGE

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
BATCH = 1024  # pages in flight at once; larger rounds are read in batches


class AioReader:
    """One AIO context and a page-aligned buffer that grows to the largest round."""

    def __init__(self) -> None:
        self.ctx = ctypes.c_ulong(0)
        if _libc.syscall(_SYS_IO_SETUP, BATCH, ctypes.byref(self.ctx)) != 0:
            raise OSError(ctypes.get_errno(), "io_setup")
        self.iocbs = np.zeros(BATCH, _IOCB)
        self.ptrs = self.iocbs.ctypes.data + np.arange(BATCH, dtype=np.uint64) * 64
        self.events = np.zeros(BATCH, _EVENT)
        self.mm: mmap.mmap | None = None
        self.base = 0

    def _ensure(self, n: int) -> None:
        if self.mm is not None and len(self.mm) >= n * PAGE:
            return
        if self.mm is not None:
            self.mm.close()
        self.mm = mmap.mmap(-1, max(n, 64) * PAGE)  # page aligned, for O_DIRECT
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def read(self, fd: int, ids: np.ndarray) -> memoryview:
        """Pages `ids` of `fd`, in the order given, as one contiguous view."""
        n = ids.size
        self._ensure(n)
        for s in range(0, n, BATCH):
            self._batch(fd, ids[s : s + BATCH], s)
        assert self.mm is not None
        return memoryview(self.mm)[: n * PAGE]

    def _batch(self, fd: int, ids: np.ndarray, first: int) -> None:
        n = ids.size
        cb = self.iocbs[:n]
        cb["opcode"] = 0  # IOCB_CMD_PREAD
        cb["fildes"] = fd
        cb["buf"] = self.base + (first + np.arange(n, dtype=np.uint64)) * PAGE
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

    def close(self) -> None:
        _libc.syscall(_SYS_IO_DESTROY, self.ctx)
        if self.mm is not None:
            self.mm.close()
