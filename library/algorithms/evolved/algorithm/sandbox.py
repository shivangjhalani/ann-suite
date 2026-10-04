"""Untrusted side of the evolved harness: runs candidate code as `nobody`.

Started by runner.py (root inside the container) with two pipe fds and, for a
build, an fd of the base vectors; the index directory arrives as an fd too. The
candidate therefore sees paths like /proc/self/fd/N and never /data itself,
which is mode 0700 on the host: it cannot open queries, other datasets or other
candidates' programs. The container has no network. Queries arrive one at a
time and the next is sent only after the previous answer, so a Searcher cannot
see the query set in advance.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from algorithm import wire
from algorithm.harness import PAGE, BuildContext, QueryIO, SearchContext

# Imported up front so the measured memory floor (a null program) includes the
# libraries candidates are expected to use.
with contextlib.suppress(ImportError):
    import faiss  # noqa: F401
with contextlib.suppress(ImportError):
    import numba  # noqa: F401


def _load_program(source: str) -> ModuleType:
    path = Path(tempfile.mkdtemp(prefix="candidate-")) / "program.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the candidate program")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fd_path(fd: int) -> Path:
    return Path(f"/proc/self/fd/{fd}")


def _build(msg: dict[str, Any]) -> None:
    data = np.load(_fd_path(msg["base_fd"]), mmap_mode="r")
    ctx = BuildContext(data, _fd_path(msg["index_fd"]), int(msg["threads"]), msg["metric"])
    _load_program(msg["source"]).build(ctx)
    ctx.close()


def _serve(rfd: int, wfd: int, msg: dict[str, Any]) -> None:
    ctx = SearchContext(
        _fd_path(msg["index_fd"]), int(msg["threads"]), msg["metric"], msg["disk_pages"]
    )

    def fetch(name: str, ids: np.ndarray) -> np.ndarray:
        wire.send(wfd, b"R", wire.pack_read(name, ids.astype("<i8").tobytes()))
        kind, payload = wire.recv(rfd, ids.size * PAGE)
        if kind != b"P" or len(payload) != ids.size * PAGE:
            raise OSError(f"page read of {name} failed")
        return np.frombuffer(payload, dtype=np.uint8).reshape(-1, PAGE)

    searcher = _load_program(msg["source"]).Searcher(ctx, dict(msg["params"]))
    wire.send_json(wfd, {"ok": True})
    k, dim, dtype = int(msg["k"]), int(msg["dim"]), np.dtype(msg["dtype"])
    while True:
        kind, payload = wire.recv(rfd, dim * dtype.itemsize)
        if kind == b"X":
            return
        query = np.frombuffer(payload, dtype=dtype).copy()
        io = QueryIO(ctx, fetch)
        res = np.asarray(searcher.search(query, k, io), dtype=np.int64).ravel()[:k]
        out = np.full(k, -1, dtype=np.int64)
        out[: res.size] = res
        wire.send(wfd, b"Q", out.tobytes() + struct.pack("<ii", io.pages, io.rounds))


def main() -> None:
    rfd, wfd = int(sys.argv[1]), int(sys.argv[2])
    msg = json.loads(wire.recv(rfd, 1 << 24)[1])
    try:
        if msg["cmd"] == "build":
            _build(msg)
            wire.send_json(wfd, {"ok": True})
        else:
            _serve(rfd, wfd, msg)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        with contextlib.suppress(OSError):
            wire.send_json(
                wfd,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                    "traceback": traceback.format_exc()[-4000:],
                },
            )


if __name__ == "__main__":
    main()
