"""Framing for the pipes between the trusted runner and the candidate sandbox.

Every message is a 1-byte kind, a 4-byte little-endian length and the payload.
Kinds: b"J" JSON control message, b"Q" query (runner -> sandbox: raw query bytes;
sandbox -> runner: k int64 ids + int32 pages + int32 rounds), b"X" exit. The
runner never unpickles anything the sandbox sends: replies are JSON or fixed-size
binary, and every length is bounded before it is read.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any

HEADER = struct.Struct("<cI")
MAX_CONTROL = 1 << 20


def _read_exact(fd: int, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError("sandbox pipe closed")
        buf += chunk
    return bytes(buf)


def send(fd: int, kind: bytes, payload: bytes = b"") -> None:
    view = memoryview(HEADER.pack(kind, len(payload)) + payload)
    while view:
        view = view[os.write(fd, view) :]


def recv(fd: int, max_len: int) -> tuple[bytes, bytes]:
    kind, n = HEADER.unpack(_read_exact(fd, HEADER.size))
    if n > max_len:
        raise ValueError(f"message of {n} bytes exceeds the {max_len}-byte limit")
    return kind, _read_exact(fd, n)


def send_json(fd: int, obj: Any) -> None:
    send(fd, b"J", json.dumps(obj).encode())


def recv_json(fd: int, max_len: int = MAX_CONTROL) -> Any:
    kind, payload = recv(fd, max_len)
    if kind != b"J":
        raise ValueError(f"expected a control message, got kind {kind!r}")
    return json.loads(payload)
