"""The evolved runner's AIO page reader returns the same bytes as plain preads."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "library" / "algorithms" / "evolved"))
from algorithm.aio import BATCH, AioReader  # noqa: E402
from algorithm.harness import PAGE  # noqa: E402

PAGES = 64


def open_direct(tmp: Path) -> int:
    path = tmp / "pages.bin"
    rng = np.random.default_rng(0)
    path.write_bytes(rng.integers(0, 256, PAGES * PAGE, dtype=np.uint8).tobytes())
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as exc:  # tmpfs and some file systems refuse O_DIRECT
        pytest.skip(f"O_DIRECT unsupported here: {exc}")


@pytest.mark.parametrize("n", [1, 7, BATCH + 3])
def test_aio_matches_pread(tmp_path: Path, n: int) -> None:
    fd = open_direct(tmp_path)
    ids = np.random.default_rng(n).integers(0, PAGES, n)
    reader = AioReader()
    try:
        got = bytes(reader.read(fd, ids))
        data = (tmp_path / "pages.bin").read_bytes()
        want = b"".join(data[int(i) * PAGE : (int(i) + 1) * PAGE] for i in ids)
        assert got == want
        assert bytes(reader.read(fd, ids[:1])) == want[:PAGE]  # buffer reuse after a larger round
    finally:
        reader.close()
        os.close(fd)
