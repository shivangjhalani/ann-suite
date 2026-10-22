"""Device state of the drive holding an index: what it is, and whether it is the state asked for.

``probe_device`` (host_state.py) measures how fast the index file reads right now. This module
adds what the probe cannot see: which drive and firmware the file sits on, how full the file
system is, how much the host has written to the drive, the drive's own temperature and the
index file's age. It also implements the ``device_state:`` acceptance gate.

A state is defined by its acceptance check, not by the procedure that reaches it: on the isfcr
QLC drive, writing and deleting ~100 GB gave *a* state, never the same one twice. So a state is
a band of probe values per drive (e.g. QD1 p50 within 25-40 us), and ``prepare`` only tries to
reach it; the point runs only if the probe then lands inside the band.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ann_suite.core.constants import SUDO_PASSWORD_ENV

logger = logging.getLogger(__name__)

SYSFS = Path("/sys")
GB = 1e9
SECTOR = 512
NVME_DATA_UNIT = 512_000  # NVMe SMART data units are 1000 x 512 bytes
TMP_SUFFIX = ".ann-suite-rewrite"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def block_device(path: Path, sysfs: Path = SYSFS) -> str | None:
    """Whole-disk block device (e.g. ``nvme0n1``) holding ``path``; None if unknown."""
    st_dev = path.stat().st_dev
    node = sysfs / "dev" / "block" / f"{os.major(st_dev)}:{os.minor(st_dev)}"
    try:
        real = node.resolve(strict=True)
    except OSError:
        return None
    if (real / "partition").exists():
        real = real.parent
    return real.name


def drive_record(path: Path, sysfs: Path = SYSFS) -> dict[str, Any]:
    """Drive identity, fill level, host writes, temperature and file age for ``path``.

    Cheap and best-effort: every key is omitted when it cannot be read. ``dev_smart_*`` needs
    root (``nvme smart-log``); see ``smart_log``.
    """
    rec: dict[str, Any] = {}
    usage = shutil.disk_usage(path)
    rec["dev_fs_used_pct"] = round(100 * usage.used / usage.total, 1)  # excl. root reserve
    rec["dev_fs_avail_gb"] = round(usage.free / GB, 1)
    rec["index_file_age_h"] = round((time.time() - path.stat().st_mtime) / 3600, 2)

    disk = block_device(path, sysfs)
    if disk is None:
        return rec
    base = sysfs / "block" / disk
    rec["dev_name"] = disk
    for key, rel in (("dev_model", "device/model"), ("dev_firmware", "device/firmware_rev")):
        if value := _read(base / rel):
            rec[key] = value
    if stat := _read(base / "stat"):
        fields = stat.split()
        if len(fields) > 6:
            rec["dev_written_since_boot_gb"] = round(int(fields[6]) * SECTOR / GB, 1)
    temps = [
        int(v) / 1000
        for p in sorted(base.glob("device/hwmon*/temp1_input"))
        if (v := _read(p)) is not None
    ]
    if temps:
        rec["dev_temp_c"] = temps[0]

    controller = (base / "device").resolve().name if (base / "device").exists() else None
    if controller and controller.startswith("nvme"):
        rec.update(smart_log(controller))
    return rec


def _run_privileged(cmd: list[str]) -> str | None:
    """Run ``cmd`` directly, then via passwordless sudo, then with ANN_SUITE_SUDO_PASSWORD."""
    attempts: list[tuple[list[str], str | None]] = [(cmd, None), (["sudo", "-n", *cmd], None)]
    if (password := os.environ.get(SUDO_PASSWORD_ENV)) is not None:
        attempts.append((["sudo", "-S", "-p", "", *cmd], password + "\n"))
    for argv, stdin in attempts:
        try:
            proc = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return proc.stdout
    return None


def smart_log(controller: str) -> dict[str, Any]:
    """Host bytes written, wear and temperature from the NVMe SMART log; {} without root."""
    if shutil.which("nvme") is None:
        return {}
    out = _run_privileged(["nvme", "smart-log", f"/dev/{controller}", "-o", "json"])
    if out is None:
        logger.debug("nvme smart-log unavailable (needs root or a sudoers rule)")
        return {}
    try:
        log = json.loads(out)
    except ValueError:
        return {}
    rec: dict[str, Any] = {}
    if (units := log.get("data_units_written")) is not None:
        rec["dev_smart_written_gb"] = round(int(units) * NVME_DATA_UNIT / GB, 1)
    used = log.get("percent_used", log.get("percentage_used"))
    if used is not None:
        rec["dev_smart_percent_used"] = used
    if (kelvin := log.get("temperature")) is not None:
        rec["dev_smart_temp_c"] = int(kelvin) - 273
    return rec


def in_bands(probe: dict[str, Any], bands: dict[str, tuple[float | None, float | None]]) -> bool:
    """True when every banded probe value is present and within its [low, high] band."""
    for key, (low, high) in bands.items():
        value = probe.get(key)
        if value is None:
            return False
        if (low is not None and value < low) or (high is not None and value > high):
            return False
    return True


def wait_for_state(
    probe: Callable[[], dict[str, Any]],
    bands: dict[str, tuple[float | None, float | None]],
    max_waits: int,
    wait_s: float,
) -> tuple[dict[str, Any], int]:
    """Probe until the values fall in ``bands`` or ``max_waits`` re-probes are spent.

    Dips on this drive were often transient (QD64 fell from ~150 to ~25 kIOPS mid-session and
    recovered), so a miss is re-probed after a pause before the point is refused.
    """
    result = probe()
    waits = 0
    while not in_bands(result, bands) and waits < max_waits:
        waits += 1
        time.sleep(wait_s)
        result = probe()
    return result, waits


def _real_files(index_dir: Path) -> list[Path]:
    """Real files of an index dir, following symlinks to where the bytes live (deduplicated)."""
    files = {p.resolve() for p in index_dir.resolve().rglob("*") if p.is_file()}
    return sorted(f for f in files if not f.name.endswith(TMP_SUFFIX))


def rewrite_files(index_dir: Path, min_free_gb: float) -> float:
    """Make the index freshly written: copy each file, fsync, rename over the original.

    Content is unchanged; only the flash location (and mtime) change. The original stays in
    place until its copy is complete, so an interrupted run leaves at most a ``*.ann-suite-
    rewrite`` temp file. Returns GB written.
    """
    files = _real_files(index_dir)
    if not files:
        raise RuntimeError(f"no files to rewrite under {index_dir}")
    written = 0
    linked = [f for f in files if f.stat().st_nlink > 1]
    if linked:  # renaming over a hard link would leave the other names on the old copy
        raise RuntimeError(f"refusing to rewrite hard-linked files: {linked}")
    for src in files:
        size = src.stat().st_size
        free = shutil.disk_usage(src.parent).free
        if free - size < min_free_gb * GB:
            raise RuntimeError(
                f"rewriting {src} ({size / GB:.1f} GB) would leave less than "
                f"{min_free_gb} GB free on its file system ({free / GB:.1f} GB free now)"
            )
        tmp = src.with_name(src.name + TMP_SUFFIX)
        try:
            shutil.copyfile(src, tmp)
            shutil.copymode(src, tmp)
            with tmp.open("rb+") as f:
                os.fsync(f.fileno())
            if tmp.stat().st_size != size:
                raise RuntimeError(f"short copy of {src}")
            os.replace(tmp, src)
        finally:
            tmp.unlink(missing_ok=True)
        written += size
    return written / GB


def write_and_delete(filler_dir: Path, gb: float, min_free_gb: float, chunk_mb: int = 64) -> None:
    """Write ``gb`` of incompressible data to ``filler_dir``, fsync, then delete it.

    Used to push freshly written index pages out of the SLC cache. The file system is not
    trimmed afterwards (that needs root), so the drive still sees those LBAs as valid until the
    next fstrim; the acceptance check, not this step, decides whether the state was reached.
    """
    filler_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(filler_dir).free
    if free - gb * GB < min_free_gb * GB:
        raise RuntimeError(
            f"writing {gb} GB to {filler_dir} would leave less than {min_free_gb} GB free "
            f"({free / GB:.1f} GB free now)"
        )
    chunk = mmap.mmap(-1, chunk_mb << 20)
    chunk.write(os.urandom(chunk_mb << 20))
    target = filler_dir / "ann-suite-filler.bin"
    chunks = int(gb * GB) // (chunk_mb << 20)
    try:
        with target.open("wb") as f:
            for i in range(chunks):
                f.write(chunk)
                if i % 16 == 15:  # bound dirty page cache: flush and drop every ~1 GB
                    f.flush()
                    os.fsync(f.fileno())
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            f.flush()
            os.fsync(f.fileno())
    finally:
        target.unlink(missing_ok=True)
        os.sync()
