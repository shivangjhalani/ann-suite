"""Host and storage-device state probes recorded with every search point.

Absolute throughput of disk-resident indices depends on state that neither Docker nor the
algorithm controls: which flash mode (SLC cache vs QLC) the index file's pages currently sit
in, CPU frequency policy, thermal state. On a consumer QLC SSD, freshly written index files
were measured reading ~3x faster than the same bytes weeks later, which moved PipeANN QPS by
3-5x with identical code, config and CPU settings. These probes make that state visible so
points measured under different device states are never silently compared.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAGE = 4096
QD1_READS = 2000
FIO_SECONDS = 4


def _largest_file(directory: Path) -> Path | None:
    files = [p for p in directory.rglob("*") if p.is_file()]
    return max(files, key=lambda p: p.stat().st_size, default=None)


def _read_loop(fd: int, pages: int, stop: float | None, count: int | None) -> list[float]:
    """Random 4 KiB O_DIRECT reads; returns per-read latencies in seconds."""
    buf = mmap.mmap(-1, PAGE)  # page-aligned, as O_DIRECT requires
    rng = random.Random()
    lat: list[float] = []
    while (count is None or len(lat) < count) and (stop is None or time.perf_counter() < stop):
        start = time.perf_counter()
        os.preadv(fd, [buf], rng.randrange(pages) * PAGE)
        lat.append(time.perf_counter() - start)
    return lat


def _fio_qd64(target: Path) -> dict[str, Any]:
    """4 KiB QD64 io_uring random reads via fio (the depth PipeANN-class searches run at)."""
    if shutil.which("fio") is None:
        return {}
    cmd = [
        "fio", "--name=probe", f"--filename={target}", "--rw=randread", "--bs=4k",
        "--iodepth=64", "--ioengine=io_uring", "--direct=1", "--time_based",
        f"--runtime={FIO_SECONDS}", "--ramp_time=1", "--output-format=json",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FIO_SECONDS + 30)
        read = json.loads(proc.stdout)["jobs"][0]["read"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        logger.warning("fio device probe failed: %s", exc)
        return {}
    return {
        "probe_qd64_kiops": round(read["iops"] / 1000, 1),
        "probe_qd64_lat_us": round(read["clat_ns"]["mean"] / 1000, 1),
    }


def probe_device(index_dir: Path) -> dict[str, Any]:
    """Measure 4 KiB random-read speed on the index's largest file (O_DIRECT, no page cache).

    Reports QD1 median latency (pure Python) and, when fio is installed, QD64 IOPS and latency.
    The probe reflects the SSD state for exactly the bytes the search will read and costs a few
    seconds. Returns an empty dict when it cannot run.
    """
    target = _largest_file(index_dir)
    if target is None or target.stat().st_size < PAGE * 1024:
        return {}
    pages = target.stat().st_size // PAGE
    try:
        fd = os.open(target, os.O_RDONLY | os.O_DIRECT)
    except OSError as exc:
        logger.warning("Device probe skipped for %s: %s", target, exc)
        return {}
    try:
        qd1 = _read_loop(fd, pages, None, QD1_READS)
    finally:
        os.close(fd)
    return {
        "probe_file": target.name,
        "probe_qd1_p50_us": round(statistics.median(qd1) * 1e6, 1),
        **_fio_qd64(target),
    }


def _read_first(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_list(spec: str | None) -> list[int]:
    if not spec:
        return []
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def host_state(cpus: str | None = None) -> dict[str, Any]:
    """CPU governor/frequency, NVMe temperature and load; cheap and best-effort."""
    cpu_ids = _cpu_list(cpus) or [0]
    khz = [
        float(v)
        for c in cpu_ids
        if (v := _read_first(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"))
    ]
    nvme_temps = [
        int(v) / 1000
        for p in Path("/sys/class/nvme").glob("nvme*/hwmon*/temp1_input")
        if (v := _read_first(str(p)))
    ]
    state: dict[str, Any] = {
        "host_cpu_governor": _read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "host_cpu_epp": _read_first(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
        ),
        "host_load1": round(os.getloadavg()[0], 2),
    }
    if khz:
        state["host_cpu_mhz_mean"] = round(statistics.mean(khz) / 1000)
    if nvme_temps:
        state["host_nvme_temp_c"] = round(max(nvme_temps), 1)
    return {k: v for k, v in state.items() if v is not None}


def drift_warning(
    previous: dict[str, Any], current: dict[str, Any], tol: float = 0.25
) -> str | None:
    """Describe a device-state change large enough to confound point-to-point comparison."""
    for key in ("probe_qd1_p50_us", "probe_qd64_kiops"):
        prev, cur = previous.get(key), current.get(key)
        if prev and cur and abs(cur - prev) / prev > tol:
            return f"{key} moved {prev} -> {cur} (>{tol:.0%})"
    return None
