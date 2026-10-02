"""Tests for host/device state probes."""

from __future__ import annotations

from pathlib import Path

from ann_suite.monitoring.host_state import (
    _cpu_list,
    drift_warning,
    host_state,
    probe_device,
)


class TestProbeDevice:
    def test_missing_or_small_dir_returns_empty(self, tmp_path: Path) -> None:
        assert probe_device(tmp_path) == {}
        (tmp_path / "tiny.bin").write_bytes(b"x" * 100)
        assert probe_device(tmp_path) == {}

    def test_probe_reports_latency_and_throughput(self, tmp_path: Path) -> None:
        (tmp_path / "idx.bin").write_bytes(b"\0" * (8 * 1024 * 1024))
        out = probe_device(tmp_path)
        if not out:  # filesystem without O_DIRECT (e.g. tmpfs)
            return
        assert out["probe_file"] == "idx.bin"
        assert out["probe_qd1_p50_us"] > 0


class TestHostState:
    def test_cpu_list_parsing(self) -> None:
        assert _cpu_list("0-2,5") == [0, 1, 2, 5]
        assert _cpu_list(None) == []

    def test_host_state_has_load(self) -> None:
        assert "host_load1" in host_state("0-1")


class TestDriftWarning:
    def test_flags_large_change_only(self) -> None:
        base = {"probe_qd1_p50_us": 100.0, "probe_qd64_kiops": 400.0}
        assert drift_warning(base, {"probe_qd1_p50_us": 110.0, "probe_qd64_kiops": 390.0}) is None
        assert drift_warning(base, {"probe_qd1_p50_us": 250.0, "probe_qd64_kiops": 390.0})
        assert drift_warning(base, {"probe_qd1_p50_us": 100.0, "probe_qd64_kiops": 150.0})
