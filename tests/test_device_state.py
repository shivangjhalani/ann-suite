"""Tests for the drive record, the device_state gate, and its prepare steps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from ann_suite.core.schemas import AlgorithmType, DeviceStateConfig
from ann_suite.evaluator import _BuildContext
from ann_suite.monitoring import device_state as ds
from tests.test_build_sweeps import make_algo, make_dataset, make_evaluator, make_phase_result

FRESH = {"probe_qd1_p50_us": (20.0, 40.0)}


def fake_sysfs(root: Path, data_file: Path) -> Path:
    """A /sys tree in which ``data_file``'s device is partition 2 of nvme0n1 on nvme0."""
    sysfs = root / "sys"
    ctrl = sysfs / "devices" / "pci" / "nvme" / "nvme0"
    part = ctrl / "nvme0n1" / "nvme0n1p2"
    part.mkdir(parents=True)
    (part / "partition").write_text("2\n")
    (ctrl / "model").write_text("WD Blue SN5100 2TB   \n")
    (ctrl / "firmware_rev").write_text("3526M0WD\n")
    (ctrl / "hwmon1").mkdir()
    (ctrl / "hwmon1" / "temp1_input").write_text("41850\n")
    block = sysfs / "block" / "nvme0n1"
    block.mkdir(parents=True)
    (block / "device").symlink_to(ctrl)
    (block / "stat").write_text("1 2 3 4 5 6 2000000 8 9 10 11\n")
    st_dev = data_file.stat().st_dev
    (sysfs / "dev" / "block").mkdir(parents=True)
    (sysfs / "dev" / "block" / f"{os.major(st_dev)}:{os.minor(st_dev)}").symlink_to(part)
    return sysfs


class TestDriveRecord:
    def test_reads_identity_fill_writes_temperature_and_age(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "idx.bin"
        f.write_bytes(b"x")
        monkeypatch.setattr(ds, "smart_log", lambda controller: {"dev_smart_written_gb": 1.0})
        rec = ds.drive_record(f, fake_sysfs(tmp_path, f))
        assert rec["dev_name"] == "nvme0n1"
        assert rec["dev_model"] == "WD Blue SN5100 2TB"
        assert rec["dev_firmware"] == "3526M0WD"
        assert rec["dev_temp_c"] == 41.85
        assert rec["dev_written_since_boot_gb"] == round(2_000_000 * 512 / 1e9, 1)
        assert rec["dev_smart_written_gb"] == 1.0
        assert 0 <= rec["dev_fs_used_pct"] <= 100
        assert rec["index_file_age_h"] < 1

    def test_unknown_device_still_reports_fill_and_age(self, tmp_path: Path) -> None:
        f = tmp_path / "idx.bin"
        f.write_bytes(b"x")
        rec = ds.drive_record(f, tmp_path / "no-sysfs")
        assert set(rec) == {"dev_fs_used_pct", "dev_fs_avail_gb", "index_file_age_h"}

    def test_smart_log_parses_nvme_cli_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log = {"data_units_written": 2_000_000, "percent_used": 3, "temperature": 315}
        monkeypatch.setattr(ds.shutil, "which", lambda name: "/usr/sbin/nvme")
        monkeypatch.setattr(ds, "_run_privileged", lambda cmd: json.dumps(log))
        assert ds.smart_log("nvme0") == {
            "dev_smart_written_gb": 1024.0,
            "dev_smart_percent_used": 3,
            "dev_smart_temp_c": 42,
        }

    def test_smart_log_without_root_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ds.shutil, "which", lambda name: "/usr/sbin/nvme")
        monkeypatch.setattr(ds, "_run_privileged", lambda cmd: None)
        assert ds.smart_log("nvme0") == {}


class TestBands:
    def test_in_bands_open_sides_and_missing_values(self) -> None:
        assert ds.in_bands({"probe_qd1_p50_us": 30.0}, FRESH)
        assert not ds.in_bands({"probe_qd1_p50_us": 45.0}, FRESH)
        assert not ds.in_bands({}, FRESH)
        assert ds.in_bands({"probe_qd64_kiops": 900.0}, {"probe_qd64_kiops": (300.0, None)})

    def test_wait_for_state_reprobes_until_in_band_or_out_of_waits(self) -> None:
        values = iter([90.0, 80.0, 30.0])
        probe, waits = ds.wait_for_state(
            lambda: {"probe_qd1_p50_us": next(values)}, FRESH, max_waits=4, wait_s=0
        )
        assert probe["probe_qd1_p50_us"] == 30.0 and waits == 2
        probe, waits = ds.wait_for_state(
            lambda: {"probe_qd1_p50_us": 90.0}, FRESH, max_waits=1, wait_s=0
        )
        assert waits == 1 and not ds.in_bands(probe, FRESH)


class TestPrepare:
    def test_rewrite_keeps_bytes_follows_links_and_renews_inode(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        (store / "graph.bin").write_bytes(b"abc" * 1000)
        idx = tmp_path / "idx"
        idx.mkdir()
        (idx / "graph.bin").symlink_to(store / "graph.bin")
        inode = (store / "graph.bin").stat().st_ino
        gb = ds.rewrite_files(idx, min_free_gb=0)
        assert gb == pytest.approx(3000 / 1e9)
        assert (idx / "graph.bin").is_symlink()
        assert (store / "graph.bin").read_bytes() == b"abc" * 1000
        assert (store / "graph.bin").stat().st_ino != inode
        assert not list(store.glob("*" + ds.TMP_SUFFIX))

    def test_rewrite_refuses_hard_links_and_low_space(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"x")
        os.link(tmp_path / "a.bin", tmp_path / "b.bin")
        with pytest.raises(RuntimeError, match="hard-linked"):
            ds.rewrite_files(tmp_path, min_free_gb=0)
        (tmp_path / "b.bin").unlink()
        with pytest.raises(RuntimeError, match="free"):
            ds.rewrite_files(tmp_path, min_free_gb=1e9)

    def test_write_and_delete_leaves_nothing(self, tmp_path: Path) -> None:
        ds.write_and_delete(tmp_path / "filler", gb=0.004, min_free_gb=0, chunk_mb=1)
        assert list((tmp_path / "filler").iterdir()) == []
        with pytest.raises(RuntimeError, match="free"):
            ds.write_and_delete(tmp_path / "filler", gb=1, min_free_gb=1e9)


class TestDeviceStateConfig:
    def test_default_records_only(self) -> None:
        assert not DeviceStateConfig().active

    def test_target_needs_bands(self) -> None:
        with pytest.raises(ValidationError, match="defines no probe bands"):
            DeviceStateConfig(target="fresh")

    def test_rejects_unknown_probe_and_inverted_band(self) -> None:
        with pytest.raises(ValidationError, match="unknown probe"):
            DeviceStateConfig(states={"fresh": {"qps": (1, 2)}})
        with pytest.raises(ValidationError, match="low > high"):
            DeviceStateConfig(states={"fresh": {"probe_qd1_p50_us": (40, 20)}})

    def test_folded_prepare_needs_filler_dir(self) -> None:
        with pytest.raises(ValidationError, match="filler_dir"):
            DeviceStateConfig(
                target="folded", states={"folded": {"probe_qd1_p50_us": (70, None)}}, prepare=True
            )
        cfg = DeviceStateConfig(target="fresh", states={"fresh": FRESH})
        assert cfg.active and cfg.bands == FRESH

    def test_size_classes_pick_bands_by_file_size(self) -> None:
        small = {"probe_qd1_p50_us": (70.0, 85.0)}
        mid = {"probe_qd1_p50_us": (85.0, 100.0)}
        cfg = DeviceStateConfig(
            target="folded",
            states={"folded": mid},
            size_classes=[
                {"max_gb": 2.5, "states": {"folded": small}},
                {"max_gb": 20, "states": {"fresh": FRESH}},
            ],
        )
        assert cfg.bands_for(1.6) == small
        assert cfg.bands_for(5.9) == mid  # covering class lacks folded: fall back to states
        assert cfg.bands_for(340) == mid
        only_classes = DeviceStateConfig(
            target="folded", size_classes=[{"max_gb": 2.5, "states": {"folded": small}}]
        )
        assert only_classes.bands_for(5.9) == {}

    def test_size_classes_validated(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            DeviceStateConfig(size_classes=[{"max_gb": 20}, {"max_gb": 2}])
        with pytest.raises(ValidationError, match=r"size_classes\[0\].*unknown probe"):
            DeviceStateConfig(size_classes=[{"max_gb": 2, "states": {"fresh": {"qps": (1, 2)}}}])


class TestEvaluatorGate:
    def _setup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qd1: list[float],
        **gate: Any,
    ) -> tuple[Any, Path, MagicMock]:
        evaluator, _ = make_evaluator(tmp_path)
        evaluator.config.device_state = DeviceStateConfig(
            target="fresh", states={"fresh": FRESH}, wait_s=0, settle_s=0, **gate
        )
        idx = tmp_path / "idx"
        idx.mkdir()
        (idx / "graph.bin").write_bytes(b"x" * 4096)
        values = iter(qd1)
        monkeypatch.setattr(
            "ann_suite.evaluator.probe_device",
            lambda d: {"probe_file": "graph.bin", "probe_qd1_p50_us": next(values)},
        )
        monkeypatch.setattr("ann_suite.evaluator.drive_record", lambda f: {"dev_name": "x"})
        rewrite = MagicMock(return_value=0.0)
        monkeypatch.setattr("ann_suite.evaluator.rewrite_files", rewrite)
        return evaluator, idx, rewrite

    def test_waits_out_a_transient_dip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator, idx, rewrite = self._setup(tmp_path, monkeypatch, [90.0, 30.0])
        state = evaluator._device_state(idx)
        assert state["state_ok"] and state["state_waits"] == 1 and state["state_prepares"] == 0
        assert state["dev_name"] == "x"
        rewrite.assert_not_called()

    def test_prepares_then_accepts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        evaluator, idx, rewrite = self._setup(
            tmp_path, monkeypatch, [90.0, 90.0, 30.0], max_waits=1, prepare=True
        )
        state = evaluator._device_state(idx)
        assert state["state_ok"] and state["state_prepares"] == 1
        rewrite.assert_called_once()

    def test_prepare_budget_is_per_index_per_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator, idx, rewrite = self._setup(
            tmp_path, monkeypatch, [90.0] * 10, max_waits=0, prepare=True, max_prepares=1
        )
        assert evaluator._device_state(idx)["state_ok"] is False
        assert evaluator._device_state(idx)["state_prepares"] == 0
        rewrite.assert_called_once()

    def test_refused_point_skips_search_and_keeps_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator, idx, _ = self._setup(tmp_path, monkeypatch, [90.0] * 10, max_waits=0)
        evaluator._run_search_phase = MagicMock()
        context = _BuildContext(
            build_result=make_phase_result(),
            host_index_dir=idx,
            container_index_path="/data/index",
            build_params={},
        )
        algo = make_algo(algorithm_type=AlgorithmType.DISK)
        result = evaluator._run_benchmark_point(
            algo, make_dataset(), context, tmp_path / "q.npy", None, {"L": 10}
        )
        evaluator._run_search_phase.assert_not_called()
        assert result.run_conditions["state_ok"] is False
        assert result.run_conditions["probe_qd1_p50_us"] == 90.0

    def test_size_class_bands_apply_and_uncovered_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluator, idx, _ = self._setup(tmp_path, monkeypatch, [80.0, 80.0], max_waits=0)
        big = {"probe_qd1_p50_us": (70.0, 85.0)}
        evaluator.config.device_state = DeviceStateConfig(
            target="folded", wait_s=0, size_classes=[{"max_gb": 1, "states": {"folded": big}}]
        )
        assert evaluator._device_state(idx)["state_ok"]  # 4 KiB file: the 1 GB class
        evaluator.config.device_state = DeviceStateConfig(
            target="folded", wait_s=0, size_classes=[{"max_gb": 1e-9, "states": {"folded": big}}]
        )
        state = evaluator._device_state(idx)
        assert state["state_ok"] is False and state["probe_qd1_p50_us"] == 80.0
