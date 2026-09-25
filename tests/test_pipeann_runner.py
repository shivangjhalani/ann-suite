"""Tests for the PipeANN algorithm runner's pure-Python helpers.

The runner (library/algorithms/pipeann/algorithm/runner.py) lives outside the
`ann_suite` package - it ships inside the algorithm's Docker image and is
invoked as `python -m algorithm.runner` there, with no dependency on the
suite's own package. It's loaded here by file path (under a private module
name, so it can't collide with any other algorithm's `algorithm.*` package)
to unit-test its format/parsing logic without needing Docker or the compiled
PipeANN binaries.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "library"
    / "algorithms"
    / "pipeann"
    / "algorithm"
    / "runner.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pipeann_runner_under_test", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


class TestDtypeMapping:
    def test_uint8(self, runner):
        assert runner._pipeann_type_for(np.dtype(np.uint8)) == "uint8"

    def test_int8(self, runner):
        assert runner._pipeann_type_for(np.dtype(np.int8)) == "int8"

    def test_float32_and_float64_map_to_float(self, runner):
        assert runner._pipeann_type_for(np.dtype(np.float32)) == "float"
        assert runner._pipeann_type_for(np.dtype(np.float64)) == "float"

    def test_numpy_dtype_for_roundtrip(self, runner):
        for name in ("uint8", "int8", "float"):
            assert runner._pipeann_type_for(runner._numpy_dtype_for(name)) == name


class TestMetricArg:
    def test_known_metrics(self, runner):
        assert runner._metric_arg("L2") == "l2"
        assert runner._metric_arg("IP") == "mips"
        assert runner._metric_arg("cosine") == "cosine"

    def test_unknown_metric_raises(self, runner):
        with pytest.raises(ValueError):
            runner._metric_arg("hamming")


class TestBinFormat:
    def test_write_bin_uint8_header_and_payload(self, runner, tmp_path):
        data = np.arange(24, dtype=np.uint8).reshape(6, 4)
        path = tmp_path / "data.bin"
        element_type = runner._write_bin(path, data)

        assert element_type == "uint8"
        raw = path.read_bytes()
        header = np.frombuffer(raw[:8], dtype=np.uint32)
        assert tuple(header) == (6, 4)
        payload = np.frombuffer(raw[8:], dtype=np.uint8).reshape(6, 4)
        np.testing.assert_array_equal(payload, data)

    def test_write_bin_casts_float64_to_float32(self, runner, tmp_path):
        data = np.random.default_rng(0).random((3, 5)).astype(np.float64)
        path = tmp_path / "data.bin"
        element_type = runner._write_bin(path, data)

        assert element_type == "float"
        raw = path.read_bytes()
        payload = np.frombuffer(raw[8:], dtype=np.float32).reshape(3, 5)
        np.testing.assert_allclose(payload, data.astype(np.float32))

    def test_write_gt_bin_ids_only_format(self, runner, tmp_path):
        gt = np.array([[3, 1, 2, 9], [0, 5, 4, 8]], dtype=np.int64)
        path = tmp_path / "gt.bin"
        runner._write_gt_bin(path, gt, k=3)

        raw = path.read_bytes()
        npts, dim = np.frombuffer(raw[:8], dtype=np.int32)
        assert (npts, dim) == (2, 3)
        ids = np.frombuffer(raw[8:], dtype=np.uint32).reshape(2, 3)
        np.testing.assert_array_equal(ids, gt[:, :3])
        # ids-only format: no trailing distances block.
        assert len(raw) == 8 + 2 * 3 * 4


class TestResultParsing:
    def test_closedloop_result_row_regex(self, runner):
        line = "    10          32     1871.92      512.01      939.00       23.24       67.40"
        m = runner._RESULT_ROW_RE.match(line)
        assert m is not None
        L, io_width, qps, avg_lat, p99_lat, mean_ios, recall = (float(g) for g in m.groups())
        assert (L, io_width) == (10.0, 32.0)
        assert qps == pytest.approx(1871.92)
        assert recall == pytest.approx(67.40)

    def test_closedloop_header_line_does_not_match(self, runner):
        header = "     L   I/O Width         QPS  AvgLat(us)     P99 Lat    Mean IOs   Recall@10"
        assert runner._RESULT_ROW_RE.match(header) is None

    def test_openloop_result_line_kv_parsing(self, runner):
        line = (
            "RESULT mode=2 T=8 W=32 L=50 lambda=1000.0 achieved_qps=943.2 recall=0.9421 "
            "lat_mean=812.3 p50=701.0 p90=1502.0 p99=3011.0 p999=8820.5 svc_mean=650.1 "
            "svc_p99=2200.0 ios=41.2 dev_iops=15342.0 dev_util=0.812 cpu=0.734 dev_lat=52.1"
        )
        values = {m.group(1): float(m.group(2)) for m in runner._OPENLOOP_KV_RE.finditer(line)}
        assert values["achieved_qps"] == pytest.approx(943.2)
        assert values["p999"] == pytest.approx(8820.5)
        assert values["dev_iops"] == pytest.approx(15342.0)
        assert values["cpu"] == pytest.approx(0.734)


class TestArrivalEnv:
    def test_defaults_no_env_overrides(self, runner):
        assert runner._arrival_env({}) == {}

    def test_sqpoll_disabled(self, runner):
        assert runner._arrival_env({"sqpoll": False}) == {"PIPEANN_SQPOLL": "0"}

    def test_governor_env_vars(self, runner):
        env = runner._arrival_env({"qstar": 64, "wmin": 2})
        assert env == {"PIPEANN_QSTAR": "64", "PIPEANN_WMIN": "2"}
