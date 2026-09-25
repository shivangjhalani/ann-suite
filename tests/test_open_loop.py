"""Tests for open-loop (arrival-rate) search configuration and sweep expansion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ann_suite.core.schemas import (
    AlgorithmConfig,
    AlgorithmType,
    ArrivalConfig,
    ArrivalMode,
    BenchmarkResult,
    SearchConfig,
)
from ann_suite.evaluator import ARRIVAL_RATE_SWEEP_KEY, search_sweep_params


class TestArrivalConfig:
    """Validation behavior for ArrivalConfig."""

    def test_defaults_are_closed_loop_equivalent(self):
        cfg = ArrivalConfig()
        assert cfg.mode == ArrivalMode.CLOSED
        assert cfg.rate_qps is None
        assert cfg.num_queries == 10000
        assert 0.0 <= cfg.warmup_fraction < 1.0

    def test_poisson_requires_rate_qps(self):
        with pytest.raises(ValidationError):
            ArrivalConfig(mode=ArrivalMode.POISSON)

    def test_poisson_with_rate_qps_ok(self):
        cfg = ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=1000.0)
        assert cfg.rate_qps == 1000.0

    def test_poisson_with_rate_list_ok(self):
        cfg = ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=[500.0, 1000.0, 2000.0])
        assert cfg.rate_qps == [500.0, 1000.0, 2000.0]

    def test_rate_qps_must_be_positive(self):
        with pytest.raises(ValidationError):
            ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=0)

    def test_resolved_returns_single_rate_copy(self):
        cfg = ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=[500.0, 1000.0])
        resolved = cfg.resolved(1000.0)
        assert resolved.rate_qps == 1000.0
        assert resolved.mode == ArrivalMode.POISSON
        # Original is untouched (resolved() copies).
        assert cfg.rate_qps == [500.0, 1000.0]

    def test_warmup_fraction_bounds(self):
        with pytest.raises(ValidationError):
            ArrivalConfig(warmup_fraction=1.0)
        with pytest.raises(ValidationError):
            ArrivalConfig(warmup_fraction=-0.1)


class TestArrivalSweepExpansion:
    """search_sweep_params() should fan out over arrival.rate_qps like a search arg."""

    def _algo_config(self, **search_kwargs) -> AlgorithmConfig:
        return AlgorithmConfig(
            name="test-algo",
            docker_image="test/algo:latest",
            algorithm_type=AlgorithmType.DISK,
            search=SearchConfig(**search_kwargs),
        )

    def test_no_arrival_unaffected(self):
        algo = self._algo_config(args={"Ls": [10, 20]})
        points = search_sweep_params(algo)
        assert len(points) == 2
        assert all(ARRIVAL_RATE_SWEEP_KEY not in p for p in points)

    def test_single_rate_adds_key_to_every_point(self):
        algo = self._algo_config(
            args={"Ls": [10, 20]},
            arrival=ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=1000.0),
        )
        points = search_sweep_params(algo)
        assert len(points) == 2
        assert all(p[ARRIVAL_RATE_SWEEP_KEY] == 1000.0 for p in points)

    def test_rate_list_multiplies_points(self):
        algo = self._algo_config(
            args={"Ls": [10, 20]},
            arrival=ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=[500.0, 1000.0, 2000.0]),
        )
        points = search_sweep_params(algo)
        # 2 Ls values x 3 rates = 6 points
        assert len(points) == 6
        rates = sorted({p[ARRIVAL_RATE_SWEEP_KEY] for p in points})
        assert rates == [500.0, 1000.0, 2000.0]
        for ls in (10, 20):
            assert sum(1 for p in points if p["Ls"] == ls) == 3

    def test_explicit_sweep_points_also_expand_over_rates(self):
        algo = self._algo_config(
            sweep=[{"Ls": 10}, {"Ls": 20}],
            arrival=ArrivalConfig(mode=ArrivalMode.POISSON, rate_qps=[500.0, 1000.0]),
        )
        points = search_sweep_params(algo)
        assert len(points) == 4

    def test_closed_mode_without_rate_qps_is_single_point(self):
        algo = self._algo_config(
            args={"Ls": 10},
            arrival=ArrivalConfig(mode=ArrivalMode.CLOSED),
        )
        points = search_sweep_params(algo)
        assert points == [{"Ls": 10}]


class TestBenchmarkResultOpenLoop:
    """BenchmarkResult.open_loop round-trips through to_summary_dict()."""

    def test_open_loop_none_by_default(self):
        result = BenchmarkResult(algorithm="a", dataset="d")
        assert result.open_loop is None
        assert result.to_summary_dict()["open_loop"] is None

    def test_open_loop_dict_passthrough(self):
        payload = {
            "achieved_qps": 950.0,
            "arrival_rate_qps": 1000.0,
            "num_queries": 20000,
            "latency_ms": {"mean": 1.2, "p50": 1.0, "p90": 2.1, "p99": 4.5, "p999": 9.0},
            "service_time_ms": {"mean": 0.8, "p50": 0.7, "p90": 1.5, "p99": 3.0, "p999": 6.0},
            "ios_per_query": 23.4,
        }
        result = BenchmarkResult(algorithm="a", dataset="d", open_loop=payload)
        assert result.open_loop == payload
        assert result.to_summary_dict()["open_loop"] == payload
