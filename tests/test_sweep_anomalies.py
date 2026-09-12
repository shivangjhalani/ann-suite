"""Tests for detect_sweep_anomalies.

The fixture data in TestRealAnomalyReproduces is the actual SIFT100M
cache-fraction sweep that first surfaced this check's motivating case: a real
hash-table clustering bug in DiskANN's node cache (see the ai-researcher
project's Knowledge/diskann-robin-map-hash-clustering-bug.md for the full
root-cause writeup). Kept here verbatim so a future refactor of the heuristic
can be checked against the case it was built for.
"""

from __future__ import annotations

from ann_suite.core.schemas import BenchmarkResult, CPUMetrics, DiskIOMetrics
from ann_suite.evaluator import detect_sweep_anomalies


def _result(num_nodes_to_cache: int, pages_per_query: float, qps: float) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm="DiskANN-R32-Lb50-fast",
        dataset="sift100m",
        hyperparameters={
            "build": {"R": 32, "L": 50},
            "search": {"Ls": 200, "beam_width": 4, "num_nodes_to_cache": num_nodes_to_cache},
            "k": 10,
        },
        disk_io=DiskIOMetrics(search_pages_per_query=pages_per_query),
        cpu=CPUMetrics(),
        qps=qps,
    )


class TestRealAnomalyReproduces:
    """The actual SIFT100M sweep that motivated this check (2026-09-13)."""

    def test_flags_entry_into_the_anomalous_region(self) -> None:
        results = [
            _result(0, 224.32, 148.12),
            _result(1_000, 214.17, 153.7),
            _result(10_000, 210.95, 155.6),
            _result(100_000, 207.90, 157.4),
            _result(500_000, 205.16, 157.9),
            _result(1_000_000, 203.44, 158.5),
            _result(5_000_000, 191.32, 114.2),  # anomalous: I/O down, QPS way down
            _result(10_000_000, 181.60, 140.5),  # still anomalous, but *recovering* vs. 5M
            _result(25_000_000, 151.41, 172.0),
            _result(50_000_000, 103.05, 194.6),
        ]

        warnings = detect_sweep_anomalies(results)

        # The heuristic is point-to-point, so it only catches the *local*
        # regression at the 1M -> 5M transition (I/O improves, QPS drops
        # sharply). 5M -> 10M is itself a local *improvement* in QPS even
        # though 10M is still anomalously low relative to the broader I/O
        # trend, so it is correctly not flagged by this simple check -- see
        # the "heuristic, not a correctness check" caveat in the docstring.
        assert len(warnings) == 1
        assert "1000000 -> 5000000" in warnings[0]
        assert "search_num_nodes_to_cache" in warnings[0]


class TestNoFalsePositiveOnCleanSweep:
    def test_smooth_monotonic_sweep_is_not_flagged(self) -> None:
        # SIFT10M sweep, no anomaly present: pages_per_query and qps move
        # together (both improve) at every step.
        results = [
            _result(0, 109.41, 187.3),
            _result(1_000, 105.58, 193.9),
            _result(100_000, 100.47, 199.3),
            _result(1_000_000, 85.68, 209.7),
            _result(5_000_000, 45.11, 300.0),
        ]

        assert detect_sweep_anomalies(results) == []


class TestEdgeCases:
    def test_fewer_than_three_points_is_skipped(self) -> None:
        results = [_result(0, 224.32, 148.12), _result(5_000_000, 191.32, 114.2)]
        assert detect_sweep_anomalies(results) == []

    def test_multiple_varying_numeric_params_is_skipped(self) -> None:
        # Two params vary at once (num_nodes_to_cache and Ls) -- the heuristic
        # deliberately declines to guess which one is responsible.
        results = [
            BenchmarkResult(
                algorithm="a",
                dataset="d",
                hyperparameters={"search": {"num_nodes_to_cache": n, "Ls": ls}},
                disk_io=DiskIOMetrics(search_pages_per_query=p),
                qps=q,
            )
            for n, ls, p, q in [
                (0, 100, 224.32, 148.12),
                (1_000_000, 150, 203.44, 158.5),
                (5_000_000, 200, 191.32, 114.2),
            ]
        ]
        assert detect_sweep_anomalies(results) == []

    def test_missing_metrics_are_skipped_not_crashed(self) -> None:
        results = [
            _result(0, 224.32, 148.12),
            BenchmarkResult(
                algorithm="DiskANN-R32-Lb50-fast",
                dataset="sift100m",
                hyperparameters={"search": {"num_nodes_to_cache": 1_000_000}},
                disk_io=DiskIOMetrics(search_pages_per_query=None),
                qps=None,
            ),
            _result(5_000_000, 191.32, 114.2),
        ]
        assert detect_sweep_anomalies(results) == []

    def test_no_results_returns_empty(self) -> None:
        assert detect_sweep_anomalies([]) == []
