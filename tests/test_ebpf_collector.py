"""Tests for EBPFCollector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ann_suite.monitoring.ebpf_collector import EBPFCollector, EBPFMetrics, LatencyHistogram


class TestLatencyHistogram:
    """Tests for LatencyHistogram dataclass."""

    def test_add(self) -> None:
        h = LatencyHistogram()
        h.add(100)
        h.add(200)
        assert h.total_count == 2
        assert h.total_us == 300
        # 100 -> bit_length=7 -> bucket 6, 200 -> bit_length=8 -> bucket 7
        assert h.buckets[6] == 1
        assert h.buckets[7] == 1

    def test_add_zero_latency(self) -> None:
        h = LatencyHistogram()
        h.add(0)
        assert h.total_count == 1
        assert h.total_us == 0
        assert h.buckets[0] == 1

    def test_percentile(self) -> None:
        h = LatencyHistogram()
        # Add 100 samples of 1us (bucket 0) and 100 samples of 1024us (bucket 10)
        for _ in range(100):
            h.add(1)
        for _ in range(100):
            h.add(1024)
        # p50 should be in the first bucket (1 << 0 = 1)
        assert h.percentile(50) == 1
        # p99 should be in the second bucket (1 << 10 = 1024)
        assert h.percentile(99) == 1024

    def test_mean(self) -> None:
        h = LatencyHistogram()
        h.add(100)
        h.add(200)
        h.add(300)
        assert h.mean() == pytest.approx(200.0)

    def test_max(self) -> None:
        h = LatencyHistogram()
        h.add(1)    # bucket 0
        h.add(100)  # bucket 6
        h.add(500)  # bucket 8
        # max bucket is 8 -> 1 << 8 = 256
        assert h.max() == 256

    def test_empty_percentile(self) -> None:
        h = LatencyHistogram()
        assert h.percentile(50) is None

    def test_empty_mean(self) -> None:
        h = LatencyHistogram()
        assert h.mean() is None

    def test_empty_max(self) -> None:
        h = LatencyHistogram()
        assert h.max() is None


class TestEBPFMetrics:
    """Tests for EBPFMetrics dataclass."""

    def test_defaults(self) -> None:
        m = EBPFMetrics()
        assert m.read_ops == 0
        assert m.write_ops == 0
        assert m.read_bytes == 0
        assert m.write_bytes == 0
        assert m.read_sizes == []
        assert m.write_sizes == []
        assert m.read_latency.total_count == 0
        assert m.write_latency.total_count == 0


class TestEBPFCollector:
    """Tests for EBPFCollector class."""

    @pytest.fixture()
    def collector(self) -> EBPFCollector:
        with patch("ann_suite.monitoring.ebpf_collector.BPF"):
            return EBPFCollector(interval_ms=100)

    def test_init(self, collector: EBPFCollector) -> None:
        assert collector.name == "ebpf_block"
        assert collector._interval_seconds == 0.1
        assert collector._running is False
        assert collector._metrics.read_ops == 0
        assert collector._metrics.write_ops == 0

    def test_is_available(self) -> None:
        with patch("ann_suite.monitoring.ebpf_collector.BPF", MagicMock()):
            c = EBPFCollector()
            assert c.is_available() is True

    def test_is_available_no_bcc(self) -> None:
        with patch("ann_suite.monitoring.ebpf_collector.BPF", None):
            c = EBPFCollector()
            assert c.is_available() is False

    def test_handle_event(self, collector: EBPFCollector) -> None:
        event = MagicMock()
        event.rwflag = 0  # Read
        event.bytes = 4096
        event.delta_us = 100

        collector._bpf = MagicMock()
        collector._bpf["events"].event.return_value = event

        collector._handle_event(0, b"raw_data", 10)

        assert collector._metrics.read_bytes == 4096
        assert collector._metrics.read_ops == 1
        assert collector._metrics.read_latency.total_count == 1
        assert collector._metrics.read_latency.total_us == 100
        assert collector._metrics.read_sizes == [4096]
        assert collector._metrics.write_bytes == 0

        # Simulate write event
        event_write = MagicMock()
        event_write.rwflag = 1
        event_write.bytes = 8192
        event_write.delta_us = 200
        collector._bpf["events"].event.return_value = event_write

        collector._handle_event(0, b"raw_data", 10)

        assert collector._metrics.write_bytes == 8192
        assert collector._metrics.write_ops == 1
        assert collector._metrics.write_latency.total_count == 1
        assert collector._metrics.write_latency.total_us == 200
        assert collector._metrics.write_sizes == [8192]

    def test_get_metrics_returns_copy(self, collector: EBPFCollector) -> None:
        collector._metrics.read_ops = 42
        collector._metrics.read_bytes = 9999
        collector._metrics.read_sizes = [512, 1024]

        result = collector.get_metrics()
        assert result.read_ops == 42
        assert result.read_bytes == 9999
        assert result.read_sizes == [512, 1024]
        # Verify list is a copy
        assert result.read_sizes is not collector._metrics.read_sizes

    def test_stop_returns_metrics(self, collector: EBPFCollector) -> None:
        collector._metrics.read_ops = 10
        collector._metrics.write_ops = 5
        collector._metrics.read_bytes = 1000
        collector._metrics.write_bytes = 2000
        collector._metrics.read_sizes = [512]
        collector._metrics.write_sizes = [1024]

        result = collector.stop()

        assert isinstance(result, EBPFMetrics)
        assert result.read_ops == 10
        assert result.write_ops == 5
        assert result.read_bytes == 1000
        assert result.write_bytes == 2000
        assert result.read_sizes == [512]
        assert result.write_sizes == [1024]
        assert collector._running is False
