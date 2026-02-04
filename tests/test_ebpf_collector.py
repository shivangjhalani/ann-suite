"""Tests for EBPFCollector."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ann_suite.monitoring.base import CollectorResult, CollectorSample
from ann_suite.monitoring.ebpf_collector import EBPFCollector


class TestEBPFCollector:
    """Tests for EBPFCollector class."""

    @pytest.fixture
    def mock_bpf(self):
        with patch("ann_suite.monitoring.ebpf_collector.BPF") as mock:
            yield mock

    @pytest.fixture
    def collector(self, mock_bpf):
        return EBPFCollector(interval_ms=100)

    def test_init(self, collector):
        assert collector.name == "ebpf_block"
        assert collector._interval_seconds == 0.1

    def test_is_available(self, mock_bpf):
        # mocked BPF is not None
        collector = EBPFCollector()
        assert collector.is_available() is True

    def test_is_available_no_bcc(self):
        with patch("ann_suite.monitoring.ebpf_collector.BPF", None):
            collector = EBPFCollector()
            assert collector.is_available() is False

    @patch("ann_suite.monitoring.ebpf_collector.os.stat")
    @patch("pathlib.Path.exists")
    def test_start_success(self, mock_exists, mock_stat, collector, mock_bpf):
        # Mock file existence and inode
        mock_exists.return_value = True
        mock_stat.return_value.st_ino = 12345
        
        # Mock BPF instance
        bpf_instance = mock_bpf.return_value
        bpf_instance.get_kprobe_functions.return_value = True # found kprobes
        
        # Mock kprobe attach
        bpf_instance.attach_kprobe.return_value = None

        # Start
        collector.start("test_container")
        
        assert collector._container_cgroup_id == 12345
        assert collector._running is True
        assert bpf_instance.attach_kprobe.call_count >= 2
        # Verify perf buffer open
        bpf_instance.__getitem__.return_value.open_perf_buffer.assert_called_once()
        
        collector.stop()

    def test_handle_event(self, collector):
        # Simulate an event
        # Event structure: rwflag, bytes, delta_us (from BPF struct)
        event = MagicMock()
        event.rwflag = 0 # Read
        event.bytes = 4096
        event.delta_us = 100
        
        # Manually create bpf mock to avoid attribute error if not started
        collector._bpf = MagicMock()
        collector._bpf["events"].event.return_value = event
        
        # Call handler (simulating callback)
        collector._handle_event(0, b"raw_data", 10)
        
        # Check aggregation
        assert collector._read_bytes == 4096
        assert collector._read_ops == 1
        assert collector._read_us == 100
        assert collector._write_bytes == 0

        # Simulate Write event
        event_write = MagicMock()
        event_write.rwflag = 1 
        event_write.bytes = 8192
        event_write.delta_us = 200
        collector._bpf["events"].event.return_value = event_write
        
        collector._handle_event(0, b"raw_data", 10)
        
        assert collector._write_bytes == 8192
        assert collector._write_ops == 1
        assert collector._write_us == 200

    def test_stop_returns_result(self, collector):
        from datetime import UTC, datetime

        # Pre-populate samples for sample-based aggregation
        # The collector now computes deltas from first to last sample
        now = time.monotonic()
        collector._start_time = now - 1.0  # 1 second duration
        collector._read_bytes = 1000
        collector._write_bytes = 2000
        collector._read_ops = 10
        collector._write_ops = 5

        # Create initial sample (counters at 0)
        initial_sample = CollectorSample(
            timestamp=datetime.now(UTC),
            monotonic_time=now - 1.0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
        )
        collector._samples.append(initial_sample)

        result = collector.stop()

        # stop() collects a final sample with current counter values
        # delta from initial (0) to final (1000, 2000, 10, 5) = 1000, 2000, 10, 5
        assert isinstance(result, CollectorResult)
        assert result.total_read_bytes == 1000
        assert result.total_write_bytes == 2000
        assert result.total_read_ops == 10
        assert result.total_write_ops == 5
        assert result.avg_read_iops == pytest.approx(10.0, rel=0.1)
        assert result.duration_seconds == pytest.approx(1.0, rel=0.1)
        assert result.sample_count == 2  # initial + final
