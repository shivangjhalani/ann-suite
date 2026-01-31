"""Tests for CgroupsV2Collector."""

from datetime import datetime, timedelta

import pytest

from ann_suite.monitoring.base import CollectorSample
from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector


class TestCgroupsV2Collector:
    """Tests for CgroupsV2Collector class."""

    def test_aggregate_samples(self):
        """Test aggregating samples into summary."""
        collector = CgroupsV2Collector(interval_ms=100)

        # Add mock samples
        now = datetime.now()
        collector._samples = [
            CollectorSample(
                timestamp=now,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_percent=0.0,
                cpu_time_ns=1000 * 1000 * 1000,  # 1s
                blkio_read_bytes=0,
                blkio_write_bytes=0,
                blkio_read_ops=0,
                blkio_write_ops=0,
            ),
            CollectorSample(
                timestamp=now + timedelta(seconds=1),
                memory_usage_bytes=200 * 1024 * 1024,
                cpu_percent=0.0,
                cpu_time_ns=1500 * 1000 * 1000,  # 1.5s (0.5s delta)
                blkio_read_bytes=1024,
                blkio_write_bytes=0,
                blkio_read_ops=10,
                blkio_write_ops=0,
            ),
        ]

        summary = collector._aggregate_samples()

        # CPU: 0.5s over 1s = 50%
        assert summary.avg_cpu_percent == pytest.approx(50.0, rel=0.1)
        assert summary.peak_memory_mb == pytest.approx(200.0, rel=0.01)
        assert summary.avg_read_iops == pytest.approx(10.0, rel=0.01)
        assert summary.duration_seconds == pytest.approx(1.0, rel=0.01)

    def test_get_summary_filtering(self):
        """Test filtering samples by time window."""
        collector = CgroupsV2Collector(interval_ms=100)

        start_time = datetime.now()

        # Create 10 samples, 1 second apart
        samples = []
        for i in range(10):
            samples.append(
                CollectorSample(
                    timestamp=start_time + timedelta(seconds=i),
                    memory_usage_bytes=100 * 1024 * 1024,
                    cpu_time_ns=i * 1000000000,
                    cpu_percent=0.0,
                    blkio_read_bytes=0,
                    blkio_write_bytes=0,
                    blkio_read_ops=0,
                    blkio_write_ops=0,
                )
            )
        collector._samples = samples

        # Filter window: from 2.5s to 5.5s (should include indices 3, 4, 5)
        # Actually logic is strictly within window: > start and < end?
        # My code said: if timestamp < start continue.
        # So it implies inclusive start, inclusive end?
        # Code: if s.timestamp < start_timestamp: continue

        window_start = start_time + timedelta(seconds=2.5)
        window_end = start_time + timedelta(seconds=5.5)

        summary = collector.get_summary(window_start, window_end)

        # Should include indices 3, 4, 5 (timestamps 3.0, 4.0, 5.0)
        # Duration: from 3.0 to 5.0 = 2.0 seconds
        assert summary.duration_seconds == pytest.approx(2.0, rel=0.01)
        assert summary.sample_count == 3

    def test_empty_samples(self):
        """Test handling of no samples."""
        collector = CgroupsV2Collector()
        collector._samples = []
        summary = collector._aggregate_samples()
        assert summary.sample_count == 0
        assert summary.peak_memory_mb == 0.0
