"""Tests for ResourceMonitor."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ann_suite.core.schemas import ResourceSample
from ann_suite.monitoring.resource_monitor import ResourceMonitor


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def create_mock_stats(
        self,
        memory_usage: int = 1024 * 1024 * 100,  # 100 MB
        memory_limit: int = 1024 * 1024 * 1024,  # 1 GB
        blkio_read: int = 1024 * 1024,  # 1 MB
        blkio_write: int = 512 * 1024,  # 512 KB
    ) -> dict:
        """Create a mock Docker stats response."""
        return {
            "memory_stats": {
                "usage": memory_usage,
                "limit": memory_limit,
                "stats": {
                },
            },
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000000000, "percpu_usage": [500000000, 500000000]},
                "system_cpu_usage": 10000000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 900000000, "percpu_usage": [450000000, 450000000]},
                "system_cpu_usage": 9000000000,
            },
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "Read", "value": blkio_read},
                    {"op": "Write", "value": blkio_write},
                ],
            },
            "pids_stats": {
                "current": 5,
            },
        }

    def test_parse_stats_memory(self):
        """Test parsing memory stats from Docker API."""
        mock_container = MagicMock()
        monitor = ResourceMonitor(mock_container, interval_ms=100)

        stats = self.create_mock_stats(memory_usage=200 * 1024 * 1024)  # 200 MB
        sample = monitor._parse_stats(stats)

        assert sample is not None
        assert sample.memory_usage_bytes == 200 * 1024 * 1024
        assert sample.memory_percent == pytest.approx(200 / 1024 * 100, rel=0.01)

    def test_parse_stats_blkio(self):
        """Test parsing block I/O stats."""
        mock_container = MagicMock()
        monitor = ResourceMonitor(mock_container, interval_ms=100)

        stats = self.create_mock_stats(
            blkio_read=10 * 1024 * 1024,  # 10 MB
            blkio_write=5 * 1024 * 1024,  # 5 MB
        )
        sample = monitor._parse_stats(stats)

        assert sample is not None
        assert sample.blkio_read_bytes == 10 * 1024 * 1024
        assert sample.blkio_write_bytes == 5 * 1024 * 1024



    def test_aggregate_samples(self):
        """Test aggregating samples into summary."""
        mock_container = MagicMock()
        monitor = ResourceMonitor(mock_container, interval_ms=100)

        # Add mock samples
        now = datetime.now()
        monitor._samples = [
            ResourceSample(
                timestamp=now,
                memory_usage_bytes=100 * 1024 * 1024,
                memory_limit_bytes=1024 * 1024 * 1024,
                memory_percent=10.0,
                cpu_percent=25.0,
                blkio_write_bytes=0,
            ),
            ResourceSample(
                timestamp=now,
                memory_usage_bytes=200 * 1024 * 1024,
                memory_limit_bytes=1024 * 1024 * 1024,
                memory_percent=20.0,
                cpu_percent=50.0,
                blkio_write_bytes=512 * 1024,
            ),
        ]

        summary = monitor._aggregate_samples()

        assert summary.peak_memory_mb == pytest.approx(200.0, rel=0.01)
        assert summary.avg_memory_mb == pytest.approx(150.0, rel=0.01)
        assert summary.peak_cpu_percent == 50.0
        assert summary.sample_count == 2

    def test_empty_samples(self):
        """Test handling of no samples."""
        mock_container = MagicMock()
        monitor = ResourceMonitor(mock_container, interval_ms=100)

        summary = monitor._aggregate_samples()

        assert summary.peak_memory_mb == 0
        assert summary.sample_count == 0
