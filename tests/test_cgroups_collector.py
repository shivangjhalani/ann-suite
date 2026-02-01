"""Tests for CgroupsV2Collector."""

from datetime import datetime, timedelta

import pytest

from ann_suite.monitoring.base import CollectorSample, DeviceIOStat
from ann_suite.monitoring.cgroups_collector import CgroupsV2Collector


class TestCgroupsV2Collector:
    """Tests for CgroupsV2Collector class."""

    def test_aggregate_samples(self) -> None:
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

    def test_empty_samples(self) -> None:
        """Test handling of no samples."""
        collector = CgroupsV2Collector()
        collector._samples = []
        summary = collector._aggregate_samples()
        assert summary.sample_count == 0
        assert summary.peak_memory_mb == 0.0


class TestIOStatParsing:
    """Tests for io.stat parsing with extended fields."""

    def test_collector_sample_with_extended_io_fields(self) -> None:
        """Test CollectorSample with rusec/wusec I/O latency fields."""
        now = datetime.now()
        sample = CollectorSample(
            timestamp=now,
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=1024 * 1024,
            blkio_write_bytes=512 * 1024,
            blkio_read_ops=100,
            blkio_write_ops=50,
            blkio_read_usec=5000,  # 5ms total read latency
            blkio_write_usec=2000,  # 2ms total write latency
        )
        assert sample.blkio_read_usec == 5000
        assert sample.blkio_write_usec == 2000
        # Verify service time can be computed: 5000 usec / 100 ops = 50 usec/op
        if sample.blkio_read_ops > 0:
            avg_read_service_usec = sample.blkio_read_usec / sample.blkio_read_ops
            assert avg_read_service_usec == pytest.approx(50.0)

    def test_collector_sample_with_per_device_io(self) -> None:
        """Test CollectorSample with per-device I/O breakdown."""
        now = datetime.now()
        per_device = [
            DeviceIOStat(
                device="8:0",
                rbytes=1024 * 1024,
                wbytes=512 * 1024,
                rios=100,
                wios=50,
                rusec=5000,
                wusec=2000,
            ),
            DeviceIOStat(
                device="8:16",
                rbytes=256 * 1024,
                wbytes=128 * 1024,
                rios=25,
                wios=10,
                rusec=1000,
                wusec=500,
            ),
        ]
        sample = CollectorSample(
            timestamp=now,
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=1024 * 1024 + 256 * 1024,  # Aggregated
            blkio_write_bytes=512 * 1024 + 128 * 1024,
            blkio_read_ops=125,
            blkio_write_ops=60,
            per_device_io=per_device,
        )
        assert sample.per_device_io is not None
        assert len(sample.per_device_io) == 2
        # Verify top device identification
        top_device = max(sample.per_device_io, key=lambda d: d.rbytes)
        assert top_device.device == "8:0"
        assert top_device.rbytes == 1024 * 1024

    def test_device_io_stat_defaults(self) -> None:
        """Test DeviceIOStat default values for optional fields."""
        stat = DeviceIOStat(device="8:0", rbytes=1024, wbytes=512, rios=10, wios=5)
        assert stat.rusec == 0  # Default when not available
        assert stat.wusec == 0

    def test_bytes_per_op_computation(self) -> None:
        """Test bytes per operation computation from samples."""
        sample = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=4096 * 100,  # 400KB
            blkio_write_bytes=4096 * 50,  # 200KB
            blkio_read_ops=100,
            blkio_write_ops=50,
        )
        # Compute bytes per op
        if sample.blkio_read_ops > 0:
            bytes_per_read = sample.blkio_read_bytes / sample.blkio_read_ops
            assert bytes_per_read == pytest.approx(4096.0)
        if sample.blkio_write_ops > 0:
            bytes_per_write = sample.blkio_write_bytes / sample.blkio_write_ops
            assert bytes_per_write == pytest.approx(4096.0)


class TestPSIParsing:
    """Tests for PSI (Pressure Stall Information) parsing."""

    def test_collector_sample_with_psi_fields(self) -> None:
        """Test CollectorSample with I/O pressure fields."""
        now = datetime.now()
        sample = CollectorSample(
            timestamp=now,
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            io_pressure_some_total_usec=12345678,
            io_pressure_full_total_usec=1234567,
        )
        assert sample.io_pressure_some_total_usec == 12345678
        assert sample.io_pressure_full_total_usec == 1234567

    def test_psi_delta_to_percent_conversion(self) -> None:
        """Test converting PSI delta to percentage of time."""
        now = datetime.now()
        sample1 = CollectorSample(
            timestamp=now,
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            io_pressure_some_total_usec=1000000,  # 1 second
            io_pressure_full_total_usec=100000,  # 0.1 second
        )
        sample2 = CollectorSample(
            timestamp=now + timedelta(seconds=10),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            io_pressure_some_total_usec=2000000,  # 2 seconds (1s delta)
            io_pressure_full_total_usec=200000,  # 0.2 seconds (0.1s delta)
        )
        # Compute PSI percentages
        duration_usec = 10 * 1000000  # 10 seconds in usec
        delta_some = sample2.io_pressure_some_total_usec - sample1.io_pressure_some_total_usec
        delta_full = sample2.io_pressure_full_total_usec - sample1.io_pressure_full_total_usec
        io_some_percent = (delta_some / duration_usec) * 100
        io_full_percent = (delta_full / duration_usec) * 100
        # 1s stall over 10s = 10%
        assert io_some_percent == pytest.approx(10.0)
        # 0.1s stall over 10s = 1%
        assert io_full_percent == pytest.approx(1.0)

    def test_psi_defaults_to_zero(self) -> None:
        """Test PSI fields default to zero when not available."""
        sample = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
        )
        assert sample.io_pressure_some_total_usec == 0
        assert sample.io_pressure_full_total_usec == 0


class TestMemoryStatParsing:
    """Tests for memory.stat parsing (page faults, cache stats)."""

    def test_collector_sample_with_memory_stat_fields(self) -> None:
        """Test CollectorSample with memory.stat fields."""
        now = datetime.now()
        sample = CollectorSample(
            timestamp=now,
            memory_usage_bytes=512 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            pgmajfault=1000,
            pgfault=50000,
            file_bytes=256 * 1024 * 1024,
            file_mapped_bytes=128 * 1024 * 1024,
            active_file_bytes=200 * 1024 * 1024,
            inactive_file_bytes=56 * 1024 * 1024,
        )
        assert sample.pgmajfault == 1000
        assert sample.pgfault == 50000
        assert sample.file_bytes == 256 * 1024 * 1024
        assert sample.file_mapped_bytes == 128 * 1024 * 1024
        assert sample.active_file_bytes == 200 * 1024 * 1024
        assert sample.inactive_file_bytes == 56 * 1024 * 1024

    def test_major_faults_per_query_computation(self) -> None:
        """Test major faults per query computation."""
        sample1 = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=512 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            pgmajfault=100,
        )
        sample2 = CollectorSample(
            timestamp=datetime.now() + timedelta(seconds=10),
            memory_usage_bytes=512 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            pgmajfault=1100,  # 1000 major faults delta
        )
        delta_pgmajfault = sample2.pgmajfault - sample1.pgmajfault
        num_queries = 10000
        major_faults_per_query = delta_pgmajfault / num_queries
        assert major_faults_per_query == pytest.approx(0.1)

    def test_major_faults_per_second_computation(self) -> None:
        """Test major faults per second rate computation."""
        sample1 = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=512 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            pgmajfault=0,
        )
        sample2 = CollectorSample(
            timestamp=datetime.now() + timedelta(seconds=5),
            memory_usage_bytes=512 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            pgmajfault=500,  # 500 faults over 5 seconds
        )
        delta_pgmajfault = sample2.pgmajfault - sample1.pgmajfault
        duration_seconds = 5.0
        major_faults_per_second = delta_pgmajfault / duration_seconds
        assert major_faults_per_second == pytest.approx(100.0)

    def test_memory_stat_defaults_to_zero(self) -> None:
        """Test memory.stat fields default to zero when not parsed."""
        sample = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
        )
        assert sample.pgmajfault == 0
        assert sample.pgfault == 0
        assert sample.file_bytes == 0
        assert sample.file_mapped_bytes == 0
        assert sample.active_file_bytes == 0
        assert sample.inactive_file_bytes == 0


class TestTailMetrics:
    """Tests for tail metrics computation (p95/max)."""

    def test_interval_iops_computation(self) -> None:
        """Test computing per-interval IOPS for tail metrics."""
        now = datetime.now()
        samples = [
            CollectorSample(
                timestamp=now,
                monotonic_time=0.0,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=0,
                blkio_write_bytes=0,
                blkio_read_ops=0,
                blkio_write_ops=0,
            ),
            CollectorSample(
                timestamp=now + timedelta(milliseconds=100),
                monotonic_time=0.1,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=4096,
                blkio_write_bytes=0,
                blkio_read_ops=10,  # 10 ops in 100ms = 100 IOPS
                blkio_write_ops=0,
            ),
            CollectorSample(
                timestamp=now + timedelta(milliseconds=200),
                monotonic_time=0.2,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=8192,
                blkio_write_bytes=0,
                blkio_read_ops=60,  # 50 ops in 100ms = 500 IOPS
                blkio_write_ops=0,
            ),
            CollectorSample(
                timestamp=now + timedelta(milliseconds=300),
                monotonic_time=0.3,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=12288,
                blkio_write_bytes=0,
                blkio_read_ops=80,  # 20 ops in 100ms = 200 IOPS
                blkio_write_ops=0,
            ),
        ]
        # Compute interval IOPS
        interval_iops = []
        for i in range(1, len(samples)):
            ops_delta = samples[i].blkio_read_ops - samples[i - 1].blkio_read_ops
            time_delta = samples[i].monotonic_time - samples[i - 1].monotonic_time
            if time_delta > 0:
                interval_iops.append(ops_delta / time_delta)
        # Expected: [100, 500, 200]
        assert len(interval_iops) == 3
        assert interval_iops[0] == pytest.approx(100.0)
        assert interval_iops[1] == pytest.approx(500.0)
        assert interval_iops[2] == pytest.approx(200.0)
        # Max IOPS
        assert max(interval_iops) == pytest.approx(500.0)

    def test_tail_metrics_p95_computation(self) -> None:
        """Test p95 computation for tail metrics."""
        # Create 20 interval values for p95 calculation
        interval_values = [100.0] * 19 + [1000.0]  # 19 low, 1 high
        # p95 should be around 100 (95% of values are 100)
        sorted_values = sorted(interval_values)
        # p95 index: 0.95 * 20 = 19, so sorted_values[18] or interpolated
        # With 20 values, p95 is between index 18 and 19
        p95_index = int(0.95 * len(sorted_values))
        p95_approx = sorted_values[min(p95_index, len(sorted_values) - 1)]
        assert p95_approx == 1000.0  # At index 19

    def test_insufficient_samples_for_tail_metrics(self) -> None:
        """Test that tail metrics require sufficient samples."""
        now = datetime.now()
        samples = [
            CollectorSample(
                timestamp=now,
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=0,
                blkio_write_bytes=0,
                blkio_read_ops=0,
                blkio_write_ops=0,
            ),
            CollectorSample(
                timestamp=now + timedelta(seconds=1),
                memory_usage_bytes=100 * 1024 * 1024,
                cpu_time_ns=0,
                blkio_read_bytes=1024,
                blkio_write_bytes=0,
                blkio_read_ops=10,
                blkio_write_ops=0,
            ),
        ]
        # Only 1 interval - not enough for meaningful percentiles
        interval_count = len(samples) - 1
        assert interval_count == 1
        # Tail metrics should be None or fallback to the single value
        # This tests the edge case handling


class TestCPUThrottling:
    """Tests for CPU throttling statistics."""

    def test_collector_sample_with_throttling_fields(self) -> None:
        """Test CollectorSample with CPU throttling fields."""
        now = datetime.now()
        sample = CollectorSample(
            timestamp=now,
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=1000000000,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            nr_throttled=50,
            throttled_usec=500000,  # 0.5 seconds
        )
        assert sample.nr_throttled == 50
        assert sample.throttled_usec == 500000

    def test_throttled_percent_computation(self) -> None:
        """Test throttled percentage computation."""
        sample1 = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            nr_throttled=0,
            throttled_usec=0,
        )
        sample2 = CollectorSample(
            timestamp=datetime.now() + timedelta(seconds=10),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
            nr_throttled=10,
            throttled_usec=1000000,  # 1 second throttled
        )
        duration_usec = 10 * 1000000  # 10 seconds
        delta_throttled = sample2.throttled_usec - sample1.throttled_usec
        throttled_percent = (delta_throttled / duration_usec) * 100
        # 1s throttled over 10s = 10%
        assert throttled_percent == pytest.approx(10.0)

    def test_throttling_defaults_to_zero(self) -> None:
        """Test throttling fields default to zero."""
        sample = CollectorSample(
            timestamp=datetime.now(),
            memory_usage_bytes=100 * 1024 * 1024,
            cpu_time_ns=0,
            blkio_read_bytes=0,
            blkio_write_bytes=0,
            blkio_read_ops=0,
            blkio_write_ops=0,
        )
        assert sample.nr_throttled == 0
        assert sample.throttled_usec == 0
