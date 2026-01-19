"""Tests for new metrics schemas and parameter sweep functionality."""


from ann_suite.core.schemas import (
    BenchmarkResult,
    CPUMetrics,
    DiskIOMetrics,
    LatencyMetrics,
    MemoryMetrics,
)
from ann_suite.evaluator import expand_sweep_params


class TestMetricsSchemas:
    """Tests for the new structured metrics schemas."""

    def test_cpu_metrics_defaults(self) -> None:
        """Test CPUMetrics with default values."""
        cpu = CPUMetrics()
        assert cpu.cpu_time_total_seconds == 0.0
        assert cpu.avg_cpu_percent == 0.0
        assert cpu.peak_cpu_percent == 0.0

    def test_cpu_metrics_with_values(self) -> None:
        """Test CPUMetrics with custom values."""
        cpu = CPUMetrics(
            cpu_time_total_seconds=5.5,
            avg_cpu_percent=150.0,
            peak_cpu_percent=380.0,
        )
        assert cpu.cpu_time_total_seconds == 5.5
        assert cpu.avg_cpu_percent == 150.0
        assert cpu.peak_cpu_percent == 380.0

    def test_memory_metrics(self) -> None:
        """Test MemoryMetrics schema."""
        mem = MemoryMetrics(peak_rss_mb=512.0, avg_rss_mb=400.0)
        assert mem.peak_rss_mb == 512.0
        assert mem.avg_rss_mb == 400.0

    def test_disk_io_metrics_critical(self) -> None:
        """Test DiskIOMetrics with CRITICAL IOPS metrics."""
        disk = DiskIOMetrics(
            avg_read_iops=1500.0,
            avg_write_iops=100.0,
            avg_read_throughput_mbps=6.0,
            avg_write_throughput_mbps=0.4,
            total_pages_read=150000,
            total_pages_written=1000,
            pages_per_query=15.0,
        )
        assert disk.avg_read_iops == 1500.0
        assert disk.total_pages_read == 150000
        assert disk.pages_per_query == 15.0

    def test_latency_metrics(self) -> None:
        """Test LatencyMetrics schema."""
        latency = LatencyMetrics(
            mean_ms=0.5,
            p50_ms=0.3,
            p95_ms=1.2,
            p99_ms=2.5,
        )
        assert latency.mean_ms == 0.5
        assert latency.p99_ms == 2.5


class TestBenchmarkResultStructured:
    """Tests for structured BenchmarkResult with nested metrics."""

    def test_benchmark_result_with_structured_metrics(self) -> None:
        """Test BenchmarkResult with all structured metrics."""
        result = BenchmarkResult(
            algorithm="HNSW",
            dataset="glove-25",
            cpu=CPUMetrics(cpu_time_total_seconds=2.5, avg_cpu_percent=150.0),
            memory=MemoryMetrics(peak_rss_mb=512.0),
            disk_io=DiskIOMetrics(avg_read_iops=1500.0),
            latency=LatencyMetrics(mean_ms=0.5, p99_ms=2.5),
            recall=0.95,
            qps=5000.0,
            hyperparameters={"build": {"M": 16}, "search": {"ef": 100}},
        )
        assert result.algorithm == "HNSW"
        assert result.cpu.cpu_time_total_seconds == 2.5
        assert result.memory.peak_rss_mb == 512.0
        assert result.disk_io.avg_read_iops == 1500.0
        assert result.recall == 0.95
        assert result.hyperparameters["search"]["ef"] == 100

    def test_to_flat_dict_flattens_nested_metrics(self) -> None:
        """Test that to_flat_dict correctly flattens nested metrics."""
        result = BenchmarkResult(
            algorithm="HNSW",
            dataset="test",
            cpu=CPUMetrics(avg_cpu_percent=100.0),
            disk_io=DiskIOMetrics(avg_read_iops=500.0),
        )
        flat = result.to_flat_dict()

        # Check flattened keys exist with prefixes
        assert "cpu_avg_cpu_percent" in flat
        assert flat["cpu_avg_cpu_percent"] == 100.0
        assert "disk_io_avg_read_iops" in flat
        assert flat["disk_io_avg_read_iops"] == 500.0
        assert "algorithm" in flat
        assert flat["algorithm"] == "HNSW"


class TestParameterSweeps:
    """Tests for parameter sweep expansion functionality."""

    def test_expand_single_list_param(self) -> None:
        """Test expanding a single list-valued parameter."""
        result = expand_sweep_params({"ef": [50, 100, 200], "num_threads": 4})
        assert len(result) == 3
        assert {"ef": 50, "num_threads": 4} in result
        assert {"ef": 100, "num_threads": 4} in result
        assert {"ef": 200, "num_threads": 4} in result

    def test_expand_multiple_list_params(self) -> None:
        """Test expanding multiple list-valued parameters (cartesian product)."""
        result = expand_sweep_params({"ef": [50, 100], "Ls": [100, 200]})
        assert len(result) == 4  # 2 x 2 = 4 combinations
        assert {"ef": 50, "Ls": 100} in result
        assert {"ef": 50, "Ls": 200} in result
        assert {"ef": 100, "Ls": 100} in result
        assert {"ef": 100, "Ls": 200} in result

    def test_expand_no_list_params(self) -> None:
        """Test that scalar params return single-element list."""
        result = expand_sweep_params({"ef": 100, "num_threads": 4})
        assert len(result) == 1
        assert result[0] == {"ef": 100, "num_threads": 4}

    def test_expand_empty_args(self) -> None:
        """Test that empty args returns single empty dict."""
        result = expand_sweep_params({})
        assert len(result) == 1
        assert result[0] == {}

    def test_expand_preserves_scalar_params(self) -> None:
        """Test that scalar params are preserved in each combination."""
        result = expand_sweep_params({
            "ef": [50, 100],
            "num_threads": 4,
            "batch_size": 1000,
        })
        assert len(result) == 2
        for combo in result:
            assert combo["num_threads"] == 4
            assert combo["batch_size"] == 1000
