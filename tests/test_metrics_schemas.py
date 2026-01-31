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
        assert cpu.build_cpu_time_seconds == 0.0
        assert cpu.build_peak_cpu_percent == 0.0
        assert cpu.search_cpu_time_seconds == 0.0
        assert cpu.search_avg_cpu_percent == 0.0
        assert cpu.search_peak_cpu_percent == 0.0

    def test_cpu_metrics_with_values(self) -> None:
        """Test CPUMetrics with custom values (phase-separated)."""
        cpu = CPUMetrics(
            build_cpu_time_seconds=2.5,
            build_peak_cpu_percent=200.0,
            search_cpu_time_seconds=5.5,
            search_avg_cpu_percent=150.0,
            search_peak_cpu_percent=380.0,
        )
        assert cpu.build_cpu_time_seconds == 2.5
        assert cpu.build_peak_cpu_percent == 200.0
        assert cpu.search_cpu_time_seconds == 5.5
        assert cpu.search_avg_cpu_percent == 150.0
        assert cpu.search_peak_cpu_percent == 380.0

    def test_memory_metrics(self) -> None:
        """Test MemoryMetrics schema (phase-separated)."""
        mem = MemoryMetrics(
            build_peak_rss_mb=512.0,
            search_peak_rss_mb=256.0,
            search_avg_rss_mb=200.0,
        )
        assert mem.build_peak_rss_mb == 512.0
        assert mem.search_peak_rss_mb == 256.0
        assert mem.search_avg_rss_mb == 200.0

    def test_disk_io_metrics_critical(self) -> None:
        """Test DiskIOMetrics with CRITICAL IOPS metrics (phase-separated)."""
        disk = DiskIOMetrics(
            # Warmup phase metrics
            warmup_read_mb=100.0,
            warmup_write_mb=5.0,
            # Search phase metrics
            search_avg_read_iops=1500.0,
            search_avg_write_iops=100.0,
            search_avg_read_throughput_mbps=6.0,
            search_avg_write_throughput_mbps=0.4,
            search_total_read_mb=600.0,
            search_total_pages_read=150000,
            search_total_pages_written=1000,
            search_pages_per_query=15.0,
            physical_block_size=4096,
            sample_count=100,
        )
        assert disk.search_avg_read_iops == 1500.0
        assert disk.search_total_pages_read == 150000
        assert disk.search_pages_per_query == 15.0
        assert disk.warmup_read_mb == 100.0
        assert disk.physical_block_size == 4096
        assert disk.sample_count == 100

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
        """Test BenchmarkResult with all structured metrics (phase-separated)."""
        result = BenchmarkResult(
            algorithm="HNSW",
            dataset="glove-25",
            cpu=CPUMetrics(
                build_cpu_time_seconds=1.0,
                warmup_cpu_time_seconds=0.5,
                search_cpu_time_seconds=2.5,
                search_avg_cpu_percent=150.0,
            ),
            memory=MemoryMetrics(
                build_peak_rss_mb=512.0,
                warmup_peak_rss_mb=256.0,
                search_peak_rss_mb=256.0,
            ),
            disk_io=DiskIOMetrics(
                warmup_read_mb=50.0,
                search_avg_read_iops=1500.0,
            ),
            latency=LatencyMetrics(mean_ms=0.5, p99_ms=2.5),
            recall=0.95,
            qps=5000.0,
            hyperparameters={"build": {"M": 16}, "search": {"ef": 100}},
        )
        assert result.algorithm == "HNSW"
        assert result.cpu.search_cpu_time_seconds == 2.5
        assert result.cpu.warmup_cpu_time_seconds == 0.5
        assert result.memory.build_peak_rss_mb == 512.0
        assert result.memory.warmup_peak_rss_mb == 256.0
        assert result.disk_io.search_avg_read_iops == 1500.0
        assert result.disk_io.warmup_read_mb == 50.0
        assert result.recall == 0.95
        assert result.hyperparameters["search"]["ef"] == 100

    def test_to_flat_dict_flattens_nested_metrics(self) -> None:
        """Test that to_flat_dict correctly flattens nested metrics (phase-separated)."""
        result = BenchmarkResult(
            algorithm="HNSW",
            dataset="test",
            cpu=CPUMetrics(search_avg_cpu_percent=100.0, warmup_cpu_time_seconds=0.5),
            disk_io=DiskIOMetrics(search_avg_read_iops=500.0, warmup_read_mb=25.0),
        )
        flat = result.to_flat_dict()

        # Check flattened keys exist with prefixes (new phase-separated names)
        assert "cpu_search_avg_cpu_percent" in flat
        assert flat["cpu_search_avg_cpu_percent"] == 100.0
        assert "cpu_warmup_cpu_time_seconds" in flat
        assert flat["cpu_warmup_cpu_time_seconds"] == 0.5
        assert "disk_io_search_avg_read_iops" in flat
        assert flat["disk_io_search_avg_read_iops"] == 500.0
        assert "disk_io_warmup_read_mb" in flat
        assert flat["disk_io_warmup_read_mb"] == 25.0
        assert "disk_io_physical_block_size" in flat
        assert "disk_io_sample_count" in flat
        assert "algorithm" in flat
        assert flat["algorithm"] == "HNSW"

    def test_to_summary_dict_phase_structured(self) -> None:
        """Test that to_summary_dict returns phase-structured output."""
        result = BenchmarkResult(
            algorithm="DiskANN",
            dataset="sift-1m",
            cpu=CPUMetrics(
                build_cpu_time_seconds=100.0,
                build_peak_cpu_percent=380.0,
                warmup_cpu_time_seconds=5.0,
                warmup_peak_cpu_percent=95.0,
                search_cpu_time_seconds=25.0,
                search_avg_cpu_percent=45.0,
                search_peak_cpu_percent=50.0,
                search_cpu_time_per_query_ms=2.5,
            ),
            memory=MemoryMetrics(
                build_peak_rss_mb=4000.0,
                warmup_peak_rss_mb=200.0,
                search_peak_rss_mb=200.0,
                search_avg_rss_mb=195.0,
            ),
            disk_io=DiskIOMetrics(
                warmup_read_mb=150.0,
                search_avg_read_iops=18000.0,
                search_total_pages_read=7500000,
                search_pages_per_query=750.0,
                physical_block_size=4096,
                sample_count=500,
            ),
            latency=LatencyMetrics(mean_ms=5.0, p50_ms=4.8, p95_ms=6.5, p99_ms=9.0),
            recall=0.995,
            qps=185.0,
            total_build_time_seconds=320.0,
            index_size_bytes=1500000000,
            hyperparameters={"build": {"R": 64}, "search": {"Ls": 100}, "k": 10},
        )
        summary = result.to_summary_dict()

        # Check top-level structure
        assert summary["algorithm"] == "DiskANN"
        assert summary["dataset"] == "sift-1m"

        # Check quality metrics
        assert summary["quality"]["recall"] == 0.995
        assert summary["quality"]["qps"] == 185.0

        # Check build phase
        assert summary["build"]["duration_seconds"] == 320.0
        assert summary["build"]["cpu_time_seconds"] == 100.0
        assert summary["build"]["peak_rss_mb"] == 4000.0
        assert summary["build"]["index_size_bytes"] == 1500000000

        # Check warmup phase
        assert summary["warmup"]["cpu_time_seconds"] == 5.0
        assert summary["warmup"]["peak_rss_mb"] == 200.0
        assert summary["warmup"]["read_mb"] == 150.0

        # Check search phase
        assert summary["search"]["cpu_time_seconds"] == 25.0
        assert summary["search"]["cpu_time_per_query_ms"] == 2.5
        assert summary["search"]["avg_rss_mb"] == 195.0

        # Check disk I/O nested under search
        assert summary["search"]["disk_io"]["avg_read_iops"] == 18000.0
        assert summary["search"]["disk_io"]["total_pages_read"] == 7500000
        assert summary["search"]["disk_io"]["pages_per_query"] == 750.0

        # Check latency
        assert summary["latency"]["mean_ms"] == 5.0
        assert summary["latency"]["p99_ms"] == 9.0

        # Check metadata
        assert summary["metadata"]["physical_block_size"] == 4096
        assert summary["metadata"]["sample_count"] == 500

        # Check hyperparameters
        assert summary["hyperparameters"]["build"]["R"] == 64
        assert summary["hyperparameters"]["k"] == 10


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
        result = expand_sweep_params(
            {
                "ef": [50, 100],
                "num_threads": 4,
                "batch_size": 1000,
            }
        )
        assert len(result) == 2
        for combo in result:
            assert combo["num_threads"] == 4
            assert combo["batch_size"] == 1000
