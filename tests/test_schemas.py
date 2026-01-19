"""Tests for ANN Suite schemas."""

from pathlib import Path

from ann_suite.core.schemas import (
    AlgorithmConfig,
    AlgorithmType,
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    DistanceMetric,
    ResourceSummary,
)


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig schema."""

    def test_valid_config(self):
        """Test valid algorithm configuration."""
        config = AlgorithmConfig(
            name="test-algo",
            docker_image="test/algo:latest",
            algorithm_type=AlgorithmType.MEMORY,
        )
        assert config.name == "test-algo"
        assert config.docker_image == "test/algo:latest"
        assert config.disabled is False

    def test_image_tag_default(self):
        """Test that :latest is added if no tag specified."""
        config = AlgorithmConfig(name="test", docker_image="test/algo")
        assert config.docker_image == "test/algo:latest"

    def test_with_build_args(self):
        """Test configuration with build arguments."""
        config = AlgorithmConfig(
            name="hnsw",
            docker_image="ann-suite/hnsw:v1",
            build={"args": {"M": 16, "ef_construction": 200}},
        )
        assert config.build.args["M"] == 16


class TestDatasetConfig:
    """Tests for DatasetConfig schema."""

    def test_valid_config(self):
        """Test valid dataset configuration."""
        config = DatasetConfig(
            name="test-dataset",
            base_path=Path("./data/base.npy"),
            dimension=128,
        )
        assert config.name == "test-dataset"
        assert config.distance_metric == DistanceMetric.L2

    def test_query_path_default(self):
        """Test that query_path defaults to base_path."""
        config = DatasetConfig(
            name="test",
            base_path=Path("./data/base.npy"),
            dimension=128,
        )
        assert config.query_path == config.base_path


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig schema."""

    def test_enabled_algorithms(self):
        """Test filtering of enabled algorithms."""
        config = BenchmarkConfig(
            algorithms=[
                AlgorithmConfig(name="a1", docker_image="img1", disabled=False),
                AlgorithmConfig(name="a2", docker_image="img2", disabled=True),
                AlgorithmConfig(name="a3", docker_image="img3", disabled=False),
            ]
        )
        enabled = config.enabled_algorithms
        assert len(enabled) == 2
        assert enabled[0].name == "a1"
        assert enabled[1].name == "a3"


class TestResourceSummary:
    """Tests for ResourceSummary schema."""

    def test_valid_summary(self):
        """Test valid resource summary."""
        summary = ResourceSummary(
            peak_memory_mb=1024.0,
            avg_memory_mb=512.0,
            avg_cpu_percent=50.0,
            peak_cpu_percent=100.0,
            total_blkio_read_mb=100.0,
            total_blkio_write_mb=50.0,
            avg_read_iops=1000.0,
            avg_write_iops=500.0,
            sample_count=100,
            duration_seconds=10.0,
        )
        assert summary.peak_memory_mb == 1024.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult schema."""

    def test_to_flat_dict(self):
        """Test flattening result for DataFrame."""
        result = BenchmarkResult(
            algorithm="test-algo",
            dataset="test-dataset",
            recall=0.95,
            qps=1000.0,
        )
        flat = result.to_flat_dict()
        assert flat["algorithm"] == "test-algo"
        assert flat["recall"] == 0.95
        assert "timestamp" in flat
