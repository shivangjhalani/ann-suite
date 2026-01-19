# ANN Benchmarking Suite

A production-grade **Approximate Nearest Neighbor (ANN) Benchmarking Suite** with:

- 🐳 **Containerized Isolation**: Every algorithm runs in its own Docker container
- 📊 **Deep Observability**: Structured metrics for CPU, Memory, Disk I/O, Latency, and Algorithm-specific data
- 🔧 **cgroups v2 Integration**: Accurate IOPS and throughput from kernel-level metrics
- 💾 **Storage Modes**: Evaluates both in-memory and disk-based algorithms
- 📈 **Parameter Sweeps**: Test multiple parameter combinations in a single run
- ⚙️ **Modular & Configurable**: YAML configs, pluggable algorithms and datasets, extensible metrics

## Quick Start

```bash
# Install dependencies
cd ann-suite && uv sync

# List available datasets
uv run ann-suite download --list

# Download test datasets
uv run ann-suite download --dataset sift-10k

# Build algorithm containers
uv run ann-suite build --algorithm HNSW
uv run ann-suite build --algorithm DiskANN

# Run benchmark
uv run ann-suite run --config configs/hnsw_vs_diskann.yaml
```

## Features

### Structured Metrics

| Category | Metrics |
|----------|---------|
| **CPU** | `cpu_time_total_seconds`, `avg_cpu_percent`, `peak_cpu_percent` |
| **Memory** | `peak_rss_mb`, `avg_rss_mb` |
| **Disk I/O** | `avg_read_iops`, `avg_write_iops`, `total_pages_read`, `pages_per_query` |
| **Latency** | `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms` |

### Parameter Sweeps

Test multiple parameter values in a single benchmark run:

```yaml
algorithms:
  - name: HNSW
    search:
      args:
        ef: [50, 100, 200, 400]  # Runs 4 benchmarks with different ef values
```

### cgroups v2 Collector

Automatically uses cgroups v2 for accurate I/O metrics when available:

```
INFO: CgroupsV2Collector available - will collect enhanced I/O metrics
```

## Requirements

- Python 3.12+
- Docker 24.0+
- uv (package manager)
- Linux with cgroups v2 (recommended for I/O metrics)

## Documentation

| Document | Description |
|----------|-------------|
| [METRICS.md](docs/METRICS.md) | Complete metrics reference |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Config file options |
| [ADDING_ALGORITHMS.md](docs/ADDING_ALGORITHMS.md) | How to add new algorithms |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |

## Available Algorithms

| Algorithm | Type | Library | Description |
|-----------|------|---------|-------------|
| HNSW | memory | hnswlib | Graph-based search with excellent recall-QPS tradeoff |
| DiskANN | disk | diskannpy | Microsoft's billion-scale disk-based algorithm |

## Available Datasets

| Dataset | Dimension | Distance | Vectors |
|---------|-----------|----------|---------|
| sift-10k | 128 | L2 | 10K |
| glove-25-10k | 25 | cosine | 10K |

## License

MIT
