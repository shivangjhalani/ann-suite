# ANN Benchmarking Suite Documentation

Welcome to the ANN Benchmarking Suite documentation. This suite provides production-grade benchmarking for Approximate Nearest Neighbor (ANN) algorithms with containerized isolation and deep observability.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./ARCHITECTURE.md) | Internal workings, components, and data flow |
| [Configuration Reference](./CONFIGURATION.md) | Complete YAML configuration options |
| [Adding Algorithms](./ADDING_ALGORITHMS.md) | How to add new algorithm implementations |
| [Adding Datasets](./ADDING_DATASETS.md) | How to add and manage datasets |
| [Metrics Reference](./METRICS.md) | All metrics collected and how they're measured |
| [Visualization Dashboard](./DASHBOARD.md) | Web dashboard for exploring and comparing results |
| [API Reference](./API.md) | Python API for programmatic usage |
| [Docker Optimizations](./DOCKER_OPTIMIZATIONS.md) | Runtime settings for research-grade performance |

## Quick Links

- **Getting Started**: See the [README](../README.md) for installation and quick start
- **Example Configs**: Check `configs/` directory for working examples
- **Library**: Algorithm and dataset implementations in `library/`

## Features

- 🐳 **Containerized Isolation**: Every algorithm runs in its own Docker container
- 📊 **Deep Observability**: Measures QPS, latency, recall, RAM, and Disk IOPS
- 💾 **Storage Modes**: Evaluates both in-memory (HNSW) and disk-based (DiskANN) algorithms
- ⚙️ **Modular & Configurable**: YAML/JSON configs, pluggable algorithms, extensible metrics

## Requirements

> [!IMPORTANT]
> **cgroups v2 is required** for running benchmarks. The suite will fail at startup if cgroups v2 is not available.

Verify with:
```bash
cat /sys/fs/cgroup/cgroup.controllers
# Should output: cpuset cpu io memory hugetlb pids rdma misc
```

See [METRICS.md](./METRICS.md#requirements) for setup instructions if cgroups v2 is not enabled.

## Project Structure

```
ann-suite/
├── src/ann_suite/            # Core benchmarking framework
│   ├── core/                 # Schemas (Pydantic models), config loading
│   │   ├── schemas.py        # BenchmarkConfig, AlgorithmConfig, *Metrics
│   │   └── config.py         # YAML/JSON loading and validation
│   ├── monitoring/           # Resource monitoring via cgroups v2
│   │   ├── base.py           # BaseCollector abstract class
│   │   └── cgroups_collector.py  # CgroupsV2Collector implementation
│   ├── runners/              # Docker container lifecycle
│   │   └── container_runner.py   # ContainerRunner with metrics
│   ├── datasets/             # Dataset loading utilities
│   ├── results/              # Result storage (JSON, CSV)
│   │   └── storage.py        # ResultsStorage class
│   ├── evaluator.py          # BenchmarkEvaluator pipeline
│   └── cli.py                # Typer CLI (run, build, report, download)
├── library/                  # Algorithm & dataset library
│   ├── algorithms/           # Algorithm implementations (HNSW, DiskANN)
│   │   ├── hnsw/             # HNSW container (Dockerfile + runner.py)
│   │   ├── diskann/          # DiskANN container
│   │   └── utils.py          # Shared utilities (compute_recall, etc.)
│   └── datasets/             # Dataset registry and download utilities
├── configs/                  # Benchmark configuration files (YAML)
├── docs/                     # This documentation
└── tests/                    # Test suite (pytest)
```
