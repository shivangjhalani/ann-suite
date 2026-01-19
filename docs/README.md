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
| [API Reference](./API.md) | Python API for programmatic usage |

## Quick Links

- **Getting Started**: See the [README](../README.md) for installation and quick start
- **Example Configs**: Check `configs/` directory for working examples
- **Library**: Algorithm and dataset implementations in `library/`

## Features

- 🐳 **Containerized Isolation**: Every algorithm runs in its own Docker container
- 📊 **Deep Observability**: Measures QPS, latency, recall, RAM, and Disk IOPS
- 💾 **Storage Modes**: Evaluates both in-memory (HNSW) and disk-based (DiskANN) algorithms
- ⚙️ **Modular & Configurable**: YAML/JSON configs, pluggable algorithms, extensible metrics

## Project Structure

```
ann-suite/
├── src/ann_suite/            # Core benchmarking framework
│   ├── core/                 # Schemas, config loading, base classes
│   ├── monitoring/           # Resource monitoring (RAM, IOPS)
│   ├── runners/              # Docker container lifecycle
│   ├── datasets/             # Dataset loading utilities
│   ├── results/              # Result storage and aggregation
│   └── cli.py                # Command-line interface
├── library/                  # Algorithm & dataset library
│   ├── algorithms/           # Algorithm implementations (HNSW, DiskANN)
│   └── datasets/             # Dataset registry and download utilities
├── configs/                  # Benchmark configuration files
├── docs/                     # This documentation
└── tests/                    # Test suite
```
