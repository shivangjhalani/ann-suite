# ANN Benchmarking Suite

The new de-facto standard for production-grade Approximate Nearest Neighbor (ANN) benchmarking. 

Designed for **Deep Observability**, **Containerized Isolation**, and **Real-World Fidelity**, this suite solves the "it works on my machine" problem in ANN research by running every algorithm in a strictly isolated container with kernel-level resource monitoring.

## 🚀 Why Use This Suite?

- **Strict Isolation**: Every algorithm runs in its own Docker container. No dependency conflicts, no pollution.
- **Kernel-Level Observability**: Uses `cgroups v2` to measure **true** Disk I/O, Page Faults, and CPU usage, not just what the Python process reports.
- **Disk & Memory Support**: First-class support for disk-based algorithms (like DiskANN) with accurate I/O metrics.
- **Production Ready**: Configurable via YAML, supports parameter sweeps, and exports results to JSON/CSV for analysis.

---

## ⚡ Quick Start

### 1. Installation
We use `uv` for modern, fast Python management.

```bash
# Clone and sync dependencies
cd ann-suite
uv sync
```

### 2. Prepare Data
Download standard datasets or bring your own.

```bash
# List available datasets
uv run ann-suite download --list

# Download SIFT-10K (Small, good for testing)
uv run ann-suite download --dataset sift-10k
```

### 3. Build Algorithms
Algorithms are containerized. You must build them before running.

```bash
# Build HNSW (In-Memory)
uv run ann-suite build --algorithm HNSW

# Build DiskANN (Disk-Based)
uv run ann-suite build --algorithm DiskANN
```

### 4. Run a Benchmark
Run a comparison using a declarative config file.

```bash
# Run a comparison between HNSW and DiskANN
uv run ann-suite run --config configs/example.yaml
```

---

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[CONFIGURATION.md](docs/CONFIGURATION.md)**: Learn how to write benchmark config files, configure parameter sweeps, and tune resource limits.
- **[METRICS.md](docs/METRICS.md)**: Comprehensive reference for all collected metrics (Recall, QPS, Latency P99, Disk IOPS, etc.).
- **[ADDING_ALGORITHMS.md](docs/ADDING_ALGORITHMS.md)**: Guide to adding new algorithms (Python, C++, Rust, etc.) via Docker.
- **[ADDING_DATASETS.md](docs/ADDING_DATASETS.md)**: How to register custom datasets.
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Deep dive into the system design, isolation strategy, and metric collection pipeline.

---

## 🛠️ CLI Reference

The suite provides a rich CLI for managing the entire benchmarking lifecycle.

### `ann-suite run`
Execute a benchmark from a config file.
```bash
uv run ann-suite run --config configs/my_benchmark.yaml --output ./results
```

### `ann-suite report`
View or export results from previous runs.
```bash
# View as table
uv run ann-suite report --run "HNSW vs DiskANN"

# Export to JSON or CSV
uv run ann-suite report --format json > results.json
```

### `ann-suite build`
Build algorithm containers.
```bash
uv run ann-suite build --algorithm HNSW --force
```

### `ann-suite download`
Manage datasets.
```bash
uv run ann-suite download --dataset glove-25-angular
```

### `ann-suite init-config`
Generate a sample configuration file to get started.
```bash
uv run ann-suite init-config --output my_config.yaml
```

---

## License

MIT
