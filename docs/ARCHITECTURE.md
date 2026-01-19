# Architecture

This document explains the internal architecture of the ANN Benchmarking Suite, including component responsibilities, data flow, and design decisions.

## Overview

The suite follows a **pipeline architecture** where benchmarks flow through distinct phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Benchmark Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Configuration Loading                                                   │
│     - Parse YAML/JSON config                                                │
│     - Validate against Pydantic schemas                                     │
│     - Create AlgorithmConfig, DatasetConfig objects                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Dataset Preparation                                                     │
│     - Load base vectors, queries, ground truth                              │
│     - Copy to container-accessible paths                                    │
│     - Validate dimensions and formats                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. For Each (Algorithm, Dataset) Pair:                                     │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │  BUILD PHASE                                                      │   │
│     │  - Pull/create Docker container                                   │   │
│     │  - Mount data volumes                                             │   │
│     │  - Start ResourceMonitor                                          │   │
│     │  - Execute: --mode build --config '{...}'                         │   │
│     │  - Collect: build_time, index_size, resource metrics              │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │  SEARCH PHASE                                                     │   │
│     │  - Run container with existing index                              │   │
│     │  - Start ResourceMonitor                                          │   │
│     │  - Execute: --mode search --config '{...}'                        │   │
│     │  - Collect: recall, QPS, latencies, resource metrics              │   │
│     └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Results Aggregation & Storage                                           │
│     - Combine build and search results                                      │
│     - Save to JSON/CSV/Parquet                                              │
│     - Generate summary reports                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration Layer (`src/ann_suite/core/`)

#### `schemas.py` - Data Models

All configuration and result types are defined as Pydantic models for validation:

```python
# Key schemas:
BenchmarkConfig      # Top-level benchmark configuration
AlgorithmConfig      # Single algorithm definition
DatasetConfig        # Single dataset definition
BuildConfig          # Build phase parameters
SearchConfig         # Search phase parameters
ResourceSample       # Single monitoring sample
ResourceSummary      # Aggregated resource metrics
PhaseResult          # Result from build or search phase
BenchmarkResult      # Complete benchmark result
```

#### `config.py` - Configuration Loading

Loads and validates YAML/JSON configuration files:

```python
from ann_suite.core.config import load_config

config = load_config("configs/benchmark.yaml")
# Returns validated BenchmarkConfig object
```

---

### 2. Evaluator (`src/ann_suite/evaluator.py`)

The **BenchmarkEvaluator** orchestrates the entire benchmark pipeline:

```python
from ann_suite.core.config import load_config
from ann_suite.evaluator import BenchmarkEvaluator

config = load_config("config.yaml")
evaluator = BenchmarkEvaluator(config)

results = evaluator.run()  # Returns List[BenchmarkResult]
```

**Responsibilities:**
- Iterates through all (algorithm, dataset) combinations
- Manages container lifecycle via ContainerRunner
- Coordinates resource monitoring
- Aggregates results

---

### 3. Container Runner (`src/ann_suite/runners/container_runner.py`)

Manages Docker container lifecycle for algorithm execution:

```python
# Simplified flow:
runner = ContainerRunner(
    data_dir=Path("./data"),
    index_dir=Path("./indices"),
    results_dir=Path("./results"),
)

# Run build phase
build_result, resources = runner.run_phase(
    algorithm=algo_config,
    mode="build",
    config={"dataset_path": "/data/base.npy", "index_path": "/data/index", ...},
)

# Run search phase
search_result, resources = runner.run_phase(
    algorithm=algo_config,
    mode="search",
    config={"index_path": "/data/index", "queries_path": "/data/queries.npy", ...},
)
```

**Volume Mounting:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `config.data_dir` | `/data` | Datasets |
| `config.index_dir/<algo>/<dataset>` | `/data/index` | Index storage |
| `config.results_dir` | `/results` | Optional outputs |

> **CRITICAL**: Disk-based algorithms MUST write indices to `/data/index/` for accurate I/O metrics.

---

### 4. Monitoring Layer (`src/ann_suite/monitoring/`)

The monitoring layer uses a **modular collector architecture**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BaseCollector (Abstract)                         │
│  - start(container_id)                                              │
│  - stop() → CollectorResult                                         │
│  - is_available() → bool                                            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│   CgroupsV2Collector          │  │   ResourceMonitor (Docker)    │
│   - Reads from cgroups v2     │  │   - Uses Docker stats API     │
│   - io.stat: rios, wios, ...  │  │   - Fallback when no cgroups  │
│   - cpu.stat: usage_usec      │  │   - stream=True for faster    │
│   - Accurate IOPS             │  │     sample delivery           │
└───────────────────────────────┘  └───────────────────────────────┘
```

**CgroupsV2Collector** (preferred when available):
```
/sys/fs/cgroup/system.slice/docker-{id}.scope/
├── io.stat      → rbytes, wbytes, rios, wios  (Disk I/O)
├── cpu.stat     → usage_usec                  (CPU time)
└── memory.current                             (Memory)
```

**ContainerRunner Integration:**

The ContainerRunner automatically uses both collectors when cgroups v2 is available:

```python
# Initialization
if cgroups_collector.is_available():
    logger.info("CgroupsV2Collector available - will collect enhanced I/O metrics")

# During container execution
monitor.start()                  # Docker stats
cgroups_monitor.start(id)        # cgroups v2

# On completion - merge more accurate I/O metrics from cgroups
resources = monitor.stop()
cgroups_result = cgroups_monitor.stop()
# Merge cgroups I/O into resources (more accurate IOPS)
```

**Docker Stats API Collection**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Docker stats stream                                                │
│  - Extract memory_stats.usage                                       │
│  - Extract blkio_stats.io_service_bytes_recursive                   │
│  - Calculate CPU% from cpu_stats deltas                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ResourceSample (stored every interval_ms)                          │
│  - timestamp, memory_mb, cpu_percent                                │
│  - blkio_read_bytes, blkio_write_bytes                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ResourceSummary (aggregated on stop())                             │
│  - peak_memory_mb, avg_memory_mb                                    │
│  - peak_cpu_percent, avg_cpu_percent                                │
│  - total_blkio_read_mb, total_blkio_write_mb                        │
│  - avg_read_iops, avg_write_iops                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5. Results Storage (`src/ann_suite/results/storage.py`)

Handles saving and loading benchmark results:

```python
storage = ResultsStorage(results_dir)

# Save results
storage.save(results, format="json")  # or "csv", "parquet"

# Load previous results
loaded = storage.load("results/benchmark_20240117.json")
```

**Output Formats:**
- **JSON**: Complete with raw samples (detailed analysis)
- **CSV**: Flat table for spreadsheets
- **Parquet**: Columnar format for large-scale analysis

---

## Container Protocol

Algorithms communicate with the suite via a JSON-based CLI protocol:

### Build Phase

**Container Invocation:**
```bash
python -m algorithm.runner --mode build --config '{"dataset_path": "/data/base.npy", "index_path": "/data/index", "dimension": 128, "metric": "L2", "build_args": {...}}'
```

**Expected Output (stdout):**
```json
{
    "status": "success",
    "build_time_seconds": 45.2,
    "index_size_bytes": 104857600
}
```

### Search Phase

**Container Invocation:**
```bash
python -m algorithm.runner --mode search --config '{"index_path": "/data/index", "queries_path": "/data/queries.npy", "ground_truth_path": "/data/gt.npy", "k": 10, "dimension": 128, "metric": "L2", "search_args": {...}}'
```

**Expected Output (stdout):**
```json
{
    "status": "success",
    "total_queries": 1000,
    "qps": 5000.0,
    "recall": 0.95,
    "mean_latency_ms": 0.2,
    "p50_latency_ms": 0.15,
    "p95_latency_ms": 0.4,
    "p99_latency_ms": 0.8
}
```

---

## Design Decisions

### Why Docker Containers?

1. **Isolation**: No interference between algorithms (memory, CPU, dependencies)
2. **Reproducibility**: Container images are versioned and shareable
3. **Fair Comparison**: All algorithms run in identical environments
4. **Resource Control**: CPU/memory limits via Docker

### Why Host-Side Resource Monitoring?

1. **Accuracy**: Container-internal monitoring affects performance
2. **Consistency**: Same monitoring code for all algorithms
3. **Disk I/O**: Block-level I/O only visible from host via cgroups

### Why Volume Mounting for Indices?

For disk-based algorithms (DiskANN, SPANN), accurate I/O metrics require:

1. Index files written to host filesystem (not container overlay)
2. Block I/O tracked through cgroups
3. Index persists between build and search phases

---

## CLI Architecture

The CLI is built with Typer for a clean interface.

### `run` Command

Run a benchmark suite from a configuration file.

```bash
ann-suite run --config config.yaml [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | *required* | Path to benchmark config (YAML/JSON) |
| `--output` | `-o` | config value | Override results directory |
| `--log-level` | `-l` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--dry-run` | | `false` | Validate config without running benchmarks |

### `report` Command

Generate a report from benchmark results.

```bash
ann-suite report [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--results` | `-r` | `./results` | Results directory to load from |
| `--run` | | latest | Specific run name to report on |
| `--format` | `-f` | `table` | Output format: `table`, `json`, `csv` |

### `build` Command

Build a Docker image for an algorithm.

```bash
ann-suite build --algorithm HNSW [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--algorithm` | `-a` | *required* | Algorithm name to build |
| `--algorithms-dir` | `-d` | `./library/algorithms` | Path to algorithms directory |
| `--force` | `-f` | `false` | Force rebuild without cache |

### `init-config` Command

Generate a sample configuration file.

```bash
ann-suite init-config [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `config.yaml` | Output configuration file path |

See `src/ann_suite/cli.py` for implementation.
