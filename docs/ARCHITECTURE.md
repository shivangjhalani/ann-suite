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
│     │  - Execute: --mode build --config '{...}'                         │   │
│     │  - Collect: build_time, index_size, resource metrics              │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │  SEARCH PHASE                                                     │   │
│     │  - Run container with existing index                              │   │
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
# Configuration schemas:
BenchmarkConfig      # Top-level benchmark configuration
AlgorithmConfig      # Single algorithm definition
DatasetConfig        # Single dataset definition
BuildConfig          # Build phase parameters
SearchConfig         # Search phase parameters

# Result schemas:
ResourceSample       # Single monitoring sample
ResourceSummary      # Aggregated resource metrics
PhaseResult          # Result from build or search phase
BenchmarkResult      # Complete benchmark result

# Structured metrics schemas (organized by category):
CPUMetrics           # CPU time and utilization (by phase)
MemoryMetrics        # Peak and average RSS (by phase)
DiskIOMetrics        # IOPS, throughput, pages read/written
LatencyMetrics       # Query latency percentiles (mean, p50, p95, p99)
TimeBases            # Explicit time denominators for rate calculations

# Container protocol schemas:
ContainerProtocol.BuildInput    # JSON input for build phase
ContainerProtocol.BuildOutput   # Expected JSON output from build
ContainerProtocol.SearchInput   # JSON input for search phase
ContainerProtocol.SearchOutput  # Expected JSON output from search
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
                                 ▼
                 ┌─────────────────────────────────┐
                 │   CgroupsV2Collector            │
                 │   - Reads from cgroups v2       │
                 │   - io.stat: rios, wios, ...    │
                 │   - cpu.stat: usage_usec        │
                 │   - Accurate IOPS               │
                 └─────────────────────────────────┘
```

**CgroupsV2Collector** (preferred when available):
```
/sys/fs/cgroup/system.slice/docker-{id}.scope/
├── io.stat      → rbytes, wbytes, rios, wios  (Disk I/O)
├── cpu.stat     → usage_usec                  (CPU time)
└── memory.current                             (Memory)
```

**ContainerRunner Integration:**

The ContainerRunner uses the CgroupsV2Collector for metrics collection:

```python
# Initialization (requires cgroups v2)
if not CgroupsV2Collector.check_available():
    raise RuntimeError("cgroups v2 is required for metrics collection")

cgroups_collector = CgroupsV2Collector(interval_ms=monitor_interval_ms)

# During container execution
cgroups_collector.start(container_id)

# On completion
cgroups_result = cgroups_collector.stop()

# Get metrics filtered to query window (if timestamps available)
refined_result = cgroups_collector.get_summary(
    start_timestamp=query_start_dt,
    end_timestamp=query_end_dt
)
```

> [!IMPORTANT]
> The suite **requires** cgroups v2 and will fail at startup if it's not available.
> See `docs/METRICS.md` for setup instructions.


---

### 5. Results Storage (`src/ann_suite/results/storage.py`)

Handles saving and loading benchmark results:

```python
storage = ResultsStorage(results_dir)

# Save results (saves to multiple formats automatically)
storage.save(results, run_name="My Benchmark")
# Creates:
#   results/My Benchmark_2026-01-29_10-30-00/
#   ├── results.json          # Summary JSON
#   ├── results_detailed.json # Full JSON with phase details
#   └── results.csv           # Flat table for spreadsheets

# Load previous results by run name
loaded = storage.load(run_name="My Benchmark")

# Load as pandas DataFrame
df = storage.load_dataframe(run_name="My Benchmark")
```

**Output Formats:**
- **results.json**: Summary metrics (BenchmarkResult core fields)
- **results_detailed.json**: Complete results including phase details and samples
- **results.csv**: Flat table with flattened metrics for spreadsheet analysis

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
python -m algorithm.runner --mode search --config '{"index_path": "/data/index", "queries_path": "/data/queries.npy", "ground_truth_path": "/data/gt.npy", "k": 10, "batch_mode": true, "dimension": 128, "metric": "L2", "search_args": {...}}'
```

**Expected Output (stdout or `/results/metrics.json`):**
```json
{
    "status": "success",
    "total_queries": 1000,
    "total_time_seconds": 0.5,
    "qps": 2000.0,
    "recall": 0.95,
    "mean_latency_ms": 0.25,
    "p50_latency_ms": 0.20,
    "p95_latency_ms": 0.45,
    "p99_latency_ms": 0.80,
    "load_duration_seconds": 0.15,
    "load_start_timestamp": "2026-01-29T10:00:00.000000+00:00",
    "load_end_timestamp": "2026-01-29T10:00:00.150000+00:00",
    "query_start_timestamp": "2026-01-29T10:00:00.150000+00:00",
    "query_end_timestamp": "2026-01-29T10:00:00.650000+00:00"
}
```

> [!IMPORTANT]
> **Required for research-grade metrics**: The timestamp fields enable query-window filtering:
> - `load_start_timestamp` / `load_end_timestamp`: Index loading phase boundaries
> - `query_start_timestamp` / `query_end_timestamp`: Query execution boundaries
>
> Timestamps must be ISO-8601 format with timezone (UTC recommended).


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

### `download` Command

Download and prepare datasets for benchmarking.

```bash
ann-suite download [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dataset` | | *none* | Dataset name to download |
| `--output` | | `library/datasets/` | Output directory |
| `--list` | | `false` | List available datasets |
| `--quiet` | | `false` | Suppress output |

**Examples:**
```bash
# List available datasets
ann-suite download --list

# Download a specific dataset
ann-suite download --dataset sift-10k

# Download to custom location
ann-suite download --dataset sift-10k --output ./data
```

### `init-config` Command

Generate a sample configuration file.

```bash
ann-suite init-config [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `config.yaml` | Output configuration file path |

See `src/ann_suite/cli.py` for implementation.
