# Python API Reference

This document provides the Python API reference for programmatic usage of the ANN Benchmarking Suite.

## Installation

```bash
cd ann-suite
uv sync
```

## Quick Start

```python
from ann_suite.core.config import load_config
from ann_suite.evaluator import BenchmarkEvaluator

# Load configuration
config = load_config("configs/benchmark.yaml")

# Run benchmarks
evaluator = BenchmarkEvaluator(config)
results = evaluator.run()

# Access results
for result in results:
    print(f"{result.algorithm} on {result.dataset}:")
    print(f"  Recall: {result.recall}")
    print(f"  QPS: {result.qps}")
```

---

## Core Modules

### `ann_suite.core.config`

Configuration loading and saving.

```python
from ann_suite.core.config import load_config

# Load from YAML or JSON
config = load_config("path/to/config.yaml")
config = load_config("path/to/config.json")
```

#### `load_config(path: str | Path) -> BenchmarkConfig`

Load and validate a benchmark configuration.

**Parameters:**
- `path`: Path to YAML or JSON configuration file

**Returns:**
- `BenchmarkConfig`: Validated configuration object

**Raises:**
- `ValidationError`: If configuration is invalid
- `FileNotFoundError`: If file doesn't exist

---

### `ann_suite.core.schemas`

Pydantic models for all data structures.

#### `BenchmarkConfig`

Top-level benchmark configuration.

```python
from ann_suite.core.schemas import BenchmarkConfig

config = BenchmarkConfig(
    name="My Benchmark",
    data_dir=Path("./data"),
    results_dir=Path("./results"),
    index_dir=Path("./indices"),
    algorithms=[...],
    datasets=[...],
    monitor_interval_ms=100,
)
```

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | `"ANN Benchmark"` | Benchmark name |
| `description` | str | `""` | Description |
| `data_dir` | Path | `"./data"` | Data directory |
| `results_dir` | Path | `"./results"` | Results directory |
| `index_dir` | Path | `"./indices"` | Index directory |
| `algorithms` | list[AlgorithmConfig] | `[]` | Algorithms to benchmark |
| `datasets` | list[DatasetConfig] | `[]` | Datasets to use |
| `monitor_interval_ms` | int | `100` | Monitoring interval |

#### `AlgorithmConfig`

Algorithm configuration.

```python
from ann_suite.core.schemas import AlgorithmConfig, AlgorithmType, BuildConfig, SearchConfig

algo = AlgorithmConfig(
    name="HNSW",
    docker_image="ann-suite/hnsw:latest",
    algorithm_type=AlgorithmType.MEMORY,
    cpu_limit="0-3",
    memory_limit="8g",
    build=BuildConfig(
        timeout_seconds=3600,
        args={"M": 16, "ef_construction": 200}
    ),
    search=SearchConfig(
        timeout_seconds=600,
        k=10,
        args={"ef": 100}
    ),
)
```

#### `DatasetConfig`

Dataset configuration.

```python
from ann_suite.core.schemas import DatasetConfig, DistanceMetric

dataset = DatasetConfig(
    name="sift-10k",
    base_path=Path("sift-10k/base.npy"),
    query_path=Path("sift-10k/queries.npy"),
    ground_truth_path=Path("sift-10k/ground_truth.npy"),
    distance_metric=DistanceMetric.L2,
    dimension=128,
    point_type="float32",
)
```

#### `BenchmarkResult`

Benchmark result structure.

```python
from ann_suite.core.schemas import BenchmarkResult

# Results are returned by BenchmarkEvaluator.run()
result: BenchmarkResult

# Identification
result.run_id          # UUID for log correlation
result.algorithm       # Algorithm name
result.dataset         # Dataset name
result.timestamp       # datetime when benchmark ran

# Quality metrics
result.recall          # Recall@k value (0.0 to 1.0)
result.qps             # Queries per second

# Build summary
result.total_build_time_seconds  # Total build wall time
result.index_size_bytes          # Index size on disk

# Hyperparameters used
result.hyperparameters           # {"build": {...}, "search": {...}, "k": 10}

# Latency metrics (from container output)
result.latency.mean_ms   # Mean query latency (ms)
result.latency.p50_ms    # Median latency (ms)
result.latency.p95_ms    # 95th percentile (ms)
result.latency.p99_ms    # 99th percentile - tail latency (ms)

# Memory metrics (phase-separated)
result.memory.build_peak_rss_mb   # Peak RAM during build (MB)
result.memory.load_peak_rss_mb    # Peak RAM during index load (MB)
result.memory.search_peak_rss_mb  # Peak RAM during search (MB)
result.memory.search_avg_rss_mb   # Average RAM during search (MB)

# CPU metrics (phase-separated)
result.cpu.build_cpu_time_seconds      # CPU time during build (s)
result.cpu.build_peak_cpu_percent      # Peak CPU during build (%)
result.cpu.load_cpu_time_seconds       # CPU time during index load (s)
result.cpu.load_peak_cpu_percent       # Peak CPU during index load (%)
result.cpu.search_cpu_time_seconds     # CPU time during search (s)
result.cpu.search_avg_cpu_percent      # Average CPU during search (%)
result.cpu.search_peak_cpu_percent     # Peak CPU during search (%)
result.cpu.search_cpu_time_per_query_ms  # CPU time per query (ms)

# Disk I/O metrics (CRITICAL for disk-based algorithms)
result.disk_io.avg_read_iops            # Read IOPS from cgroups
result.disk_io.avg_write_iops           # Write IOPS from cgroups
result.disk_io.avg_read_throughput_mbps # Read throughput (MB/s)
result.disk_io.avg_write_throughput_mbps# Write throughput (MB/s)
result.disk_io.total_pages_read         # Total 4KB pages read
result.disk_io.total_pages_written      # Total 4KB pages written
result.disk_io.pages_per_query          # Average pages per query (or None)

# Phase results (contain raw ResourceSummary)
result.build_result    # PhaseResult for build phase
result.search_result   # PhaseResult for search phase

# Convert to flat dict for DataFrame/CSV export
flat = result.to_flat_dict()
```

#### `PhaseResult`

Result from a single benchmark phase.

```python
from ann_suite.core.schemas import PhaseResult

phase: PhaseResult

phase.phase              # "build" or "search"
phase.success            # bool - did the phase complete successfully?
phase.error_message      # Error message if failed, else None
phase.duration_seconds   # Wall clock time for this phase
phase.resources          # ResourceSummary with monitoring data
phase.load_resources     # ResourceSummary for load sub-phase (search only)
phase.output             # dict - raw JSON output from container
phase.time_bases         # TimeBases with explicit time denominators
phase.stdout_path        # Path to stdout log file
phase.stderr_path        # Path to stderr log file
```

#### `ResourceSummary`

Aggregated resource metrics.

```python
from ann_suite.core.schemas import ResourceSummary

summary: ResourceSummary

summary.peak_memory_mb    # Peak memory in MB
summary.avg_memory_mb     # Average memory
summary.peak_cpu_percent  # Peak CPU %
summary.avg_cpu_percent   # Average CPU %
summary.total_blkio_read_mb   # Total disk reads
summary.total_blkio_write_mb  # Total disk writes
summary.avg_read_iops     # Average read IOPS
summary.avg_write_iops    # Average write IOPS
```

---

### `ann_suite.evaluator`

Benchmark evaluation engine.

#### `BenchmarkEvaluator`

Main evaluator class.

```python
from ann_suite.evaluator import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(config)
results = evaluator.run()
evaluator.cleanup()
```

**Methods:**

##### `__init__(config: BenchmarkConfig)`

Initialize evaluator with configuration.

##### `run() -> list[BenchmarkResult]`

Run the complete benchmark suite.

**Returns:**
- List of `BenchmarkResult` objects for all algorithm/dataset combinations

##### `cleanup()`

Clean up resources (containers, temporary files).

#### `run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]`

Convenience function for running benchmarks.

```python
from ann_suite.evaluator import run_benchmark
from ann_suite.core.config import load_config

config = load_config("config.yaml")
results = run_benchmark(config)
```

---

### `ann_suite.results.storage`

Result storage and retrieval.

#### `ResultsStorage`

```python
from ann_suite.results.storage import ResultsStorage
from pathlib import Path

storage = ResultsStorage(results_dir=Path("./results"))

# Save results (automatically saves multiple formats)
run_dir = storage.save(
    results,
    run_name="My Benchmark",       # Optional: adds timestamp automatically
    formats=["json", "csv"],       # Optional: defaults to ["json", "csv"]
)
# Creates: results/My Benchmark_2026-01-29_10-30-00/
#   ├── results.json          # Flattened summary
#   ├── results_detailed.json # Full results with phase details
#   └── results.csv           # Flat table

# Load results (by run name or latest)
loaded = storage.load(run_name="My Benchmark_2026-01-29_10-30-00")
loaded = storage.load()  # Loads latest run

# Load as pandas DataFrame
df = storage.load_dataframe(run_name="My Benchmark_2026-01-29_10-30-00")
df = storage.load_dataframe()  # Latest run
```

**Methods:**

##### `save(results: list[BenchmarkResult], run_name: str | None = None, formats: list[str] | None = None) -> Path`

Save benchmark results to multiple formats.

**Parameters:**
- `results`: List of BenchmarkResult objects
- `run_name`: Optional name for this run (timestamp appended automatically)
- `formats`: List of output formats: `["json", "csv", "parquet"]`. Defaults to `["json", "csv"]`

**Returns:**
- `Path` to the created run directory

##### `load(run_name: str | None = None) -> list[BenchmarkResult]`

Load benchmark results from a run directory.

**Parameters:**
- `run_name`: Specific run directory name, or `None` to load the latest run

**Returns:**
- List of `BenchmarkResult` objects

##### `load_dataframe(run_name: str | None = None) -> pd.DataFrame`

Load benchmark results as a pandas DataFrame.

**Parameters:**
- `run_name`: Specific run directory name, or `None` to load the latest run

**Returns:**
- `pd.DataFrame` with flattened metrics

#### Convenience Functions

```python
from ann_suite.results.storage import store_results, load_results

# Store results
run_dir = store_results(results, results_dir=Path("./results"), run_name="My Benchmark")

# Load results
loaded = load_results(results_dir=Path("./results"), run_name="My Benchmark_2026-01-29_10-30-00")
```

---


### `ann_suite.runners.container_runner`

Docker container management.

#### `ContainerRunner`

 ```python
 from ann_suite.runners.container_runner import ContainerRunner
 from pathlib import Path

 runner = ContainerRunner(
     data_dir=Path("./data"),
     index_dir=Path("./indices"),
     results_dir=Path("./results"),
     monitor_interval_ms=100,
 )

 # Pull image
 runner.pull_image("ann-suite/hnsw:latest")

 # Run build phase
 build_config = {
     "dataset_path": "/data/sift-10k/base.npy",
     "index_path": "/data/index/HNSW/sift-10k",
     "dimension": 128,
     "metric": "L2",
     "build_args": {"M": 16, "ef_construction": 200}
 }

 result, resources = runner.run_phase(
     algorithm=algo_config,
     mode="build",
     config=build_config,
 )

# Run search phase
search_config = {
    "index_path": "/data/index/HNSW/sift-10k",
    "queries_path": "/data/sift-10k/queries.npy",
    "ground_truth_path": "/data/sift-10k/ground_truth.npy",  # Optional
    "k": 10,
    "batch_mode": True,  # Enable batch processing for high QPS
    "search_args": {"ef": 100},
    "dimension": 128,
    "metric": "L2"
}

result, resources = runner.run_phase(
    algorithm=algo_config,
    mode="search",
    config=search_config,
)
```

**Volume Mounts:**

The `ContainerRunner` automatically mounts these volumes:

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `data_dir` | `/data` | rw | Datasets |
| `index_dir` | `/data/index` | rw | Index storage |
| `results_dir` | `/results` | rw | Optional outputs |

> [!IMPORTANT]
> Disk-based algorithms **must** write indices to `/data/index/` for accurate I/O metrics.

---

## Programmatic Usage Examples

### Running a Single Algorithm

```python
from ann_suite.core.schemas import (
    BenchmarkConfig, AlgorithmConfig, DatasetConfig,
    AlgorithmType, DistanceMetric, BuildConfig, SearchConfig
)
from ann_suite.evaluator import BenchmarkEvaluator

# Define configuration programmatically
config = BenchmarkConfig(
    name="Single Algorithm Test",
    data_dir=Path("./library/datasets"),
    results_dir=Path("./results"),
    index_dir=Path("./indices"),
    algorithms=[
        AlgorithmConfig(
            name="HNSW",
            docker_image="ann-suite/hnsw:latest",
            algorithm_type=AlgorithmType.MEMORY,
            build=BuildConfig(args={"M": 16, "ef_construction": 200}),
            search=SearchConfig(k=10, args={"ef": 100}),
        )
    ],
    datasets=[
        DatasetConfig(
            name="sift-10k",
            base_path=Path("sift-10k/base.npy"),
            query_path=Path("sift-10k/queries.npy"),
            ground_truth_path=Path("sift-10k/ground_truth.npy"),
            distance_metric=DistanceMetric.L2,
            dimension=128,
        )
    ],
)

evaluator = BenchmarkEvaluator(config)
results = evaluator.run()
print(f"Recall: {results[0].recall}, QPS: {results[0].qps}")
```

### Parameter Sweep

```python
from ann_suite.core.schemas import AlgorithmConfig, BuildConfig, SearchConfig

# Create multiple configurations
ef_values = [50, 100, 200, 400]
algorithms = []

for ef in ef_values:
    algo = AlgorithmConfig(
        name=f"HNSW-ef{ef}",
        docker_image="ann-suite/hnsw:latest",
        build=BuildConfig(args={"M": 16, "ef_construction": 200}),
        search=SearchConfig(k=10, args={"ef": ef}),
    )
    algorithms.append(algo)

config = BenchmarkConfig(
    name="HNSW Parameter Sweep",
    algorithms=algorithms,
    datasets=[dataset_config],
    # ...
)
```

### Analyzing Results

```python
import pandas as pd
from ann_suite.results.storage import ResultsStorage

# Load results
storage = ResultsStorage(Path("./results"))
results = storage.load("results/benchmark.json")

# Convert to DataFrame
data = []
for r in results:
    data.append({
        "algorithm": r.algorithm,
        "dataset": r.dataset,
        "recall": r.recall,
        "qps": r.qps,
        "p99_latency_ms": r.latency.p99_ms,
        "peak_memory_mb": r.memory.search_peak_rss_mb,
    })

df = pd.DataFrame(data)
print(df.to_string())

# Plot recall vs QPS
import matplotlib.pyplot as plt

for dataset in df["dataset"].unique():
    subset = df[df["dataset"] == dataset]
    plt.scatter(subset["recall"], subset["qps"], label=dataset)

plt.xlabel("Recall")
plt.ylabel("QPS")
plt.legend()
plt.show()
```

---

## Type Hints

All public APIs include type hints for IDE support:

```python
from ann_suite.core.schemas import BenchmarkConfig, BenchmarkResult
from ann_suite.evaluator import BenchmarkEvaluator

def run_my_benchmark(config_path: str) -> list[BenchmarkResult]:
    config: BenchmarkConfig = load_config(config_path)
    evaluator: BenchmarkEvaluator = BenchmarkEvaluator(config)
    results: list[BenchmarkResult] = evaluator.run()
    return results
```

---

## Error Handling

```python
from pydantic import ValidationError

try:
    config = load_config("bad_config.yaml")
except ValidationError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError:
    print("Config file not found")

try:
    evaluator = BenchmarkEvaluator(config)
    results = evaluator.run()
except DockerException as e:
    print(f"Docker error: {e}")
finally:
    evaluator.cleanup()
```
