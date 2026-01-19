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

# Access fields
result.run_id          # UUID for log correlation
result.algorithm       # Algorithm name
result.dataset         # Dataset name
result.recall          # Recall@k value
result.qps             # Queries per second

# Nested structured metrics
result.latency.mean_ms # Mean latency
result.latency.p50_ms  # P50 latency
result.latency.p95_ms  # P95 latency
result.latency.p99_ms  # P99 latency
result.memory.peak_rss_mb  # Peak RAM usage
result.cpu.avg_cpu_percent # Average CPU utilization
result.disk_io.avg_read_iops  # Read IOPS (for disk-based algorithms)

result.build_result    # PhaseResult for build
result.search_result   # PhaseResult for search
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

storage = ResultsStorage(results_dir=Path("./results"))

# Save results
storage.save(results, format="json")   # Full JSON with samples
storage.save(results, format="csv")    # Flat CSV table
storage.save(results, format="parquet")  # Columnar format

# Load results
loaded = storage.load("results/benchmark_20240117.json")
```

**Methods:**

##### `save(results: list[BenchmarkResult], format: str = "json")`

Save benchmark results.

**Parameters:**
- `results`: List of BenchmarkResult objects
- `format`: Output format (`"json"`, `"csv"`, `"parquet"`)

##### `load(path: Path) -> list[BenchmarkResult]`

Load benchmark results from file.

---

### `ann_suite.monitoring.resource_monitor`

Resource monitoring for containers.

#### `ResourceMonitor`

```python
from ann_suite.monitoring.resource_monitor import ResourceMonitor

# Usually used internally by ContainerRunner
# But can be used directly for custom monitoring

monitor = ResourceMonitor(container, interval_ms=100)
monitor.start()

# ... container runs ...

summary = monitor.stop()  # Returns ResourceSummary
```

---

### `ann_suite.runners.container_runner`

Docker container management.

#### `ContainerRunner`

```python
from ann_suite.runners.container_runner import ContainerRunner

runner = ContainerRunner(
    algo_config=algo_config,
    dataset_config=dataset_config,
    base_config=benchmark_config,
)

# Pull image
runner.pull_image()

# Run build phase
build_result = runner.run_build(
    data_path=Path("/data/base.npy"),
    index_path=Path("/data/index/algo/dataset"),
)

# Run search phase
search_result = runner.run_search(
    index_path=Path("/data/index/algo/dataset"),
    queries_path=Path("/data/queries.npy"),
    ground_truth_path=Path("/data/ground_truth.npy"),
)
```

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
        "p99_latency_ms": r.p99_latency_ms,
        "peak_memory_mb": r.peak_memory_mb,
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
