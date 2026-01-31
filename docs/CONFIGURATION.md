# Configuration Reference

This document describes the YAML/JSON configuration used by the ANN Benchmarking Suite. It is kept
in sync with the Pydantic schemas in `ann_suite.core.schemas`.

## File Format

Configurations are written in YAML (or JSON) and parsed with Pydantic validation.

```yaml
# Minimal configuration
name: "My Benchmark"
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    dimension: 128
```

## Top-Level Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"ANN Benchmark"` | Benchmark name |
| `description` | string | `""` | Optional description |
| `data_dir` | path | `"./data"` | Base directory for datasets |
| `results_dir` | path | `"./results"` | Results output directory |
| `index_dir` | path | `"./indices"` | Index output directory |
| `monitor_interval_ms` | int | `100` | Resource sampling interval (50-1000) |
| `algorithms` | list | `[]` | Algorithm configurations |
| `datasets` | list | `[]` | Dataset configurations |

## Algorithm Configuration

Each entry in `algorithms` defines one containerized algorithm.

```yaml
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    algorithm_type: memory
    cpu_limit: "0-3"
    memory_limit: "8g"
    disabled: false
    env_vars:
      OMP_NUM_THREADS: "4"
    build:
      timeout_seconds: 3600
      args:
        M: 16
        ef_construction: 200
    search:
      timeout_seconds: 600
      k: 10
      args:
        ef: 100
```

### Algorithm Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | yes | - | Algorithm identifier |
| `docker_image` | string | yes | - | Docker image tag (missing tag defaults to `:latest`) |
| `algorithm_type` | enum | no | `memory` | `memory`, `disk`, or `hybrid` (informational) |
| `datasets` | list | no | `[]` | Dataset names to run on (empty means all) |
| `cpu_limit` | string | no | `null` | CPU affinity (e.g., `"0-3"` or `"0,2"`) |
| `memory_limit` | string | no | `null` | Memory limit (e.g., `"8g"`, `"512m"`) |
| `disabled` | bool | no | `false` | Skip this algorithm |
| `env_vars` | dict | no | `{}` | Environment variables for the container |
| `build` | object | no | `{}` | Build phase settings |
| `search` | object | no | `{}` | Search phase settings |

> [!NOTE]
> `algorithm_type` is currently informational only; it does not change runtime behavior.

> [!IMPORTANT]
> Disk-based algorithms must write indices under `/data/index/` inside the container for accurate
> I/O metrics. This path is mounted from `index_dir`.

### Build Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | int | `3600` | Build timeout (>= 60) |
| `args` | dict | `{}` | Algorithm-specific build arguments |

### Search Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | int | `600` | Search timeout (>= 10) |
| `k` | int | `10` | Neighbors to retrieve (1-1000) |
| `args` | dict | `{}` | Algorithm-specific search arguments (supports sweeps) |
| `batch_mode` | bool | `true` | Enable batch queries for higher QPS |
| `warmup` | object | `{}` | Warmup/cache settings |

### Warmup Configuration

"Warmup" has two meanings:
- Warmup phase: index loading before queries (always happens)
- Cache warmup queries: optional untimed queries after load

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collect_metrics` | bool | `true` | Report warmup phase metrics |
| `cache_warmup_queries` | int | `0` | Untimed queries after load |
| `drop_caches_before` | bool | `false` | Reserved for future use |

## Dataset Configuration

Each entry in `datasets` defines one dataset.

```yaml
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    query_path: sift-10k/queries.npy
    ground_truth_path: sift-10k/ground_truth.npy
    distance_metric: L2
    dimension: 128
    point_type: float32
    base_count: 10000
    query_count: 1000
```

### Dataset Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | yes | - | Dataset identifier |
| `base_path` | path | yes | - | Base vectors path (relative to `data_dir` if not absolute) |
| `query_path` | path | no | `base_path` | Query vectors path |
| `ground_truth_path` | path | no | `null` | Ground truth neighbors path |
| `distance_metric` | enum | no | `L2` | `L2`, `IP`, `cosine`, `hamming` |
| `dimension` | int | yes | - | Vector dimension (1-65536) |
| `point_type` | string | no | `float32` | `float32`, `uint8`, `int8`, etc. |
| `base_count` | int | no | `null` | Informational only |
| `query_count` | int | no | `null` | Informational only |

### Path Resolution

`base_path`, `query_path`, and `ground_truth_path` are resolved against `data_dir` when the path is
relative. Absolute paths are used as-is.

### Supported File Formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.npy` | NumPy array | Recommended, fastest loading |
| `.npz` | NumPy archive | Loads first array-like entry |
| `.hdf5`, `.h5` | HDF5 | Supports `train`, `test`, `neighbors` datasets |
| `.bin`, `.fbin`, `.ibin`, `.u8bin` | big-ann-benchmarks binary | Includes header with `n_vectors` + `dim` |

## Parameter Sweeps

List values in `build.args` or `search.args` expand to multiple runs.

```yaml
search:
  args:
    ef: [50, 100]
    num_threads: [1, 4]
```

This produces a cartesian product of parameter combinations.

## Algorithm-Dataset Mapping

Use `datasets` on an algorithm to scope it to specific datasets.

```yaml
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    datasets: ["sift-10k", "glove-25-10k"]
```

If `datasets` is omitted or empty, the algorithm runs on all datasets.

## Environment Variables

```yaml
algorithms:
  - name: MyAlgo
    env_vars:
      OMP_NUM_THREADS: "4"
      OPENBLAS_NUM_THREADS: "4"
      MKL_NUM_THREADS: "4"
      CUDA_VISIBLE_DEVICES: "0"
```

## Validation Rules

- `name`: minimum length 1
- `dimension`: 1 <= dimension <= 65536
- `monitor_interval_ms`: 50 <= interval <= 1000
- `build.timeout_seconds`: >= 60
- `search.timeout_seconds`: >= 10
- `search.k`: 1 <= k <= 1000

## Related Documentation

- Metrics details: `docs/METRICS.md`
- Container protocol: `docs/ADDING_ALGORITHMS.md`
- Dataset preparation: `docs/ADDING_DATASETS.md`
