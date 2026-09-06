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
| `include_raw_samples` | bool | `false` | Also include raw samples in results_detailed.json (debug files always stored separately) |
| `algorithms` | list | `[]` | Algorithm configurations |
| `datasets` | list | `[]` | Dataset configurations |

## Algorithm Configuration

Each entry in `algorithms` defines one containerized algorithm.

```yaml
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    algorithm_type: memory
    cpu_affinity: "0-3"
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
| `cpu_affinity` | string | no | `null` | CPU core affinity/cpuset (e.g., `"0-3"` or `"0,2"`); pins container to these cores |
| `memory_limit` | string | no | `null` | Hard memory cap; swap is capped to the same value (e.g., `"8g"`, `"512m"`) |
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
| `prebuilt_path` | path | `null` | Existing index directory, relative to `index_dir` or absolute; skips the build phase |
| `args` | dict | `{}` | Algorithm-specific build arguments (supports sweeps) |
| `reuse_index` | bool | `true` | Reuse a built index across search sweep points instead of rebuilding per point |

### Prebuilt Indexes

Set `build.prebuilt_path` to benchmark an index that was built outside the current run:

```yaml
build:
  prebuilt_path: DiskANN/sift1m/R32_Lb50_fast
search:
  args:
    index_prefix: ann
    vector_dtype: uint8
```

The path is resolved relative to `index_dir` unless absolute. The evaluator validates the
directory, skips index construction, and mounts the resolved directory directly into the search
container. Build time and build resource metrics are reported as zero and marked `prebuilt`; index
load and query metrics are still collected normally.

The index must already use the naming convention required by its runner. For DiskANN with the
default `index_prefix: ann`, the directory must contain at least:

- `ann_disk.index`
- `ann_pq_pivots.bin`
- `ann_pq_compressed.bin`

Optional DiskANN files such as `ann_sample_data.bin` may also be present. DiskANN indexes created
by the native CLI often have no prefix. If the source already has the required names, avoid
duplicating large files by symlinking the whole index directory:

```bash
mkdir -p indices/DiskANN/sift1m
ln -s /path/to/normalized/index indices/DiskANN/sift1m/R32_Lb50_fast
```

The directory symlink target is resolved before Docker mounts the index, so an index outside
`index_dir` remains usable without duplicating storage. For native CLI files that need prefix
aliases, create a normalized wrapper directory with file symlinks; the evaluator mounts external
symlink targets automatically. The configured `vector_dtype`, dimension,
metric, and query dtype must match the prebuilt index. For example, SIFT indexes built from
DiskANN `u8bin` data require `vector_dtype: uint8` and uint8 queries.

### Search Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | int | `600` | Search timeout (>= 10) |
| `k` | int | `10` | Neighbors to retrieve (1-1000) |
| `args` | dict | `{}` | Algorithm-specific search arguments (supports sweeps) |
| `batch_mode` | bool | `false` | Research default: serial, per-query latency (percentiles). Set `true` for higher QPS but mean-only latency |
| `query_rounds` | int | `1` | Timed passes over the complete query set |
| `warmup` | object | `{}` | Warmup/cache settings |

### Warmup Configuration

"Warmup" has two meanings:
- Warmup phase: index loading before queries (always happens)
- Cache warmup queries: optional untimed queries after load

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collect_metrics` | bool | `true` | Report warmup phase metrics |
| `cache_warmup_queries` | int | `0` | Untimed queries after load |
| `drop_caches_before` | bool | `false` | Drop the OS page cache before each search phase (cold-start benchmarking). Requires root or sudo; set the `ANN_SUITE_SUDO_PASSWORD` env var for passworded sudo. On failure the search runs with a warm cache and a warning is logged. |

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

List values in `build.args` or `search.args` expand to multiple runs. The full
benchmark grid is the cartesian product of build combinations x search combinations.

```yaml
build:
  args:
    R: [32, 64]        # 2 distinct indices
search:
  args:
    Ls: [30, 50, 100]  # 3 search effort levels per index
```

This produces 6 benchmark points: each unique build combination is built **once**,
then every search combination runs against it.

### Index Reuse

By default (`build.reuse_index: true`), an index built for one search sweep point is
reused by the remaining points on the same build combination — a 4-point `Ls` sweep
builds once instead of four times. Indices are stored per build combination at
`index_dir/<algorithm>/<dataset>/<build-slug>/`, so different build parameters never
collide.

> [!NOTE]
> With reuse enabled, later search points may observe a warmer OS page cache than the
> first point (the index was just read by prior searches). For cold-cache studies set
> `reuse_index: false` (legacy behavior: rebuild for every point) or clear caches
> manually between runs.

Set `build.reuse_index: false` when build cost itself is under study or you need
identical cache conditions for every search point.

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
