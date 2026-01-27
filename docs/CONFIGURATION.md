# Configuration Reference

This document provides a complete reference for all configuration options in the ANN Benchmarking Suite.

## Configuration File Format

Configurations are written in YAML (or JSON) and parsed using Pydantic for validation.

```yaml
# Example minimal configuration
name: "My Benchmark"
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    dimension: 128
```

---

## Top-Level Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"ANN Benchmark"` | Human-readable benchmark name |
| `description` | string | `""` | Optional description |
| `data_dir` | path | `"./data"` | Base directory for datasets |
| `results_dir` | path | `"./results"` | Directory for benchmark results |
| `index_dir` | path | `"./indices"` | Directory for built indices |
| `monitor_interval_ms` | int | `100` | Resource sampling interval (50-1000) |
| `algorithms` | list | `[]` | List of algorithm configurations |
| `datasets` | list | `[]` | List of dataset configurations |

---

## Algorithm Configuration

Each algorithm is defined as an object in the `algorithms` list:

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
| `name` | string | ✓ | - | Unique algorithm identifier |
| `docker_image` | string | ✓ | - | Docker image tag (e.g., `ann-suite/hnsw:latest`) |
| `algorithm_type` | enum | | `memory` | Type: `memory`, `disk`, or `hybrid` |
| `datasets` | list | | `[]` | Dataset names to run on (empty = all datasets) |
| `cpu_limit` | string | | `null` | CPU core affinity (e.g., `"0-3"`, `"0,2,4"`) |
| `memory_limit` | string | | `null` | Memory limit (e.g., `"8g"`, `"512m"`) |
| `disabled` | bool | | `false` | Skip this algorithm in benchmark |
| `env_vars` | dict | | `{}` | Environment variables for container |
| `build` | object | | `{}` | Build phase configuration |
| `search` | object | | `{}` | Search phase configuration |

### Algorithm Types

| Type | Description | Examples |
|------|-------------|----------|
| `memory` | In-memory algorithms | HNSW, Annoy, FAISS-IVF |
| `disk` | Disk-based algorithms | DiskANN, SPANN |
| `hybrid` | Mixed memory/disk | Cached disk indices |

> **Note**: The `algorithm_type` field is currently **informational only**. It is used for categorization, display, and documentation purposes but does not change benchmark behavior. Both memory and disk algorithms run through the same pipeline. Future versions may use this to enable type-specific optimizations (e.g., automatic page cache clearing for disk-based algorithms).

### Build Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | int | `3600` | Maximum build time (≥60) |
| `args` | dict | `{}` | Algorithm-specific build arguments |

### Search Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout_seconds` | int | `600` | Maximum search time (≥10) |
| `k` | int | `10` | Number of neighbors to retrieve (1-1000) |
| `args` | dict | `{}` | Algorithm-specific search arguments |

---

## Dataset Configuration

Each dataset is defined as an object in the `datasets` list:

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
| `name` | string | ✓ | - | Unique dataset identifier |
| `base_path` | path | ✓ | - | Path to base vectors (relative to `data_dir`) |
| `query_path` | path | | `base_path` | Path to query vectors |
| `ground_truth_path` | path | | `null` | Path to ground truth neighbors |
| `distance_metric` | enum | | `L2` | Distance metric: `L2`, `IP`, `cosine`, `hamming` |
| `dimension` | int | ✓ | - | Vector dimension (1-65536) |
| `point_type` | string | | `float32` | Data type: `float32`, `uint8`, `int8`, etc. |
| `base_count` | int | | `null` | Number of base vectors (informational) |
| `query_count` | int | | `null` | Number of query vectors (informational) |

### Distance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `L2` | Euclidean distance | `√(Σ(a - b)²)` |
| `IP` | Inner product (dot product) | `Σ(a * b)` |
| `cosine` | Cosine similarity | `1 - (a · b) / (‖a‖ ‖b‖)` |
| `hamming` | Hamming distance (binary) | `popcount(a XOR b)` |

### Supported File Formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.npy` | NumPy array | Recommended, fastest loading |
| `.hdf5`, `.h5` | HDF5 | Supports datasets named `train`, `test`, `neighbors` |
| `.fvecs` | Float vectors | Common in ANN benchmarks |
| `.bvecs` | Byte vectors | For uint8 data |
| `.ivecs` | Integer vectors | For ground truth indices |

---

## Example Configurations

### Minimal Configuration

```yaml
name: "Quick Test"
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
datasets:
  - name: test
    base_path: test/vectors.npy
    dimension: 128
```

### Full Production Configuration

```yaml
name: "Production HNSW vs DiskANN Benchmark"
description: "Comparing in-memory and disk-based algorithms"

data_dir: "./library/datasets"
results_dir: "./results"
index_dir: "./indices"
monitor_interval_ms: 100

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
        num_threads: 4
    search:
      timeout_seconds: 600
      k: 10
      args:
        ef: 100

  - name: DiskANN
    docker_image: ann-suite/diskann:latest
    algorithm_type: disk
    cpu_limit: "0-3"
    memory_limit: "4g"
    env_vars:
      OMP_NUM_THREADS: "4"
    build:
      timeout_seconds: 7200
      args:
        R: 64
        L: 100
        alpha: 1.2
        num_threads: 4
    search:
      timeout_seconds: 600
      k: 10
      args:
        Ls: 100
        num_threads: 4

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

  - name: glove-25-10k
    base_path: glove-25-10k/base.npy
    query_path: glove-25-10k/queries.npy
    ground_truth_path: glove-25-10k/ground_truth.npy
    distance_metric: cosine
    dimension: 25
    point_type: float32
    base_count: 10000
    query_count: 1000
```

### Parameter Sweep Configuration

The suite supports **inline parameter sweeps** using list values. Each combination is tested automatically:

```yaml
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    build:
      args:
        M: 16
        ef_construction: 200
    search:
      args:
        # List values expand to multiple benchmark runs
        ef: [50, 100, 200, 400]  # → 4 separate runs
```

**Multi-parameter sweeps** generate cartesian products:

```yaml
search:
  args:
    ef: [50, 100]      # 2 values
    num_threads: [1, 4] # 2 values
    # → 4 runs: (50,1), (50,4), (100,1), (100,4)
```

**Console output shows progress:**
```
INFO: Running 4 parameter combinations for sweep
  [1/4] ef=50
  [2/4] ef=100
  [3/4] ef=200
  [4/4] ef=400
```


---

### Algorithm-Dataset Mapping

Run specific algorithms only on specific datasets:

```yaml
algorithms:
  - name: HNSW
    docker_image: ann-suite/hnsw:latest
    datasets: ["sift-10k", "glove-25-10k"]  # Runs on both
    build:
      args:
        M: 16

  - name: DiskANN
    docker_image: ann-suite/diskann:latest
    datasets: ["sift-10k"]  # Runs only on sift-10k
    build:
      args:
        R: 64

datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    dimension: 128
  - name: glove-25-10k
    base_path: glove-25-10k/base.npy
    dimension: 25
```

> **Note**: If `datasets` is omitted or empty, the algorithm runs on all datasets.

---

## Environment Variables

Environment variables can be passed to algorithm containers:

```yaml
algorithms:
  - name: MyAlgo
    env_vars:
      OMP_NUM_THREADS: "4"        # OpenMP thread count
      OPENBLAS_NUM_THREADS: "4"   # OpenBLAS thread count
      MKL_NUM_THREADS: "4"        # Intel MKL thread count
      CUDA_VISIBLE_DEVICES: "0"   # GPU selection
```

---

## Validation Rules

The configuration is validated using Pydantic with these constraints:

- `name`: Minimum length 1
- `dimension`: 1 ≤ dimension ≤ 65536
- `monitor_interval_ms`: 50 ≤ interval ≤ 1000
- `build.timeout_seconds`: ≥ 60
- `search.timeout_seconds`: ≥ 10
- `search.k`: 1 ≤ k ≤ 1000

Invalid configurations will raise validation errors with helpful messages.

---

## Reproducibility Recommendations

For reproducible and fair benchmark results, consider the following best practices:

### System Preparation

```bash
# Clear page cache before disk-based benchmarks (requires root)
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Disable CPU frequency scaling for consistent performance
sudo cpupower frequency-set -g performance
```

### Configuration Settings

```yaml
algorithms:
  - name: DiskANN
    docker_image: ann-suite/diskann:latest
    
    # Pin to specific CPU cores for consistent timing
    cpu_limit: "0-3"
    
    # Set memory limit to prevent swap usage
    memory_limit: "4g"
    
    # Control thread count explicitly
    env_vars:
      OMP_NUM_THREADS: "4"
```

### Checklist for Reproducibility

| Factor | Recommendation |
|--------|----------------|
| **Page Cache** | Clear before each disk-based benchmark run |
| **CPU Affinity** | Use `cpu_limit` to pin containers to specific cores |
| **Memory Limit** | Set explicit limits to prevent swap |
| **Thread Count** | Control via environment variables |
| **Warm-up** | Consider running a warm-up iteration before measurement |
| **Multiple Runs** | Report mean and variance from multiple runs |
| **Container Images** | Version and tag all Docker images |
| **Dataset Subsets** | Use deterministic sampling (suite uses seed=42) |

### What's NOT Controlled

The suite does not automatically control:

- Page cache state (must be cleared manually for disk benchmarks)
- CPU frequency scaling (handled at OS level)
- Background processes and system load
- Container image caching (first run may include image pull)

> **Tip**: For rigorous benchmarks, run each configuration multiple times and report the median with confidence intervals.

