# Adding New Algorithms Guide

This guide explains how to add new ANN algorithm implementations to the benchmarking suite.

## Overview

All algorithms run in isolated Docker containers and communicate with the suite via a **JSON-based CLI protocol**. This design ensures:

1. **Fair isolation** - No interference between algorithms
2. **Reproducible results** - Container images are versioned
3. **Accurate resource monitoring** - Host-side metrics capture
4. **Support for disk-based algorithms** - Proper volume mounting

## Container Protocol

### Command Structure

The benchmark suite invokes your container with:

```bash
python -m algorithm.runner --mode <build|search> --config '<json_string>'
```

### Build Phase

**Input JSON:**
```json
{
    "dataset_path": "/data/dataset_name/base.npy",
    "index_path": "/data/index/algorithm_name/dataset_name",
    "dimension": 128,
    "metric": "L2",
    "build_args": {
        "M": 16,
        "ef_construction": 200
    }
}
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

**Input JSON:**
```json
{
    "index_path": "/data/index/algorithm_name/dataset_name",
    "queries_path": "/data/dataset_name/queries.npy",
    "ground_truth_path": "/data/dataset_name/ground_truth.npy",
    "k": 10,
    "batch_mode": true,
    "dimension": 128,
    "metric": "L2",
    "search_args": {
        "ef": 100
    }
}
```

**Expected Output (stdout or `/results/metrics.json`):**
```json
{
    "status": "success",
    "total_queries": 1000,
    "total_time_seconds": 0.5,
    "qps": 2000.0,
    "recall": 0.95,
    "mean_latency_ms": 0.5,
    "p50_latency_ms": 0.4,
    "p95_latency_ms": 0.8,
    "p99_latency_ms": 1.2,
    "load_duration_seconds": 0.1,
    "load_start_timestamp": "2026-01-28T21:46:45.000000+00:00",
    "load_end_timestamp": "2026-01-28T21:46:45.100000+00:00",
    "query_start_timestamp": "2026-01-28T21:46:45.100000+00:00",
    "query_end_timestamp": "2026-01-28T21:46:45.600000+00:00"
}
```

> [!WARNING]
> **Required for Research-Grade Metrics**: The timestamp fields enable accurate resource metrics:
> - `load_duration_seconds`: Time spent loading the index from disk
> - `load_start_timestamp`: ISO-8601 UTC timestamp when index loading began
> - `load_end_timestamp`: ISO-8601 UTC timestamp when index loading completed
> - `query_start_timestamp`: ISO-8601 UTC timestamp when query execution began
> - `query_end_timestamp`: ISO-8601 UTC timestamp when query execution ended
>
> These fields enable phase-separated resource filtering for fair algorithm comparison.
> Without them, all metrics are computed over the entire container lifetime.

## Critical: Volume Mounting

The suite mounts host directories into containers:

| Host Path (from config) | Container Path | Mode | Purpose |
|-------------------------|----------------|------|---------|
| `data_dir` | `/data` | rw | Datasets (base vectors, queries, ground truth) |
| `index_dir` | `/data/index` | rw | Index storage (persists between build/search) |
| `results_dir` | `/results` | rw | Optional outputs (e.g., `metrics.json`) |

**Example with default config:**
```
Host: ./data/sift-10k/base.npy    → Container: /data/sift-10k/base.npy
Host: ./indices/HNSW/sift-10k/    → Container: /data/index/HNSW/sift-10k/
Host: ./results/metrics.json      → Container: /results/metrics.json
```

### For Disk-Based Algorithms (DiskANN, SPANN, etc.)

> ⚠️ **CRITICAL**: You MUST write indices to `/data/index/` for:

1. **Accurate I/O metrics** - The monitor tracks block I/O from the host
2. **Index persistence** - Indices must survive between build and search phases
3. **Fair comparison** - Memory-only indices in overlay FS would not trigger disk metrics

**Example:**
```python
# CORRECT - writes to host-mounted directory
index_path = Path("/data/index/my_algo/dataset")
np.save(index_path / "graph.bin", graph_data)

# WRONG - writes to container overlay (not tracked)
index_path = Path("/tmp/index")
```

## Step-by-Step: Adding a New Algorithm

### 1. Create Directory Structure

```
library/algorithms/my_algorithm/
├── Dockerfile
├── requirements.txt
└── algorithm/
    ├── __init__.py
    └── runner.py
```

### 2. Implement Runner

```python
# algorithm/runner.py
import argparse
import json
import sys
from pathlib import Path

def run_build(config: dict) -> dict:
    # Load data
    data = np.load(config["dataset_path"])
    index_path = Path(config["index_path"])

    # Build your index
    index = MyAlgorithm(**config["build_args"])
    index.build(data)
    index.save(index_path)

    return {
        "status": "success",
        "build_time_seconds": build_time,
        "index_size_bytes": index.size_bytes(),
    }

def run_search(config: dict) -> dict:
    from datetime import datetime, timezone
    import time

    # Load index - track timing
    load_start = time.perf_counter()
    index = MyAlgorithm.load(config["index_path"])
    queries = np.load(config["queries_path"])
    load_duration_seconds = time.perf_counter() - load_start

    # Run search with timing - emit timestamps
    query_start_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    results, latencies = index.search(queries, k=config["k"])
    total_time = time.perf_counter() - start_time
    query_end_timestamp = datetime.now(timezone.utc).isoformat()

    # Compute recall if ground truth available
    recall = compute_recall(results, ground_truth)

    return {
        "status": "success",
        "total_queries": len(queries),
        "total_time_seconds": total_time,
        "qps": len(queries) / total_time,
        "recall": recall,
        # ... latency percentiles
        "load_duration_seconds": load_duration_seconds,
        "query_start_timestamp": query_start_timestamp,
        "query_end_timestamp": query_end_timestamp,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "search"])
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = json.loads(args.config)

    if args.mode == "build":
        result = run_build(config)
    else:
        result = run_search(config)

    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

### 3. Create Dockerfile

We recommend using `uv` for faster dependency installation:

```dockerfile
FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY algorithm/ ./algorithm/

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "algorithm.runner"]

VOLUME ["/data", "/results"]
```

### 4. Build and Register

```bash
# Build the image
uv run ann-suite build --algorithm MyAlgorithm

# Add to config
algorithms:
  - name: MyAlgorithm
    docker_image: ann-suite/my-algorithm:latest
    algorithm_type: memory  # or disk
    build:
      args:
        param1: value1
    search:
      k: 10
      args:
        search_param: value
```

---

## Reference Implementations

The suite includes production implementations you can use as templates:

| Algorithm | Path | Type | Library |
|-----------|------|------|---------|
| HNSW | `library/algorithms/hnsw/` | in-memory | hnswlib |
| DiskANN | `library/algorithms/diskann/` | disk-based | diskannpy |

### Key Files to Study

**HNSW (simpler, good starting point):**
- [runner.py](../library/algorithms/hnsw/algorithm/runner.py) - Complete implementation
- [Dockerfile](../library/algorithms/hnsw/Dockerfile) - Modern Docker setup with uv

**DiskANN (disk-based example):**
- [runner.py](../library/algorithms/diskann/algorithm/runner.py) - Disk index handling
- [Dockerfile](../library/algorithms/diskann/Dockerfile) - Python version considerations

---

## Optional: Shared Utilities

The suite provides optional shared utilities in `library/algorithms/utils.py` to reduce boilerplate code when implementing new algorithms. **These are completely optional** - you can write fully standalone runners like HNSW and DiskANN do.

### What's Included in `utils.py`

| Utility | Purpose |
|---------|---------|
| `compute_recall(predicted, ground_truth, k)` | Standard recall@k calculation |
| `compute_latency_percentiles(latencies)` | Compute mean, p50, p95, p99 latencies (ms) |

### When to Use Utilities

**Use `utils.py` when:**
- You want less boilerplate code
- You're implementing multiple algorithms and want consistency
- You prefer importing tested utilities over copy-pasting

**Write standalone runners when:**
- You want maximum control and transparency
- Your algorithm has unique requirements
- You're copying from an existing runner as a template

### Example: Using Shared Utilities

```python
# algorithm/runner.py
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timezone

# Import shared utilities (optional)
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # Add library/algorithms to path
from utils import compute_recall, compute_latency_percentiles

class MyIndex:
    def build(self, data, **kwargs): ...
    def search(self, query, k): ...

def run_build(config):
    data = np.load(config["dataset_path"])
    start = time.perf_counter()
    index = MyIndex()
    index.build(data, **config.get("build_args", {}))
    build_time = time.perf_counter() - start
    index.save(config["index_path"])
    return {
        "status": "success",
        "build_time_seconds": build_time,
        "index_size_bytes": index.size_bytes()
    }

def run_search(config):
    # Load index with timing
    load_start = datetime.now(timezone.utc)
    load_start_time = time.perf_counter()
    index = MyIndex.load(config["index_path"])
    queries = np.load(config["queries_path"])
    ground_truth = np.load(config.get("ground_truth_path", "")) if config.get("ground_truth_path") else None
    load_end = datetime.now(timezone.utc)
    load_duration = time.perf_counter() - load_start_time

    # Execute queries with timing
    query_start = datetime.now(timezone.utc)
    latencies = []
    results = []
    for query in queries:
        start = time.perf_counter()
        results.append(index.search(query, k=config["k"]))
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    query_end = datetime.now(timezone.utc)
    total_time = sum(latencies) / 1000  # seconds

    # Use shared utilities
    latency_stats = compute_latency_percentiles(latencies)
    recall = compute_recall(np.array(results), ground_truth, config["k"]) if ground_truth is not None else None

    return {
        "status": "success",
        "total_queries": len(queries),
        "total_time_seconds": total_time,
        "qps": len(queries) / total_time,
        "recall": recall,
        **latency_stats,
        # Required for research-grade metrics:
        "load_duration_seconds": load_duration,
        "load_start_timestamp": load_start.isoformat(),
        "load_end_timestamp": load_end.isoformat(),
        "query_start_timestamp": query_start.isoformat(),
        "query_end_timestamp": query_end.isoformat(),
    }
```

### Example: Standalone Runner (No Dependencies)

```python
# algorithm/runner.py - Fully self-contained, no imports from utils
import argparse
import json
import sys
import time
import numpy as np
from datetime import datetime, timezone

def compute_recall(predicted, ground_truth, k):
    """Compute recall@k."""
    total = 0.0
    gt_k = min(k, ground_truth.shape[1])
    for i in range(len(predicted)):
        pred = set(predicted[i, :k].tolist())
        true = set(ground_truth[i, :gt_k].tolist())
        total += len(pred & true) / gt_k
    return total / len(predicted)

def run_build(config): ...
def run_search(config): ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "search"])
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = json.loads(args.config)
    result = run_build(config) if args.mode == "build" else run_search(config)
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "success" else 1)

if __name__ == "__main__":
    main()
```

> [!NOTE]
> The existing HNSW and DiskANN implementations are standalone and do not use `utils.py`.
> They serve as complete, working examples of this approach.

## Debugging Tips

1. **Test locally first:**
   ```bash
   docker run -v $(pwd)/library/datasets:/data ann-suite/my-algo:latest \
     --mode build --config '{"dataset_path": "/data/sift-10k/base.npy", "index_path": "/data/index", "dimension": 128, "metric": "L2", "build_args": {}}'
   ```

2. **Check container logs:**
   ```bash
   docker logs <container_id>
   ```

3. **Verify volume mounts:**
   ```bash
   docker inspect <container_id> | jq '.[0].Mounts'
   ```

4. **Validate JSON output:**
   - Must be valid JSON
   - Must be the last line of stdout
   - Use stderr for debug messages

5. **Test search phase:**
   ```bash
   docker run -v $(pwd)/library/datasets:/data -v $(pwd)/indices:/data/index \
     ann-suite/my-algo:latest \
     --mode search --config '{"index_path": "/data/index", "queries_path": "/data/sift-10k/queries.npy", "k": 10, "dimension": 128, "metric": "L2", "search_args": {}}'
   ```

---

## Algorithm-Specific Considerations

### In-Memory Algorithms

- Set `algorithm_type: memory`
- Index lives in container memory
- Save index to `/data/index/` for persistence between runs
- Example: HNSW, Annoy, FAISS-IVF

### Disk-Based Algorithms

- Set `algorithm_type: disk`
- **CRITICAL**: Write all index files to `/data/index/`
- This enables accurate I/O metrics from host monitoring
- Example: DiskANN, SPANN

### GPU Algorithms

```yaml
algorithms:
  - name: FAISS-GPU
    docker_image: ann-suite/faiss-gpu:latest
    env_vars:
      CUDA_VISIBLE_DEVICES: "0"
```

Dockerfile:
```dockerfile
FROM nvidia/cuda:12.1-base
# ...
```

Run with:
```bash
docker run --gpus all ann-suite/faiss-gpu:latest ...
```

---

## Testing Your Implementation

### 1. Verify Build Phase

```bash
# Build container
uv run ann-suite build --algorithm Test

# Download test data
uv run ann-suite download --dataset sift-10k

# Run build
mkdir -p indices/test
docker run -v $(pwd)/library/datasets:/data \
  -v $(pwd)/indices/test:/data/index \
  ann-suite/test:latest \
  --mode build \
  --config '{"dataset_path": "/data/sift-10k/base.npy", "index_path": "/data/index", "dimension": 128, "metric": "L2", "build_args": {"M": 16}}'
```

Expected output:
```json
{"status": "success", "build_time_seconds": 12.5, "index_size_bytes": 5242880}
```

### 2. Verify Search Phase

```bash
docker run -v $(pwd)/library/datasets:/data \
  -v $(pwd)/indices/test:/data/index \
  ann-suite/test:latest \
  --mode search \
  --config '{"index_path": "/data/index", "queries_path": "/data/sift-10k/queries.npy", "ground_truth_path": "/data/sift-10k/ground_truth.npy", "k": 10, "dimension": 128, "metric": "L2", "search_args": {"ef": 100}}'
```

Expected output:
```json
{"status": "success", "total_queries": 1000, "qps": 5000.0, "recall": 0.95, "mean_latency_ms": 0.2, "p50_latency_ms": 0.15, "p95_latency_ms": 0.4, "p99_latency_ms": 0.8}
```

### 3. Integration Test

Add to a config and run full benchmark:

```yaml
# configs/test_new_algo.yaml
name: "Test New Algorithm"
data_dir: "./library/datasets"
algorithms:
  - name: MyAlgorithm
    docker_image: ann-suite/test:latest
datasets:
  - name: sift-10k
    base_path: sift-10k/base.npy
    query_path: sift-10k/queries.npy
    ground_truth_path: sift-10k/ground_truth.npy
    distance_metric: L2
    dimension: 128
```

```bash
uv run ann-suite run --config configs/test_new_algo.yaml
```
