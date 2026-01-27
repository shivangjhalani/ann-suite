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
    "search_args": {
        "ef": 100
    }
}
```

**Expected Output (stdout):**
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
    "p99_latency_ms": 1.2
}
```

## Critical: Volume Mounting

The suite mounts host directories into containers:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `data_dir` | `/data` | Datasets (base vectors, queries, ground truth) |
| `index_dir` | `/data/index` | Index storage |
| `results_dir` | `/results` | Optional additional outputs |

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
    # Load index and queries
    index = MyAlgorithm.load(config["index_path"])
    queries = np.load(config["queries_path"])

    # Run search with timing
    results, latencies = index.search(queries, k=config["k"])

    # Compute recall if ground truth available
    recall = compute_recall(results, ground_truth)

    return {
        "status": "success",
        "total_queries": len(queries),
        "qps": len(queries) / total_time,
        "recall": recall,
        # ... latency percentiles
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

## Optional: Shared Base Runner Utilities

The suite provides optional shared utilities in `library/algorithms/base_runner.py` to reduce boilerplate code when implementing new algorithms. **These are completely optional** - you can write fully standalone runners like HNSW and DiskANN do.

### What's Included

| Utility | Purpose |
|---------|---------|
| `compute_recall()` | Standard recall@k calculation |
| `compute_latency_percentiles()` | Compute mean, p50, p95, p99 latencies |
| `run_algorithm()` | CLI entrypoint wrapper with JSON parsing |
| `write_result_to_file()` | Write metrics.json for robust protocol |
| `measure_search_latencies()` | Per-query latency measurement loop |

### When to Use

**Use base_runner.py when:**
- You want less boilerplate code
- You're implementing multiple algorithms and want consistency
- You prefer importing tested utilities over copy-pasting

**Write standalone runners when:**
- You want maximum control and transparency
- Your algorithm has unique requirements
- You're copying from an existing runner as a template

### Example: Using Base Runner

```python
# algorithm/runner.py
import numpy as np
from pathlib import Path

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # Add library/algorithms to path
from base_runner import compute_recall, compute_latency_percentiles, run_algorithm

class MyIndex:
    def build(self, data, **kwargs): ...
    def search(self, query, k): ...

def run_build(config):
    data = np.load(config["dataset_path"])
    index = MyIndex()
    index.build(data)
    index.save(config["index_path"])
    return {"status": "success", "build_time_seconds": 1.0, "index_size_bytes": 1000}

def run_search(config):
    index = MyIndex.load(config["index_path"])
    queries = np.load(config["queries_path"])
    
    # Measure latencies
    latencies = []
    results = []
    for query in queries:
        start = time.perf_counter()
        results.append(index.search(query, k=config["k"]))
        latencies.append((time.perf_counter() - start) * 1000)
    
    # Use shared utilities
    latency_stats = compute_latency_percentiles(latencies)
    recall = compute_recall(np.array(results), ground_truth, config["k"])
    
    return {
        "status": "success",
        "recall": recall,
        "qps": len(queries) / sum(latencies) * 1000,
        **latency_stats,
    }

if __name__ == "__main__":
    run_algorithm("My Algorithm", run_build, run_search)
```

### Example: Standalone Runner (No Dependencies)

```python
# algorithm/runner.py - Fully self-contained, no imports from base_runner
import argparse
import json
import sys
import time
import numpy as np

def compute_recall(predicted, ground_truth, k):
    """Compute recall@k."""
    total = 0.0
    for i in range(len(predicted)):
        pred = set(predicted[i, :k].tolist())
        true = set(ground_truth[i, :k].tolist())
        total += len(pred & true) / k
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

> **Note**: The existing HNSW and DiskANN implementations are standalone and do not use `base_runner.py`. They serve as complete, working examples of this approach.

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
uv run python library/datasets/download.py --dataset sift-10k

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
