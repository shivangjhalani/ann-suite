# Adding Datasets Guide

This guide explains how to add new datasets to the ANN Benchmarking Suite.

## Overview

Datasets are managed through the `library/datasets/` directory with:
- **Registry** (`registry.yaml`): Declarative manifest of available datasets
- **Download Utility** (`download.py`): Script to fetch and prepare datasets
- **Local Storage**: Downloaded datasets stored in `library/datasets/<name>/`

## Quick Start

```bash
# List available datasets
uv run ann-suite download --list

# Download a dataset subset
uv run ann-suite download --dataset sift-10k

# Download full dataset (by using the parent name)
uv run ann-suite download --dataset sift-128-euclidean

# Download to custom location
uv run ann-suite download --dataset sift-10k --output ./data
```

---

## Dataset File Structure

Each prepared dataset has this structure:

```
library/datasets/<dataset-name>/
├── base.npy          # Base vectors (N x D, float32)
├── queries.npy       # Query vectors (Q x D, float32)
├── ground_truth.npy  # Ground truth neighbors (Q x K, int32)
└── metadata.yaml     # Dataset metadata
```

### File Formats

| File | Format | Shape | Type | Description |
|------|--------|-------|------|-------------|
| `base.npy` | NumPy | (N, D) | float32 | Database vectors |
| `queries.npy` | NumPy | (Q, D) | float32 | Query vectors |
| `ground_truth.npy` | NumPy | (Q, K) | int32 | True neighbor indices |
| `metadata.yaml` | YAML | - | - | Dataset information |

---

## Adding a Dataset from ann-benchmarks.com

### Step 1: Add to Registry

Edit `library/datasets/registry.yaml`:

```yaml
datasets:
  # ... existing datasets ...

  my-new-dataset:
    description: "Description of my dataset"
    url: "http://ann-benchmarks.com/my-dataset.hdf5"
    format: hdf5
    distance_metric: L2  # or: IP, cosine, hamming
    dimension: 128
    base_count: 1000000
    query_count: 10000
    point_type: float32
```

### Step 2: Download and Prepare

```bash
uv run ann-suite download --dataset my-new-dataset
```

The download script will:
1. Download the HDF5 file
2. Extract `train` (base), `test` (queries), `neighbors` (ground truth)
3. Convert to NumPy format
4. Generate `metadata.yaml`

### Step 3: Add Subsets (Optional)

For large datasets, define a subset for quick testing:

```yaml
datasets:
  sift-128-euclidean:
    description: "SIFT image descriptors"
    url: "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    distance_metric: L2
    dimension: 128
    base_count: 1000000
    query_count: 10000
    # Subsets for quick tests
    subsets:
      sift-10k:
        base_count: 10000
        query_count: 1000
```

Now you can download either:

```bash
# Full dataset (default if parent name used)
uv run ann-suite download --dataset sift-128-euclidean

# Subset (must use subset name)
uv run ann-suite download --dataset sift-10k
```

---

## Adding a Custom Dataset

### Option 1: Direct NumPy Files

If you already have NumPy files:

```python
import numpy as np

# Save your data
np.save("library/datasets/my-dataset/base.npy", base_vectors)
np.save("library/datasets/my-dataset/queries.npy", queries)
np.save("library/datasets/my-dataset/ground_truth.npy", ground_truth)
```

Create `metadata.yaml`:

```yaml
name: my-dataset
description: "My custom dataset"
distance_metric: L2
dimension: 128
base_count: 10000
query_count: 1000
point_type: float32
```

### Option 2: Custom Download Script

For datasets not from ann-benchmarks.com, create a custom download script:

```python
# library/datasets/download_custom.py
import numpy as np
from pathlib import Path

def download_my_dataset(output_dir: Path):
    """Download and prepare my custom dataset."""

    # 1. Download from your source
    # ...

    # 2. Convert to numpy arrays
    base = ...  # shape: (N, D), dtype: float32
    queries = ...  # shape: (Q, D), dtype: float32

    # 3. Compute ground truth if not available
    from library.datasets.download import compute_ground_truth
    ground_truth = compute_ground_truth(base, queries, k=100, metric="L2")

    # 4. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "base.npy", base.astype(np.float32))
    np.save(output_dir / "queries.npy", queries.astype(np.float32))
    np.save(output_dir / "ground_truth.npy", ground_truth.astype(np.int32))
```

---

## Computing Ground Truth

If your dataset doesn't include ground truth, the suite can compute it:

```python
from library.datasets.download import compute_ground_truth

# Compute k=100 nearest neighbors for each query
ground_truth = compute_ground_truth(
    base=base_vectors,      # (N, D) array
    queries=query_vectors,  # (Q, D) array
    k=100,
    metric="L2"  # or "IP", "cosine"
)
```

**Warning**: This uses brute-force search and can be slow for large datasets.

---

## Using Datasets in Benchmarks

Reference your dataset in the benchmark configuration:

```yaml
data_dir: "./library/datasets"

datasets:
  - name: my-dataset
    base_path: my-dataset/base.npy
    query_path: my-dataset/queries.npy
    ground_truth_path: my-dataset/ground_truth.npy
    distance_metric: L2
    dimension: 128
    point_type: float32
    base_count: 10000
    query_count: 1000
```

---

## Available Datasets

The registry includes these pre-configured datasets:

| Name | Dimension | Metric | Vectors | Description |
|------|-----------|--------|---------|-------------|
| `sift-10k` | 128 | L2 | 10K | SIFT subset (quick testing) |
| `sift-128-euclidean` | 128 | L2 | 1M | Full SIFT dataset |
| `glove-25-10k` | 25 | cosine | 10K | GloVe subset (quick testing) |
| `glove-25-angular` | 25 | cosine | 1.2M | Full GloVe-25 |
| `glove-100-angular` | 100 | cosine | 1.2M | GloVe-100 |
| `fashion-mnist-784-euclidean` | 784 | L2 | 60K | Fashion-MNIST |
| `gist-960-euclidean` | 960 | L2 | 1M | GIST descriptors |

---

## Dataset Best Practices

### For Quick Development

Use small subsets (10K-100K vectors):
```bash
uv run ann-suite download --dataset sift-10k
```

### For Production Benchmarks

Use full datasets with proper ground truth:
```bash
uv run ann-suite download --dataset sift-128-euclidean
```

### For Fair Comparison

1. **Same ground truth depth**: Use k=100 ground truth for k=10 search
2. **Consistent metrics**: Match dataset metric with algorithm metric
3. **Multiple datasets**: Test on different data distributions

### Memory Considerations

Large datasets require significant memory for:
- Loading base vectors into container
- Computing ground truth (if not provided)
- Building indices

| Dataset Size | Estimated Memory |
|--------------|------------------|
| 10K × 128D | ~5 MB |
| 1M × 128D | ~500 MB |
| 10M × 128D | ~5 GB |
| 1B × 128D | ~500 GB |

---

## Troubleshooting

### Dataset Download Fails

```bash
# Check network connectivity
curl -I http://ann-benchmarks.com/sift-128-euclidean.hdf5

# Try without quiet mode for error messages
uv run ann-suite download --dataset sift-128-euclidean 2>&1
```

### Ground Truth Mismatch

Ensure ground truth matches:
- Same distance metric as algorithm
- At least k neighbors (typically k=100)
- Same base vectors (no shuffling)

### Dimension Mismatch

Validate your dataset:
```python
import numpy as np

base = np.load("library/datasets/my-dataset/base.npy")
queries = np.load("library/datasets/my-dataset/queries.npy")

print(f"Base: {base.shape}, dtype: {base.dtype}")
print(f"Queries: {queries.shape}, dtype: {queries.dtype}")

assert base.shape[1] == queries.shape[1], "Dimension mismatch!"
```
