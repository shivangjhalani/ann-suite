"""Dataset download and preparation utilities.

Downloads datasets from ann-benchmarks.com and converts them to NumPy format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import h5py
import numpy as np
import yaml


def download_file(url: str, dest: Path, quiet: bool = False) -> None:
    """Download a file if it doesn't exist.

    Args:
        url: Source URL
        dest: Destination path
        quiet: Suppress progress output
    """
    if dest.exists():
        if not quiet:
            print(f"  Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"  Downloading: {url}")
        print(f"  To: {dest}")

    # Use custom User-Agent to avoid 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ann-suite/0.1.0; +https://github.com/ann-suite)"
    }
    req = Request(url, headers=headers)

    with urlopen(req) as response, open(dest, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        block_size = 8192
        downloaded = 0

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break

            out_file.write(buffer)
            downloaded += len(buffer)

            if not quiet and total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()

    if not quiet:
        print()  # Newline after progress


def load_registry(registry_path: Path | None = None) -> dict[str, Any]:
    """Load the dataset registry.

    Args:
        registry_path: Path to registry.yaml (default: same directory as this file)

    Returns:
        Registry dictionary

    Raises:
        FileNotFoundError: If registry file is not found
    """
    if registry_path is None:
        # Default to user-accessible registry in library/datasets
        # We assume the user is running from project root or the package pattern
        registry_path = Path("library/datasets/registry.yaml")
        if not registry_path.exists():
            # Fallback: try finding it relative to file if running in dev mode
            fallback = Path(__file__).parents[3] / "library/datasets/registry.yaml"
            registry_path = fallback if fallback.exists() else Path("registry.yaml")

    with open(registry_path) as f:
        result = yaml.safe_load(f)
        # Ensure we return a dict even if file is empty
        return result if isinstance(result, dict) else {}


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    metric: str = "L2",
) -> np.ndarray:
    """Compute ground truth neighbors using brute force.

    Args:
        base: Base vectors (N x D)
        queries: Query vectors (Q x D)
        k: Number of neighbors
        metric: Distance metric

    Returns:
        Ground truth indices (Q x k)
    """
    print(f"  Computing ground truth (k={k})...")
    n_queries = len(queries)
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)

    # Precompute base norms outside the loop for efficiency
    base_sq_norms: np.ndarray | None = None
    base_normalized: np.ndarray | None = None

    if metric in ("L2", "euclidean"):
        # Precompute ||base||^2 for L2: ||base - query||^2 = ||base||^2 - 2*base.query + ||query||^2
        base_sq_norms = np.sum(base * base, axis=1)
    elif metric in ("cosine", "angular"):
        # Precompute normalized base vectors
        base_norms = np.linalg.norm(base, axis=1, keepdims=True)
        base_normalized = base / (base_norms + 1e-10)

    for i, query in enumerate(queries):
        if metric in ("L2", "euclidean"):
            # Use dot products: ||base - query||^2 = ||base||^2 - 2*base.query + ||query||^2
            assert base_sq_norms is not None
            query_sq_norm = np.dot(query, query)
            distances = base_sq_norms - 2.0 * np.dot(base, query) + query_sq_norm
        elif metric in ("IP", "inner_product"):
            distances = -np.dot(base, query)
        elif metric in ("cosine", "angular"):
            # Use precomputed normalized base
            assert base_normalized is not None
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            distances = -np.dot(base_normalized, query_norm)
        else:
            # Fallback: use dot product formulation for L2
            query_sq_norm = np.dot(query, query)
            distances = np.sum(base * base, axis=1) - 2.0 * np.dot(base, query) + query_sq_norm

        # Use argpartition for top-k (O(n) instead of O(n log n))
        if k < len(distances):
            top_k_unsorted = np.argpartition(distances, k)[:k]
            # Sort within top-k for stable output
            top_k_sorted = top_k_unsorted[np.argsort(distances[top_k_unsorted])]
            ground_truth[i] = top_k_sorted
        else:
            ground_truth[i] = np.argsort(distances)[:k]

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{n_queries} queries")

    return ground_truth


def download_dataset(
    name: str,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    quiet: bool = False,
) -> Path:
    """Download and prepare a dataset.

    Args:
        name: Dataset name (or subset name) from registry
        output_dir: Output directory (default: library/datasets/)
        registry_path: Path to registry.yaml
        quiet: Suppress output

    Returns:
        Path to the prepared dataset directory

    Raises:
        ValueError: If dataset name is not found in registry
    """
    registry = load_registry(registry_path)
    datasets: dict[str, Any] = registry.get("datasets", {})

    # Check if name is a top-level dataset or a subset
    parent_config: dict[str, Any] | None = None
    subset_config: dict[str, Any] | None = None
    dataset_name = name

    if name in datasets:
        # It's a full dataset
        parent_config = datasets[name]
    else:
        # Check if it's a subset name
        for ds_name, ds_config in datasets.items():
            subsets = ds_config.get("subsets")
            if subsets and name in subsets:
                parent_config = ds_config
                subset_config = subsets[name]
                dataset_name = ds_name
                break

    # If we still don't have a config, the dataset wasn't found
    if parent_config is None:
        available: list[str] = []
        for ds_name, ds_config in datasets.items():
            available.append(ds_name)
            subsets = ds_config.get("subsets")
            if subsets:
                available.extend(subsets.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    if output_dir is None:
        output_dir = Path(__file__).parent

    # Determine output name
    output_name = name
    dataset_dir = output_dir / output_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Preparing dataset: {output_name}")
        description = parent_config.get("description", "N/A")
        if subset_config:
            print(f"  Subset of: {dataset_name}")
        print(f"  Description: {description}")

    # Download HDF5 file (always download parent)
    hdf5_path = output_dir / f"{dataset_name}.hdf5"
    url = parent_config.get("url")
    if not url:
        raise ValueError(f"Dataset {dataset_name} has no URL configured")
    download_file(url, hdf5_path, quiet=quiet)

    # Convert to NumPy format
    with h5py.File(hdf5_path, "r") as f:
        # Load base vectors
        if "train" in f:
            base = np.array(f["train"])
        else:
            raise ValueError("HDF5 file must have 'train' dataset")

        # Load query vectors
        queries = np.array(f["test"]) if "test" in f else base[:1000]

        # Load ground truth if available (only for full dataset)
        ground_truth = np.array(f["neighbors"]) if "neighbors" in f and not subset_config else None

    # Apply subset if requested
    if subset_config:
        base_count = subset_config.get("base_count", len(base))
        query_count = subset_config.get("query_count", len(queries))

        if not quiet:
            print(f"  Creating subset: {base_count} base, {query_count} queries")

        # Random subset (deterministic)
        rng = np.random.default_rng(seed=42)

        # Ensure we don't request more than available
        actual_base_count = min(base_count, len(base))
        actual_query_count = min(query_count, len(queries))

        base_indices = rng.choice(len(base), size=actual_base_count, replace=False)
        query_indices = rng.choice(len(queries), size=actual_query_count, replace=False)

        base = base[base_indices]
        queries = queries[query_indices]

        # Always recompute ground truth for subsets as indices change
        ground_truth = None

    # Compute ground truth if not available
    if ground_truth is None:
        ground_truth = compute_ground_truth(
            base, queries, k=100, metric=parent_config.get("distance_metric", "L2")
        )

    # Save NumPy files
    base_path = dataset_dir / "base.npy"
    queries_path = dataset_dir / "queries.npy"
    gt_path = dataset_dir / "ground_truth.npy"

    np.save(base_path, base.astype(np.float32))
    np.save(queries_path, queries.astype(np.float32))
    np.save(gt_path, ground_truth.astype(np.int32))

    # Save metadata
    metadata = {
        "name": output_name,
        "source": dataset_name,
        "description": parent_config.get("description", ""),
        "distance_metric": parent_config.get("distance_metric", "L2"),
        "dimension": base.shape[1],
        "base_count": len(base),
        "query_count": len(queries),
        "point_type": "float32",
    }

    if subset_config:
        metadata["is_subset"] = True
        metadata["subset_config"] = subset_config

    metadata_path = dataset_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    if not quiet:
        print(f"  Saved to: {dataset_dir}")
        print(f"    base.npy: {base.shape}")
        print(f"    queries.npy: {queries.shape}")
        print(f"    ground_truth.npy: {ground_truth.shape}")

    return dataset_dir


def list_datasets(registry_path: Path | None = None) -> None:
    """Print available datasets."""
    registry = load_registry(registry_path)
    datasets = registry.get("datasets", {})

    print("Available datasets:")
    print("-" * 60)
    for name, config in datasets.items():
        print(f"  {name}")
        print(f"    {config.get('description', 'No description')}")
        print(f"    Dimension: {config.get('dimension')}, Metric: {config.get('distance_metric')}")

        if "subsets" in config:
            print("    Subsets:")
            for sub_name, sub_config in config["subsets"].items():
                print(f"      - {sub_name} ({sub_config.get('base_count')} vectors)")
        print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for ANN benchmarking",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to download (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: library/datasets/)",
    )
    # --full argument is deprecated/removed as it is now default behavior for parent names
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.print_help()
        print("\nUse --list to see available datasets")
        return

    output_dir = Path(args.output) if args.output else None
    download_dataset(
        name=args.dataset,
        output_dir=output_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
