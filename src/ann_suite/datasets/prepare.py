"""Dataset preparation: HDF5 -> NumPy conversion, subsetting, metadata.

Turns a raw ann-benchmarks HDF5 file (``train``/``test``/``neighbors`` keys)
into the canonical per-dataset layout consumed by the benchmark pipeline:

    <data_dir>/<name>/
        base.npy          - base vectors (N x D, float32)
        queries.npy       - query vectors (Q x D, float32)
        ground_truth.npy  - neighbor indices (Q x k, int32)
        metadata.yaml     - dataset metadata

Reuses :class:`ann_suite.datasets.loader.DatasetLoader` for parsing so there is
exactly one canonical implementation of each file format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from ann_suite.datasets.download import download_file
from ann_suite.datasets.ground_truth import compute_ground_truth
from ann_suite.datasets.registry import (
    DEFAULT_DATA_DIR,
    find_dataset,
)

logger = logging.getLogger(__name__)

#: Number of neighbors computed when a dataset ships without ground truth.
DEFAULT_K = 100


def prepare_dataset(
    name: str,
    *,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    quiet: bool = False,
) -> Path:
    """Download and prepare a dataset.

    Args:
        name: Dataset name (or subset name) from the registry.
        output_dir: Output directory (default: :data:`DEFAULT_DATA_DIR`).
        registry_path: Optional explicit registry path.
        quiet: Suppress progress bars and informational logging.

    Returns:
        Path to the prepared dataset directory.

    Raises:
        ValueError: If ``name`` is not in the registry or has no URL.
    """
    parent_name, parent_config, subset_config = find_dataset(name, registry_path)

    output_dir = (output_dir or DEFAULT_DATA_DIR).resolve()
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing dataset: %s (source: %s)", name, parent_name)
    if subset_config:
        logger.info("  Subset of: %s", parent_name)
    logger.info("  Description: %s", parent_config.get("description", "N/A"))

    # Download the parent HDF5 alongside the prepared output.
    url = parent_config.get("url")
    if not url:
        raise ValueError(f"Dataset {parent_name} has no URL configured")
    hdf5_path = output_dir / f"{parent_name}.hdf5"
    download_file(url, hdf5_path, show_progress=not quiet)

    # Parse the HDF5 into arrays (only the keys this pipeline produces).
    with h5py.File(hdf5_path, "r") as f:
        if "train" not in f:
            raise ValueError("HDF5 file must have a 'train' dataset")
        base = np.array(f["train"])
        queries = np.array(f["test"]) if "test" in f else base[:1000]
        has_neighbors = "neighbors" in f
        # Only trust shipped ground truth for the FULL dataset (indices shift
        # once a subset is sampled).
        ground_truth = np.array(f["neighbors"]) if has_neighbors and not subset_config else None

    # Apply subset if requested (deterministic random sampling).
    if subset_config:
        base_count = int(subset_config.get("base_count", len(base)))
        query_count = int(subset_config.get("query_count", len(queries)))
        logger.info("  Creating subset: %d base, %d queries", base_count, query_count)

        rng = np.random.default_rng(seed=42)
        base = base[rng.choice(len(base), size=min(base_count, len(base)), replace=False)]
        queries = queries[
            rng.choice(len(queries), size=min(query_count, len(queries)), replace=False)
        ]
        # Sampled indices invalidate shipped ground truth.
        ground_truth = None

    if ground_truth is None:
        ground_truth = compute_ground_truth(
            base,
            queries,
            k=DEFAULT_K,
            metric=parent_config.get("distance_metric", "L2"),
        )

    _save_prepared(
        dataset_dir, name, parent_name, parent_config, subset_config, base, queries, ground_truth
    )

    if not quiet:
        logger.info("    base.npy: %s", base.shape)
        logger.info("    queries.npy: %s", queries.shape)
        logger.info("    ground_truth.npy: %s", ground_truth.shape)

    return dataset_dir


def _save_prepared(
    dataset_dir: Path,
    name: str,
    parent_name: str,
    parent_config: dict[str, Any],
    subset_config: dict[str, Any] | None,
    base: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    """Write the canonical per-dataset files."""
    np.save(dataset_dir / "base.npy", base.astype(np.float32))
    np.save(dataset_dir / "queries.npy", queries.astype(np.float32))
    np.save(dataset_dir / "ground_truth.npy", ground_truth.astype(np.int32))

    metadata: dict[str, Any] = {
        "name": name,
        "source": parent_name,
        "description": parent_config.get("description", ""),
        "distance_metric": parent_config.get("distance_metric", "L2"),
        "dimension": base.shape[1],
        "base_count": int(len(base)),
        "query_count": int(len(queries)),
        "point_type": "float32",
    }
    if subset_config:
        metadata["is_subset"] = True
        metadata["subset_config"] = subset_config

    metadata_path = dataset_dir / "metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info("  Saved to: %s", dataset_dir)
