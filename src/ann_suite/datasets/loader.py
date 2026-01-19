"""Dataset loading utilities for ANN benchmarking.

Supports loading datasets from various formats:
- HDF5 (.h5, .hdf5) - Standard ann-benchmarks format
- NumPy (.npy, .npz)
- Binary (.bin, .fbin, .ibin) - big-ann-benchmarks format
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ann_suite.core.schemas import DatasetConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loader for ANN benchmark datasets.

    Handles multiple formats including HDF5 (standard ann-benchmarks),
    NumPy arrays, and binary formats (big-ann-benchmarks).
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize loader with base data directory."""
        self.data_dir = Path(data_dir).resolve()

    def load(
        self, config: DatasetConfig
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32] | None]:
        """Load a dataset based on its configuration.

        Args:
            config: Dataset configuration

        Returns:
            Tuple of (base_vectors, query_vectors, ground_truth_indices)
            Ground truth may be None if not provided.
        """
        base_path = self._resolve_path(config.base_path)
        query_path = self._resolve_path(config.query_path) if config.query_path else base_path
        gt_path = self._resolve_path(config.ground_truth_path) if config.ground_truth_path else None

        # Load base vectors
        logger.info(f"Loading base vectors from {base_path}")
        base_vectors = self._load_vectors(base_path, config.dimension, config.point_type)
        logger.info(f"Loaded {base_vectors.shape[0]} base vectors (dim={base_vectors.shape[1]})")

        # Load query vectors
        if query_path != base_path:
            logger.info(f"Loading query vectors from {query_path}")
            query_vectors = self._load_vectors(query_path, config.dimension, config.point_type)
        else:
            # Use a sample of base vectors as queries
            logger.info("Using sampled base vectors as queries")
            n_queries = min(10000, len(base_vectors) // 10)
            indices = np.random.choice(len(base_vectors), n_queries, replace=False)
            query_vectors = base_vectors[indices].copy()
        logger.info(f"Loaded {query_vectors.shape[0]} query vectors")

        # Load ground truth if available
        ground_truth = None
        if gt_path is not None and gt_path.exists():
            logger.info(f"Loading ground truth from {gt_path}")
            ground_truth = self._load_ground_truth(gt_path)
            logger.info(f"Loaded ground truth with shape {ground_truth.shape}")

        return base_vectors, query_vectors, ground_truth

    def _resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to data_dir if not absolute."""
        if path.is_absolute():
            return path
        return self.data_dir / path

    def _load_vectors(
        self, path: Path, dimension: int, point_type: str = "float32"
    ) -> NDArray[np.float32]:
        """Load vectors from various file formats."""
        suffix = path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            return self._load_hdf5(path)
        elif suffix == ".npy":
            return self._load_numpy(path)
        elif suffix == ".npz":
            return self._load_numpy_archive(path)
        elif suffix in (".bin", ".fbin", ".ibin", ".u8bin"):
            return self._load_binary(path, dimension, point_type)
        else:
            # Try to auto-detect format
            return self._load_numpy(path)

    def _load_hdf5(self, path: Path) -> NDArray[np.float32]:
        """Load vectors from HDF5 file (ann-benchmarks format)."""
        import h5py

        with h5py.File(path, "r") as f:
            # Standard ann-benchmarks format uses 'train' for base vectors
            if "train" in f:
                data = np.array(f["train"])
            elif "base" in f:
                data = np.array(f["base"])
            else:
                # Try first dataset
                key = list(f.keys())[0]
                data = np.array(f[key])

        return data.astype(np.float32)

    def _load_numpy(self, path: Path) -> NDArray[np.float32]:
        """Load vectors from NumPy file."""
        data = np.load(path, mmap_mode="r")
        return data  # type: ignore

    def _load_numpy_archive(self, path: Path) -> NDArray[np.float32]:
        """Load vectors from NumPy archive (.npz)."""
        archive = np.load(path, mmap_mode="r")
        # Try common key names
        for key in ["arr_0", "data", "vectors", "base", "train"]:
            if key in archive:
                return archive[key]  # type: ignore
        # Fall back to first array
        return archive[list(archive.keys())[0]]  # type: ignore

    def _load_binary(
        self, path: Path, dimension: int, point_type: str = "float32"
    ) -> NDArray[np.float32]:
        """Load vectors from binary file (big-ann-benchmarks format).

        Format: [n_vectors (4 bytes)] [dim (4 bytes)] [vectors...]
        """
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "uint8": np.uint8,
            "int8": np.int8,
        }
        dtype = dtype_map.get(point_type, np.float32)

        with open(path, "rb") as f:
            # Read header
            n_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]

            if dim != dimension:
                logger.warning(f"Dimension mismatch: config={dimension}, file={dim}")

            # Read vectors
            data = np.frombuffer(f.read(), dtype=dtype)
            data = data.reshape(n_vectors, dim)

        return data.astype(np.float32)

    def _load_ground_truth(self, path: Path) -> NDArray[np.int32]:
        """Load ground truth neighbors."""
        suffix = path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            import h5py

            with h5py.File(path, "r") as f:
                if "neighbors" in f:
                    return np.array(f["neighbors"]).astype(np.int32)
                elif "test" in f:
                    return np.array(f["test"]).astype(np.int32)
                else:
                    key = list(f.keys())[0]
                    return np.array(f[key]).astype(np.int32)
        elif suffix == ".npy":
            return np.load(path).astype(np.int32)
        elif suffix in (".bin", ".ibin"):
            # Binary format: [n_queries][k][indices...]
            with open(path, "rb") as f:
                n_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                k = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                data = np.frombuffer(f.read(), dtype=np.int32)
                return data.reshape(n_queries, k)
        else:
            return np.load(path).astype(np.int32)


def load_dataset(
    config: DatasetConfig, data_dir: Path | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32] | None]:
    """Convenience function to load a dataset.

    Args:
        config: Dataset configuration
        data_dir: Optional base data directory

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth)
    """
    loader = DatasetLoader(data_dir or Path("."))
    return loader.load(config)



