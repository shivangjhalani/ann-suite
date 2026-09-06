"""Prepare local BIGANN SIFT subsets for ANN Suite.

The SPANN runner consumes canonical NumPy files. BIGANN/SPTAG stores vectors as
two-int32-header ``u8bin`` files and ground truth as ``ivecs`` files, so this
script converts those files without changing vector ids or query ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def load_u8bin(path: Path) -> np.ndarray:
    """Load an SPTAG DEFAULT UInt8 vector file."""
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=2)
    count, dimension = (int(value) for value in header)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(count, dimension))


def load_ivecs(path: Path) -> np.ndarray:
    """Load a BIGANN ivecs file with its per-row dimension headers."""
    raw = np.fromfile(path, dtype="<i4")
    if raw.size < 1:
        raise ValueError(f"Empty ground-truth file: {path}")
    width = int(raw[0])
    row_size = width + 1
    if width <= 0 or raw.size % row_size != 0:
        raise ValueError(f"Malformed ivecs file: {path}")
    rows = raw.reshape(-1, row_size)
    if not np.all(rows[:, 0] == width):
        raise ValueError(f"Inconsistent ivecs row widths: {path}")
    return rows[:, 1:].astype(np.int32, copy=False)


def save_float32_npy(source: np.ndarray, path: Path, chunk_size: int = 250_000) -> None:
    """Convert a vector source to float32 NPY incrementally."""
    destination = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=source.shape
    )
    for start in range(0, len(source), chunk_size):
        end = min(start + chunk_size, len(source))
        destination[start:end] = source[start:end]
    destination.flush()
    del destination


def prepare_dataset(
    source_dir: Path, output_dir: Path, name: str, base_count: int, gt_name: str
) -> None:
    """Convert one SIFT subset into the ANN Suite layout."""
    base_source = source_dir / "spann_bench" / f"bigann_{name}_base.u8bin"
    query_source = source_dir / "spann_bench" / "bigann_query_10k.u8bin"
    gt_source = source_dir / "gnd" / gt_name
    destination = output_dir / f"sift{name}"
    destination.mkdir(parents=True, exist_ok=True)

    base = load_u8bin(base_source)
    queries = load_u8bin(query_source)
    ground_truth = load_ivecs(gt_source)
    if len(base) != base_count:
        raise ValueError(f"Expected {base_count} base vectors, found {len(base)}")
    if len(queries) != len(ground_truth):
        raise ValueError("Query and ground-truth row counts differ")

    save_float32_npy(base, destination / "base.npy")
    save_float32_npy(queries, destination / "queries.npy")
    np.save(destination / "ground_truth.npy", ground_truth)
    metadata = {
        "name": f"sift{name}",
        "source": "BIGANN SIFT1B prefix",
        "description": f"BIGANN SIFT prefix with {base_count} base vectors",
        "distance_metric": "L2",
        "dimension": 128,
        "base_count": int(len(base)),
        "query_count": int(len(queries)),
        "ground_truth_count": int(ground_truth.shape[1]),
        "point_type": "float32",
    }
    (destination / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/home/gem/bigann"))
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.output_dir, "1m", 1_000_000, "idx_1M.ivecs")
    prepare_dataset(args.source_dir, args.output_dir, "10m", 10_000_000, "idx_10M.ivecs")


if __name__ == "__main__":
    main()
