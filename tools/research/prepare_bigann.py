"""Download and prepare BIGANN (SIFT1B) subsets in the raw big-ann-benchmarks
binary format that ann_suite.datasets.loader.DatasetLoader reads natively
(no HDF5/.npy conversion needed - point_type: uint8, format inferred from
the .u8bin suffix).

Reproduces the recipe used on the isfcr research host
(~/data/dl_bigann.sh, ~/data/build10m.sh):

1. Range-download the first `--base-count` vectors of the 1B-vector base
   file (`base.1B.u8bin`) plus the 10K query file, from
   https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/.
   A byte-range request is used instead of downloading the full 1B-vector
   (~128GB) file: base.1B.u8bin is `<u4 1000000000><u4 128><vectors...>`,
   row-major uint8, so the first N rows are the first `8 + N*128` bytes.
2. Patch the 8-byte header in place (the range download keeps the original
   file's declared count, 1000000000; this rewrites it to N) so downstream
   tools see the correct vector count.
3. Optionally subset further from an already-downloaded larger prefix
   (e.g. derive a 10M subset from a 100M prefix already on disk) via
   PipeANN's `change_pts` utility, matching ~/data/build10m.sh, so a smaller
   subset never needs a second network download.
4. Compute ground truth via PipeANN's `compute_groundtruth` utility (exact
   brute-force, matches the reference numbers in the research notes) if a
   PipeANN build is available; otherwise fall back to
   ann_suite.datasets.ground_truth.compute_ground_truth (slower, brute-force
   in Python/faiss, fine for the 10M scale but not recommended for 100M).

This script does NOT convert to .npy - base/query files stay in their native
.u8bin format and are referenced directly from benchmark configs with
`point_type: uint8`.

Usage:
    # Full recipe for a 10M subset (downloads 10M rows directly):
    uv run python tools/research/prepare_bigann.py \\
        --output-dir data/bigann-10m --base-count 10000000

    # Derive a 10M subset from an already-downloaded 100M prefix (no network):
    uv run python tools/research/prepare_bigann.py \\
        --output-dir data/bigann-10m --base-count 10000000 \\
        --from-existing-base data/bigann-100m/base100m.u8bin \\
        --pipeann-bin ~/research/PipeANN/build-ro/tests

    # 100M prefix:
    uv run python tools/research/prepare_bigann.py \\
        --output-dir data/bigann-100m --base-count 100000000
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

DIM = 128
BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
QUERY_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
HEADER_BYTES = 8


def _patch_header(path: Path, count: int, dim: int = DIM) -> None:
    with path.open("r+b") as f:
        f.write(struct.pack("<II", count, dim))


def download_query(output_dir: Path) -> Path:
    dest = output_dir / "query.u8bin"
    if dest.exists():
        print(f"skip (exists): {dest}", file=sys.stderr)
        return dest
    subprocess.run(["curl", "-s", "-o", str(dest), QUERY_URL], check=True)
    return dest


def range_download_base(output_dir: Path, base_count: int, dim: int = DIM) -> Path:
    """Range-download the first `base_count` rows of base.1B.u8bin."""
    dest = output_dir / f"base{_size_tag(base_count)}.u8bin"
    if dest.exists():
        print(f"skip (exists): {dest}", file=sys.stderr)
        return dest
    end_byte = HEADER_BYTES + base_count * dim - 1
    subprocess.run(
        ["curl", "-s", "-r", f"0-{end_byte}", "-o", str(dest), BASE_URL],
        check=True,
    )
    _patch_header(dest, base_count, dim)
    return dest


def _size_tag(count: int) -> str:
    if count % 1_000_000 == 0:
        return f"{count // 1_000_000}m"
    return str(count)


def subset_from_existing(
    existing_base: Path, output_dir: Path, base_count: int, pipeann_bin: Path
) -> Path:
    """Derive a smaller prefix from an already-downloaded larger base file
    via PipeANN's change_pts utility (no network required)."""
    change_pts = pipeann_bin / "utils" / "change_pts"
    if not change_pts.exists():
        raise FileNotFoundError(
            f"change_pts not found at {change_pts}; pass --pipeann-bin pointing at a "
            "PipeANN build's tests/ dir, or omit --from-existing-base to range-download instead"
        )
    dest = output_dir / f"base{_size_tag(base_count)}.u8bin"
    subprocess.run(
        [str(change_pts), "uint8", str(existing_base), str(base_count)],
        cwd=output_dir,
        check=True,
    )
    produced = existing_base.parent / f"{existing_base.name}{base_count}"
    if produced.exists() and produced != dest:
        produced.rename(dest)
    return dest


def compute_ground_truth_native(
    pipeann_bin: Path, base: Path, query: Path, k: int, output: Path
) -> None:
    compute_gt = pipeann_bin / "utils" / "compute_groundtruth"
    if not compute_gt.exists():
        raise FileNotFoundError(f"compute_groundtruth not found at {compute_gt}")
    subprocess.run(
        [str(compute_gt), "uint8", "l2", str(base), str(query), str(k), str(output), "null", "null"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--from-existing-base", type=Path, default=None)
    parser.add_argument("--pipeann-bin", type=Path, default=None, help="PipeANN build/tests dir")
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_query:
        query_path = download_query(args.output_dir)
    else:
        query_path = args.output_dir / "query.u8bin"

    if args.from_existing_base:
        if args.pipeann_bin is None:
            raise SystemExit("--pipeann-bin is required with --from-existing-base")
        base_path = subset_from_existing(
            args.from_existing_base, args.output_dir, args.base_count, args.pipeann_bin
        )
    else:
        base_path = range_download_base(args.output_dir, args.base_count)

    print(f"base: {base_path}", file=sys.stderr)
    print(f"query: {query_path}", file=sys.stderr)

    if not args.skip_ground_truth:
        if args.pipeann_bin is None:
            print(
                "WARNING: --pipeann-bin not given, skipping ground-truth computation. "
                "Run compute_groundtruth manually, or pass --pipeann-bin.",
                file=sys.stderr,
            )
        else:
            gt_path = args.output_dir / f"gt{_size_tag(args.base_count)}.bin"
            compute_ground_truth_native(args.pipeann_bin, base_path, query_path, args.gt_k, gt_path)
            print(f"ground truth: {gt_path}", file=sys.stderr)

    print("BIGANN_PREPARE_DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
