"""Download and prepare DEEP1B subsets in the raw big-ann-benchmarks binary
format (.fbin, float32) that ann_suite.datasets.loader.DatasetLoader reads
natively (point_type: float32).

Reproduces the recipe used on the isfcr research host (~/data/dl_deep.sh):

1. Range-download the first `--base-count` vectors of the 1B-vector base
   file (`base.1B.fbin`, dim=96, float32) plus the 10K query file, from
   https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/.
   base.1B.fbin is `<u4 1000000000><u4 96><vectors...>` row-major float32
   (4 bytes/component), so the first N rows are the first `8 + N*96*4` bytes.
2. Patch the 8-byte header in place to the actual row count N.
3. Compute ground truth via PipeANN's `compute_groundtruth` utility (element
   type "float"), matching the reference recipe; falls back to a warning if
   no PipeANN build is available (see prepare_bigann.py for the same
   fallback discussion).

Does NOT convert to .npy - stays in native .fbin, read directly by
DatasetLoader/PipeANN with point_type: float32.

Usage:
    uv run python tools/research/prepare_deep.py \\
        --output-dir data/deep-100m --base-count 100000000 \\
        --pipeann-bin ~/research/PipeANN/build-ro/tests
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

DIM = 96
BYTES_PER_COMPONENT = 4
BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
QUERY_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin"
HEADER_BYTES = 8


def _patch_header(path: Path, count: int, dim: int = DIM) -> None:
    with path.open("r+b") as f:
        f.write(struct.pack("<II", count, dim))


def _size_tag(count: int) -> str:
    if count % 1_000_000 == 0:
        return f"{count // 1_000_000}m"
    return str(count)


def download_query(output_dir: Path) -> Path:
    dest = output_dir / "query.fbin"
    if dest.exists():
        print(f"skip (exists): {dest}", file=sys.stderr)
        return dest
    subprocess.run(["curl", "-s", "-o", str(dest), QUERY_URL], check=True)
    return dest


def range_download_base(output_dir: Path, base_count: int) -> Path:
    dest = output_dir / f"base{_size_tag(base_count)}.fbin"
    if dest.exists():
        print(f"skip (exists): {dest}", file=sys.stderr)
        return dest
    end_byte = HEADER_BYTES + base_count * DIM * BYTES_PER_COMPONENT - 1
    subprocess.run(
        ["curl", "-s", "-r", f"0-{end_byte}", "-o", str(dest), BASE_URL],
        check=True,
    )
    _patch_header(dest, base_count)
    return dest


def compute_ground_truth_native(
    pipeann_bin: Path, base: Path, query: Path, k: int, output: Path
) -> None:
    compute_gt = pipeann_bin / "utils" / "compute_groundtruth"
    if not compute_gt.exists():
        raise FileNotFoundError(f"compute_groundtruth not found at {compute_gt}")
    subprocess.run(
        [str(compute_gt), "float", "l2", str(base), str(query), str(k), str(output), "null", "null"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--pipeann-bin", type=Path, default=None, help="PipeANN build/tests dir")
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_path = download_query(args.output_dir) if not args.skip_query else args.output_dir / "query.fbin"
    base_path = range_download_base(args.output_dir, args.base_count)

    print(f"base: {base_path}", file=sys.stderr)
    print(f"query: {query_path}", file=sys.stderr)

    if not args.skip_ground_truth:
        if args.pipeann_bin is None:
            print(
                "WARNING: --pipeann-bin not given, skipping ground-truth computation.",
                file=sys.stderr,
            )
        else:
            gt_path = args.output_dir / f"gt{_size_tag(args.base_count)}.bin"
            compute_ground_truth_native(args.pipeann_bin, base_path, query_path, args.gt_k, gt_path)
            print(f"ground truth: {gt_path}", file=sys.stderr)

    print("DEEP_PREPARE_DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
