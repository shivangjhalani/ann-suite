#!/usr/bin/env bash
# Build the same diskannpy the DiskANN image ships (same upstream commit, same stats
# patch, same numpy/pybind11 pins) into a host venv, so the ann-suite DiskANN runner
# can be run natively for the Docker-overhead comparison.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$REPO/.native/diskann}
COMMIT=$(grep -oP '^ARG DISKANN_COMMIT=\K\S+' "$REPO/library/algorithms/diskann/Dockerfile")

rm -rf "$OUT/src"
mkdir -p "$OUT"
git clone -q --branch cpp_main https://github.com/microsoft/DiskANN.git "$OUT/src"
git -C "$OUT/src" checkout -q "$COMMIT"
patch -s -d "$OUT/src" -p1 < "$REPO/library/algorithms/diskann/patches/stats.patch"

uv venv -q --allow-existing --python 3.11 "$OUT/venv"
export VIRTUAL_ENV="$OUT/venv"
uv pip install -q "numpy==1.25" cmake==3.27.9 ninja pybind11==2.11.1 setuptools wheel
CMAKE_ARGS="-DPYBIND=ON" uv pip install -q --no-cache "$OUT/src"
"$OUT/venv/bin/python" -c "import diskannpy; print('diskannpy', diskannpy.__file__)"
