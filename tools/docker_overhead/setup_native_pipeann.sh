#!/usr/bin/env bash
# Build the same PipeANN binaries the image ships (pinned upstream commit +
# load_aware patch + vendored files, same READ_ONLY_TESTS/NO_MAPPING flags),
# natively, for the Docker-overhead comparison (in particular the io_uring path).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ALGO="$REPO/library/algorithms/pipeann"
OUT=${1:-$REPO/.native/pipeann}
COMMIT=$(grep -oP '^ARG PIPEANN_COMMIT=\K\S+' "$ALGO/Dockerfile")

rm -rf "$OUT/src"
mkdir -p "$OUT"
git clone -q https://github.com/thustorage/PipeANN.git "$OUT/src"
git -C "$OUT/src" checkout -q "$COMMIT"

cp "$ALGO/vendor/io_governor.h" "$OUT/src/include/utils/io_governor.h"
cp "$ALGO/vendor/search_openloop.cpp" "$OUT/src/tests/search_openloop.cpp"
patch -s -d "$OUT/src" -p1 < "$ALGO/patches/load_aware.patch"

( cd "$OUT/src/third_party/liburing" && ./configure && make -j"$(nproc)" )

mkdir -p "$OUT/src/build"
( cd "$OUT/src/build" && \
  ADDITIONAL_DEFINITIONS="-DREAD_ONLY_TESTS -DNO_MAPPING" cmake .. \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_MILVUS_SERVER=OFF && \
  make -j"$(nproc)" build_disk_index build_memory_index search_disk_index search_openloop \
    gen_random_slice compute_groundtruth vecs_to_bin change_pts )

echo "PIPEANN_BIN=$OUT/src/build/tests"
"$OUT/src/build/tests/search_disk_index" 2>&1 | head -1 || true
