#!/usr/bin/env bash
# Build the same SPTAG/SPANN binaries the image ships (pinned commit, same
# cmake flags/targets), natively, for the Docker-overhead comparison.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ALGO="$REPO/library/algorithms/spann"
OUT=${1:-$REPO/.native/spann}
COMMIT=$(grep -oP '^ARG SPTAG_COMMIT=\K\S+' "$ALGO/Dockerfile")

rm -rf "$OUT/src"
mkdir -p "$OUT"
git clone -q --recurse-submodules https://github.com/microsoft/SPTAG.git "$OUT/src"
git -C "$OUT/src" checkout -q "$COMMIT"
git -C "$OUT/src" submodule update --init --recursive

cmake -S "$OUT/src" -B "$OUT/src/build" -DCMAKE_BUILD_TYPE=Release -DSPDK=OFF -DROCKSDB=OFF -DGPU=OFF
cmake --build "$OUT/src/build" --parallel "$(nproc)" --target indexbuilder indexsearcher

echo "SPTAG_BIN=$OUT/src/Release"
ls "$OUT/src/Release" | grep -iE "indexbuilder|indexsearcher"
