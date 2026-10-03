#!/usr/bin/env bash
# Run frontier baseline configs in two lanes (at most two 10M builds at once).
# Scored baseline metrics are immune to cross-container interference (per-run
# O_DIRECT read counts, per-cgroup io.stat and memory), so lanes may overlap.
# Usage: tools/evolve/run_baselines.sh [a|b|ab]   (default ab; logs in results/evolve/baselines/)
#   lane a: SPANN, PipeANN, DiskANN    lane b: Starling, PageANN + LAANN
set -u
LANES=${1:-ab}
cd "$(dirname "$0")/../.."
LOG=results/evolve/baselines
mkdir -p "$LOG"
wait_image() { until docker image inspect "$1" >/dev/null 2>&1 && [ -z "$(pgrep -f "docker build.*$2/Dockerfile")" ]; do sleep 60; done; }
run() { .venv/bin/python tools/evolve/evolve_bench.py baselines "configs/evolve/baselines/$1.yaml" > "$LOG/$1.out" 2> "$LOG/$1.err"; echo "$1 rc=$?" >> "$LOG/lanes.log"; }
watchdog() { while sleep 60; do echo "$(date +%T) $(awk '/MemAvailable/{print int($2/1048576)"G avail"}' /proc/meminfo) $(df -BG --output=avail / | tail -1)disk" >> "$LOG/watchdog.log"; done; }
watchdog & WD=$!
pids=()
if [[ $LANES == *a* ]]; then
  ( run spann; run pipeann; wait_image ann-suite/diskann:latest diskann; run diskann ) & pids+=($!)
fi
if [[ $LANES == *b* ]]; then
  ( run starling; wait_image ann-suite/pageann:latest pageann; run pageann ) & pids+=($!)
fi
wait "${pids[@]}"
kill $WD
echo "LANES_DONE $LANES" >> "$LOG/lanes.log"
