#!/bin/bash
# Search-only runs of the page-voting variants on the cached fast-scan index.
set -x
cd /home/isfcr/shivang/ann-suite
for v in best_small vote_soft1 vote_soft03 vote_count; do
  .venv/bin/python tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_fastscan_$v.py --name fs-$v
done
echo ALLDONE
