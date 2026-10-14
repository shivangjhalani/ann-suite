#!/bin/bash
# Serial ablation: per layout one build (inside the warm-up run), a 10 min settle for the
# SLC cache, then the measured vec and page programs on the cached index.
set -x
cd /home/isfcr/shivang/ann-suite
PY=.venv/bin/python
for layout in cluster local; do
  $PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_${layout}_vec.py --name warm-${layout}
  sleep 600
  for rank in vec page; do
    $PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_${layout}_${rank}.py --name ${layout}-${rank}
  done
done
echo ALLDONE
