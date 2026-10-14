#!/bin/bash
# Serial held-out run of the ablation programs on DEEP-10M and T2I-10M: per dataset and
# layout one build (the warm-up run), a 10 min SSD-cache settle, then the measured searches.
set -x
cd /home/isfcr/shivang/ann-suite
PY=.venv/bin/python
for ds in deep t2i; do
  for layout in cluster local; do
    if [ $layout = cluster ]; then ranks="vec page"; else ranks="vec page heap"; fi
    first=$(echo $ranks | cut -d" " -f1)
    $PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_${layout}_${first}.py --stage heldout_$ds --name warm-$ds-$layout
    sleep 600
    for rank in $ranks; do
      $PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_${layout}_${rank}.py --stage heldout_$ds --name $ds-$layout-$rank
    done
  done
done
echo ALLDONE
