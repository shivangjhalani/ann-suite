#!/bin/bash
# Serial: build on the full stage (warm-up run), 10 min SSD-cache settle, measured searches.
set -x
cd /home/isfcr/shivang/ann-suite
PY=.venv/bin/python
$PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_fastscan_local.py --name warm-fs
sleep 600
$PY tools/evolve/evolve_bench.py measure tools/evolve/ablation/abl_fastscan_local.py --name fastscan-local
echo ALLDONE
