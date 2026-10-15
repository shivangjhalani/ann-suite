#!/bin/bash
# End-to-end head-to-head on BIGANN-10M (results in results/evolve/e2e/):
#  1. SSD state of every index file (qd1 p50 must be ~90 us, i.e. folded to QLC).
#  2. Two interleaved repetitions of: ours (40 B fast-scan + refinement) at 1 thread,
#     8 forked workers and 8 threads, public and hidden queries; Starling-B0.40 and
#     B0.32 at 1 and 8 threads. Rep 1 adds PageANN-mem0.15 and SPANN-ratio0.15 for context.
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e; mkdir -p $OUT
H=$1   # built index dir of tools/evolve/ablation/abl_fastscan_h40.py
PTS='[{"nprobe": 64, "shortlist": 768, "pages": 16}, {"nprobe": 96, "shortlist": 1024, "pages": 16}, {"nprobe": 96, "shortlist": 1024, "pages": 20}, {"nprobe": 128, "shortlist": 1024, "pages": 20}, {"nprobe": 160, "shortlist": 768, "pages": 24}, {"nprobe": 128, "shortlist": 1536, "pages": 24}, {"nprobe": 192, "shortlist": 1536, "pages": 24}, {"nprobe": 256, "shortlist": 2048, "pages": 28}]'
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
ours() {  # $1 = mode flags, $2 = query set, $3 = out name
  docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
    -v $PWD:/repo:ro -v $H:/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
    /repo/tools/evolve/e2e/e2e_bench.py --program /repo/tools/evolve/ablation/abl_fastscan_h40.py \
    --index /idx --queries /data/$2/queries.npy --gt /data/$2/ground_truth.npy --points "$PTS" $1 \
    > $OUT/$3.jsonl
}
base() {  # $1 = config, $2 = threads, $3 = out, $4.. = --only names
  cfgf=$1; th=$2; out=$3; shift 3
  .venv/bin/python tools/evolve/e2e/run_baselines_e2e.py $cfgf --threads $th --out $out --only "$@" \
    --sweep '[{"Ls": 15}, {"Ls": 25}, {"Ls": 40}, {"Ls": 60}]'
}
FILES="$H/disk/lists.pages $(find /home/isfcr/evolve_indices/baselines/Starling-B0.40 /home/isfcr/evolve_indices/baselines/Starling-B0.32 -type f -size +2G)"
python3 tools/evolve/e2e/ssd_qd1.py $FILES
for rep in 1 2; do
  ours "--threads 1" bigann10m-q2k ours-h40-t1-r$rep
  base configs/evolve/baselines/starling_b040.yaml 1 starling040-t1-r$rep Starling-B0.40
  base configs/evolve/baselines/starling.yaml 1 starling032-t1-r$rep Starling-B0.32
  ours "--procs 8" bigann10m-q2k ours-h40-p8-r$rep
  ours "--threads 8" bigann10m-q2k ours-h40-t8-r$rep
  base configs/evolve/baselines/starling_b040.yaml 8 starling040-t8-r$rep Starling-B0.40
  base configs/evolve/baselines/starling.yaml 8 starling032-t8-r$rep Starling-B0.32
done
ours "--threads 1" bigann10m-hidden ours-h40-t1-hidden
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
