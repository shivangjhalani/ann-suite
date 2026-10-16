#!/bin/bash
# 100M scale: design v2 (tools/evolve/ablation/abl_v2.py, 40 B/vector) vs Starling-B4.0
# (40 B PQ/vector) on BIGANN-100M (make_heldout.py bigann100m-q2k). Serial: our build
# (stage scale100m), Starling build, SLC flush (20 GB) + settle, qd1 check, two
# interleaved repetitions at 1 and 8 threads, qd1 check. Search containers capped at 8g.
# Compare: compare_v2.py 100m 0.9 0.95 0.97 (files v2-100m-*, v2star-100m-*).
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
PY=.venv/bin/python
P=tools/evolve/ablation/abl_v2.py
G=tools/evolve/e2e/heldout_paths.py
SB=configs/evolve/baselines/starling_100m.yaml
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
PTS=$($PY $G grid 128:1024,192:1536,256:2048,384:3072,512:4096,768:6144,1024:8192 16,24,32,48)
SW='[{"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}]'
IDX=$($PY $G index bigann scale100m bigann100m-q2k $P)
ours() { # $1 mode flags, $2 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 8g --network none \
		-v "$PWD":/repo:ro -v "$IDX":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program /repo/$P --index /idx \
		--queries /data/bigann100m-q2k/queries.npy --gt /data/bigann100m-q2k/ground_truth.npy \
		--metric L2 --points "$PTS" $1 >"$OUT/$2.jsonl"
}
star() { # $1 threads, $2 out
	$PY tools/evolve/e2e/run_baselines_e2e.py $SB --threads "$1" --out "$2" --sweep "$SW"
}

# 1. builds (timings of fresh files are not used)
[ -d "$IDX/disk" ] || $PY tools/evolve/evolve_bench.py measure $P --stage scale100m --name v2-100m-model
$PY tools/evolve/e2e/run_baselines_e2e.py $SB --build --threads 1 --out s100m-build-discard --sweep '[{"Ls": 25}]'
df -h /

# 2. fold everything out of the SLC cache, settle, check
$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$IDX/disk/lists.pages $(find /home/isfcr/evolve_indices/baselines/Starling-B4.0-100m -type f -size +5G)"
python3 tools/evolve/e2e/ssd_qd1.py $FILES

# 3. interleaved repetitions
for rep in 1 2; do
	ours "--threads 1" "v2-100m-t1-r$rep"
	star 1 "v2star-100m-t1-r$rep"
	ours "--threads 8" "v2-100m-t8-r$rep"
	star 8 "v2star-100m-t8-r$rep"
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
