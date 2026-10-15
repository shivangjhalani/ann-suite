#!/bin/bash
# End-to-end comparison of design v2 (tools/evolve/ablation/abl_v2.py: 16x4-bit fast-scan +
# 32x8-bit table refinement, OPQ, 40 B/vector) with Starling on BIGANN-10M, DEEP-10M and
# T2I-10M. Serial: builds, SSD SLC flush (20 GB write+delete) + settle, qd1 check of every
# index file, then two interleaved repetitions per dataset of our parameter grid and
# Starling's Ls sweep at 1 and 8 threads, qd1 check again. Pick Pareto points on rep 1,
# report them from rep 2 (tools/evolve/e2e/compare_e2e.py).
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
mkdir -p $OUT
PY=.venv/bin/python
PROG=tools/evolve/ablation/abl_v2.py
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
declare -A STAGE=([bigann]=full [deep]=heldout_deep [t2i]=heldout_t2i)
declare -A DS=([bigann]=bigann10m-q2k [deep]=deep10m-q2k [t2i]=t2i10m-q2k)
declare -A MET=([bigann]=L2 [deep]=L2 [t2i]=IP)
declare -A SB=([bigann]=configs/evolve/baselines/starling_b040.yaml [deep]=configs/evolve/baselines/starling_heldout.yaml [t2i]=configs/evolve/baselines/starling_heldout.yaml)
declare -A ONLY=([bigann]="Starling-B0.40" [deep]="Starling-B0.32-deep Starling-B0.40-deep" [t2i]="Starling-B0.32-t2i Starling-B0.40-t2i")
declare -A SW=(
	[bigann]='[{"Ls": 12}, {"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}]'
	[deep]='[{"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}]'
	[t2i]='[{"Ls": 25}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}, {"Ls": 160}, {"Ls": 250}]'
)
G=tools/evolve/e2e/heldout_paths.py
declare -A PTS=(
	[bigann]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 12,16,24,32)
	[deep]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 16,24,32,48)
	[t2i]=$($PY $G grid 128:1024,192:1536,256:2048,384:3072,512:4096,768:6144 32,48,64,96,128)
)
idx() { $PY $G index "$1" "${STAGE[$1]}" "${DS[$1]}" $PROG; }
ours() { # $1 dataset, $2 mode flags, $3 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$(idx "$1")":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program /repo/$PROG \
		--index /idx --queries "/data/${DS[$1]}/queries.npy" --gt "/data/${DS[$1]}/ground_truth.npy" \
		--metric "${MET[$1]}" --points "${PTS[$1]}" $2 >"$OUT/$3.jsonl"
}
star() { # $1 dataset, $2 threads, $3 out
	$PY tools/evolve/e2e/run_baselines_e2e.py ${SB[$1]} --threads "$2" --out "$3" \
		--only ${ONLY[$1]} --sweep "${SW[$1]}"
}

# 1. builds (their modelled measurements are kept; timings of fresh files are not used)
for d in bigann deep t2i; do
	$PY tools/evolve/evolve_bench.py measure $PROG --stage "${STAGE[$d]}" --name "v2-$d-model"
done

# 2. fold everything out of the SLC cache, settle, check
$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$(idx bigann)/disk/lists.pages $(idx deep)/disk/lists.pages $(idx t2i)/disk/lists.pages"
for a in Starling-B0.40 Starling-B0.32-deep Starling-B0.40-deep Starling-B0.32-t2i Starling-B0.40-t2i; do
	FILES="$FILES $(find /home/isfcr/evolve_indices/baselines/$a -type f -size +3G)"
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES

# 3. interleaved repetitions
for rep in 1 2; do
	for d in bigann deep t2i; do
		ours $d "--threads 1" "v2-$d-t1-r$rep"
		star $d 1 "v2star-$d-t1-r$rep"
		ours $d "--threads 8" "v2-$d-t8-r$rep"
		star $d 8 "v2star-$d-t8-r$rep"
	done
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
