#!/bin/bash
# Held-out end-to-end comparison (DEEP-10M L2, T2I-10M IP): our design
# (tools/evolve/ablation/abl_fastscan_g40_*.py) vs Starling B0.32 / B0.40. Serial:
# builds, SSD SLC flush (20 GB write+delete) + settle, qd1 check of every index file,
# two interleaved repetitions at 1 and 8 threads, qd1 check again.
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
mkdir -p $OUT
PY=.venv/bin/python
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
SB=configs/evolve/baselines/starling_heldout.yaml
declare -A STAGE=([deep]=heldout_deep [t2i]=heldout_t2i)
declare -A DS=([deep]=deep10m-q2k [t2i]=t2i10m-q2k)
declare -A MET=([deep]=L2 [t2i]=IP)
declare -A SW
SW[deep]='[{"Ls": 15}, {"Ls": 25}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}]'
SW[t2i]='[{"Ls": 25}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}, {"Ls": 160}, {"Ls": 250}]'

idx() { # cached index dir of our program for dataset $1
	$PY tools/evolve/e2e/heldout_paths.py index "$1" "${STAGE[$1]}" "${DS[$1]}"
}
pts() { # SEARCH_POINTS of our program for dataset $1, as JSON
	$PY tools/evolve/e2e/heldout_paths.py points "$1"
}
ours() { # $1 dataset, $2 mode flags, $3 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$(idx "$1")":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program "/repo/tools/evolve/ablation/abl_fastscan_g40_$1.py" \
		--index /idx --queries "/data/${DS[$1]}/queries.npy" --gt "/data/${DS[$1]}/ground_truth.npy" \
		--metric "${MET[$1]}" --points "$(pts "$1")" $2 >"$OUT/$3.jsonl"
}
star() { # $1 dataset, $2 threads, $3 out
	$PY tools/evolve/e2e/run_baselines_e2e.py $SB --threads "$2" --out "$3" \
		--only "Starling-B0.32-$1" "Starling-B0.40-$1" --sweep "${SW[$1]}"
}

# 1. builds (timings of these runs are discarded: fresh indexes read from the SLC cache)
for d in deep t2i; do
	$PY tools/evolve/evolve_bench.py measure "tools/evolve/ablation/abl_fastscan_g40_$d.py" \
		--stage "${STAGE[$d]}" --name "g40-$d-model"
done
$PY tools/evolve/e2e/run_baselines_e2e.py $SB --build --threads 1 --out heldout-build-discard

# 2. fold everything out of the SLC cache, settle
$PY tools/evolve/e2e/heldout_paths.py flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$(idx deep)/disk/lists.pages $(idx t2i)/disk/lists.pages"
for a in Starling-B0.32-deep Starling-B0.40-deep Starling-B0.32-t2i Starling-B0.40-t2i; do
	FILES="$FILES $(find /home/isfcr/evolve_indices/baselines/$a -type f -size +3G)"
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES

# 3. interleaved repetitions
for rep in 1 2; do
	for d in deep t2i; do
		ours $d "--threads 1" "ours-g40-$d-t1-r$rep"
		star $d 1 "starling-$d-t1-r$rep"
		ours $d "--threads 8" "ours-g40-$d-t8-r$rep"
		star $d 8 "starling-$d-t8-r$rep"
	done
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
