#!/bin/bash
# Second DRAM budget: design v2 at 24 B/vector (tools/evolve/ablation/abl_v2_b24.py, ~410 MB
# at 10M) vs Starling-B0.32 (~476-593 MB) on BIGANN, DEEP and T2I. Serial: builds, SLC
# flush (20 GB) + settle, qd1 check, two interleaved repetitions at 1 and 8 threads, qd1
# check. Compare: compare_v2.py b24-bigann / b24-deep / b24-t2i (Starling rows from
# v2star-b24-*: B0.32 only).
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
PY=.venv/bin/python
P=tools/evolve/ablation/abl_v2_b24.py
G=tools/evolve/e2e/heldout_paths.py
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
declare -A STAGE=([bigann]=full [deep]=heldout_deep [t2i]=heldout_t2i)
declare -A DS=([bigann]=bigann10m-q2k [deep]=deep10m-q2k [t2i]=t2i10m-q2k)
declare -A MET=([bigann]=L2 [deep]=L2 [t2i]=IP)
declare -A SB=([bigann]=configs/evolve/baselines/starling.yaml [deep]=configs/evolve/baselines/starling_heldout.yaml [t2i]=configs/evolve/baselines/starling_heldout.yaml)
declare -A ONLY=([bigann]=Starling-B0.32 [deep]=Starling-B0.32-deep [t2i]=Starling-B0.32-t2i)
declare -A SW=(
	[bigann]='[{"Ls": 12}, {"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}]'
	[deep]='[{"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}]'
	[t2i]='[{"Ls": 25}, {"Ls": 40}, {"Ls": 60}, {"Ls": 100}, {"Ls": 160}, {"Ls": 250}]'
)
declare -A PTS=(
	[bigann]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 12,16,24,32)
	[deep]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 16,24,32,48)
	[t2i]=$($PY $G grid 128:1024,192:1536,256:2048,384:3072,512:4096,768:6144 32,48,64,96,128)
)
idx() { $PY $G index "$1" "${STAGE[$1]}" "${DS[$1]}" $P; }
ours() { # $1 dataset, $2 mode flags, $3 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$(idx "$1")":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program /repo/$P \
		--index /idx --queries "/data/${DS[$1]}/queries.npy" --gt "/data/${DS[$1]}/ground_truth.npy" \
		--metric "${MET[$1]}" --points "${PTS[$1]}" $2 >"$OUT/$3.jsonl"
}
star() { # $1 dataset, $2 threads, $3 out
	$PY tools/evolve/e2e/run_baselines_e2e.py ${SB[$1]} --threads "$2" --out "$3" \
		--only ${ONLY[$1]} --sweep "${SW[$1]}"
}

for d in bigann deep t2i; do
	[ -d "$(idx $d)/disk" ] || $PY tools/evolve/evolve_bench.py measure $P --stage "${STAGE[$d]}" --name "v2-b24-$d-model"
done
$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$(idx bigann)/disk/lists.pages $(idx deep)/disk/lists.pages $(idx t2i)/disk/lists.pages"
for a in Starling-B0.32 Starling-B0.32-deep Starling-B0.32-t2i; do
	FILES="$FILES $(find /home/isfcr/evolve_indices/baselines/$a -type f -size +1G)"
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
for rep in 1 2; do
	for d in bigann deep t2i; do
		ours $d "--threads 1" "v2-b24-$d-t1-r$rep"
		star $d 1 "v2star-b24-$d-t1-r$rep"
		ours $d "--threads 8" "v2-b24-$d-t8-r$rep"
		star $d 8 "v2star-b24-$d-t8-r$rep"
	done
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
