#!/bin/bash
# End-to-end component ablation of design v2 (tools/evolve/ablation/abl_v2*.py), plus a
# BIGANN re-measurement of v2 and Starling-B0.40 on a verified-folded SSD (the v2 BIGANN
# file read 81 us at qd1 in run_v2_e2e.sh) and higher-recall T2I points. Serial: builds,
# SLC flush (20 GB) + settle, qd1 check, two interleaved repetitions at 1 and 8 threads,
# qd1 check. Compare with tools/evolve/e2e/compare_v2.py (DATASET or variant-DATASET, e.g. noopq-deep).
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
PY=.venv/bin/python
A=tools/evolve/ablation
G=tools/evolve/e2e/heldout_paths.py
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
declare -A STAGE=([bigann]=full [deep]=heldout_deep [t2i]=heldout_t2i)
declare -A DS=([bigann]=bigann10m-q2k [deep]=deep10m-q2k [t2i]=t2i10m-q2k)
declare -A MET=([bigann]=L2 [deep]=L2 [t2i]=IP)
declare -A PTS=(
	[bigann]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 12,16,24,32)
	[deep]=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 16,24,32,48)
	[t2i]=$($PY $G grid 512:4096,768:6144,1024:8192,1536:12288 128,192,256,320)
)
# variant -> program; result files are v2-<dataset>-... (base) or v2-<variant>-<dataset>-...
declare -A PROG=([base]=$A/abl_v2.py [noopq]=$A/abl_v2_noopq.py [cluster]=$A/abl_v2_cluster.py [m12]=$A/abl_v2_m12.py)
idx() { $PY $G index "$1" "${STAGE[$1]}" "${DS[$1]}" "${PROG[$2]}"; } # $1 dataset, $2 variant
ours() { # $1 dataset, $2 variant, $3 mode flags, $4 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$(idx "$1" "$2")":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program "/repo/${PROG[$2]}" \
		--index /idx --queries "/data/${DS[$1]}/queries.npy" --gt "/data/${DS[$1]}/ground_truth.npy" \
		--metric "${MET[$1]}" --points "${PTS[$1]}" $3 >"$OUT/$4.jsonl"
}
star() { # $1 config, $2 threads, $3 out, $4 sweep, $5.. names
	cf=$1 th=$2 o=$3 sw=$4
	shift 4
	$PY tools/evolve/e2e/run_baselines_e2e.py $cf --threads "$th" --out "$o" --only "$@" --sweep "$sw"
}
RUNS="bigann:base bigann:noopq bigann:cluster deep:noopq deep:cluster deep:m12"

# 1. builds
for r in $RUNS; do
	d=${r%%:*} v=${r#*:}
	[ -d "$(idx $d "$v")/disk" ] || $PY tools/evolve/evolve_bench.py measure "${PROG[$v]}" \
		--stage "${STAGE[$d]}" --name "v2-$v-$d-model"
done

# 2. fold everything out of the SLC cache, settle, check
$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$(idx t2i base)/disk/lists.pages $(find /home/isfcr/evolve_indices/baselines/Starling-B0.40 -type f -size +1G)"
for r in $RUNS; do FILES="$FILES $(idx ${r%%:*} ${r#*:})/disk/lists.pages"; done
python3 tools/evolve/e2e/ssd_qd1.py $FILES

# 3. interleaved repetitions (rep numbers 3/4 for BIGANN v2 + Starling: a new SSD session)
for rep in 1 2; do
	for r in $RUNS; do
		d=${r%%:*} v=${r#*:}
		[ "$v" = base ] && n="v2-$d" || n="v2-$v-$d"
		[ "$v" = base ] && rr=$((rep + 2)) || rr=$rep
		ours $d "$v" "--threads 1" "$n-t1-r$rr"
		ours $d "$v" "--threads 8" "$n-t8-r$rr"
	done
	star configs/evolve/baselines/starling_b040.yaml 1 "v2star-bigann-t1-r$((rep + 2))" \
		'[{"Ls": 12}, {"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}]' Starling-B0.40
	star configs/evolve/baselines/starling_b040.yaml 8 "v2star-bigann-t8-r$((rep + 2))" \
		'[{"Ls": 12}, {"Ls": 15}, {"Ls": 20}, {"Ls": 25}, {"Ls": 30}, {"Ls": 40}, {"Ls": 60}]' Starling-B0.40
	ours t2i base "--threads 1" "v2-t2ihi-t1-r$rep"
	ours t2i base "--threads 8" "v2-t2ihi-t8-r$rep"
	star configs/evolve/baselines/starling_heldout.yaml 1 "v2star-t2ihi-t1-r$rep" \
		'[{"Ls": 250}, {"Ls": 400}, {"Ls": 600}]' Starling-B0.32-t2i Starling-B0.40-t2i
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
