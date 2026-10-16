#!/bin/bash
# Tail-latency study of design v2 (1 thread, BIGANN-10M and DEEP-10M): per-query latency,
# CPU and pages (e2e_bench.py --dump) at fixed points, Python GC on vs off, three
# alternating repetitions. Analyse with tail_analysis.py results/evolve/e2e/tail.
# Serial: SLC flush + settle + qd1 first.
set -x
cd /home/isfcr/shivang/ann-suite
PY=.venv/bin/python
P=tools/evolve/ablation/abl_v2.py
G=tools/evolve/e2e/heldout_paths.py
T=results/evolve/e2e/tail
mkdir -p $T
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
declare -A STAGE=([bigann]=full [deep]=heldout_deep)
declare -A DS=([bigann]=bigann10m-q2k [deep]=deep10m-q2k)
declare -A PTS=(
	[bigann]='[{"nprobe": 64, "shortlist": 512, "pages": 16}, {"nprobe": 128, "shortlist": 1024, "pages": 24}]'
	[deep]='[{"nprobe": 128, "shortlist": 1024, "pages": 24}, {"nprobe": 192, "shortlist": 1536, "pages": 24}]'
)
idx() { $PY $G index "$1" "${STAGE[$1]}" "${DS[$1]}" $P; }
ours() { # $1 dataset, $2 gc mode, $3 name
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$PWD/$T":/out -v "$(idx "$1")":/idx:ro -v /home/isfcr/data:/data:ro \
		ann-suite/evolved:latest /repo/tools/evolve/e2e/e2e_bench.py --program /repo/$P --index /idx \
		--queries "/data/${DS[$1]}/queries.npy" --gt "/data/${DS[$1]}/ground_truth.npy" \
		--points "${PTS[$1]}" --threads 1 --gc "$2" --dump "/out/$3" >"$T/$3.jsonl"
}
$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
python3 tools/evolve/e2e/ssd_qd1.py "$(idx bigann)/disk/lists.pages" "$(idx deep)/disk/lists.pages"
for rep in 1 2 3; do
	for d in bigann deep; do
		for g in on off; do
			ours $d $g "$d-gc$g-r$rep"
		done
	done
done
echo ALLDONE
