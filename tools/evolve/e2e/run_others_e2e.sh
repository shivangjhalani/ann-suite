#!/bin/bash
# The other packaged disk-ANN systems under the same end-to-end protocol on BIGANN-10M at
# the ~0.45 GB budget: DiskANN-B0.32, PipeANN-pq32, PageANN-B0.32, LAANN-B0.32 (each with
# its config's own sweep), Starling-B0.32, and design v2 at 24 B/vector
# (tools/evolve/ablation/abl_v2_b24.py). Serial: SLC flush (20 GB) + settle, qd1 check, two
# interleaved repetitions at 1 and 8 threads, qd1 check. Results: v2-b24-bigann-t*-r{3,4},
# others-<threads>-r<rep>-<config>.json. Compare: compare_others.py.
set -x
cd /home/isfcr/shivang/ann-suite
OUT=results/evolve/e2e
PY=.venv/bin/python
P=tools/evolve/ablation/abl_v2_b24.py
G=tools/evolve/e2e/heldout_paths.py
B=configs/evolve/baselines
ENV="-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1"
PTS=$($PY $G grid 48:384,64:512,96:768,128:1024,192:1536,256:2048,384:3072 12,16,24,32)
IDX=$($PY $G index bigann full bigann10m-q2k $P)
ours() { # $1 mode flags, $2 out
	docker run --rm $ENV --entrypoint python --cpuset-cpus 0-7 --memory 1g --network none \
		-v "$PWD":/repo:ro -v "$IDX":/idx:ro -v /home/isfcr/data:/data:ro ann-suite/evolved:latest \
		/repo/tools/evolve/e2e/e2e_bench.py --program /repo/$P --index /idx \
		--queries /data/bigann10m-q2k/queries.npy --gt /data/bigann10m-q2k/ground_truth.npy \
		--metric L2 --points "$PTS" $1 >"$OUT/$2.jsonl"
}
base() { # $1 config, $2 threads, $3 out, $4.. names (each config's own sweep)
	cf=$1 th=$2 o=$3
	shift 3
	$PY tools/evolve/e2e/run_baselines_e2e.py $B/$cf --threads "$th" --out "$o" --only "$@"
}

$PY $G flush /home/isfcr/slc_flush.bin 20
sleep 600
FILES="$IDX/disk/lists.pages"
for a in DiskANN-B0.32 PipeANN-pq32 PageANN-B0.32 LAANN-B0.32 Starling-B0.32; do
	FILES="$FILES $(find /home/isfcr/evolve_indices/baselines/$a -type f -size +1G)"
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES

for rep in 1 2; do
	for th in 1 8; do
		ours "--threads $th" "v2-b24-bigann-t$th-r$((rep + 2))"
		base diskann.yaml $th "others-t$th-r$rep-diskann" DiskANN-B0.32
		base pipeann.yaml $th "others-t$th-r$rep-pipeann" PipeANN-pq32
		base pageann.yaml $th "others-t$th-r$rep-pageann" PageANN-B0.32 LAANN-B0.32
		base starling.yaml $th "others-t$th-r$rep-starling" Starling-B0.32
	done
done
python3 tools/evolve/e2e/ssd_qd1.py $FILES
echo ALLDONE
