# Research tools

Analysis and data-preparation scripts from the ANN-on-SSD latency-under-load
research project, migrated in from the isfcr benchmark host so every
experiment is reproducible from this repo instead of loose scripts under a
researcher's home directory. All scripts are plain `uv run python
tools/research/<script>.py --help`-able, take paths/params as CLI args (no
hard-coded home paths), and print results to stdout for redirection.

## Dataset preparation

- `prepare_bigann.py` - range-download a BIGANN (SIFT1B) prefix of N base
  vectors + the 10K query set, in native `.u8bin` (no `.npy` conversion -
  `ann_suite.datasets.loader.DatasetLoader` reads `.u8bin`/`.fbin` directly
  when the dataset config sets the matching `point_type`). Ground truth via
  PipeANN's `compute_groundtruth` if `--pipeann-bin` is given.
- `prepare_deep.py` - same recipe for DEEP1B (`.fbin`, float32, dim=96).

See `docs/ADDING_DATASETS.md` and the `bigann-10m`/`bigann-100m`/`deep-100m`
entries in `data/registry.yaml` for the full recipe and provenance notes.

## Locality / co-location analysis

These three came from an investigation into how much of PipeANN's
beam-search I/O cost is attributable to base-vector layout (i.e., would
re-clustering the base set so a query's true neighbors land in fewer disk
blocks meaningfully cut I/Os-per-query).

- `colocate.py` - clusters the base set into groups of a given size (a
  proxy for "disk blocks"), reports how many groups a query's true top-k
  spans, and how many IVF-style probes are needed to recover 90% of it.
- `largek.py` - the same analysis, but sweeping ground-truth depth k
  (10/100/1000) instead of group size, to see how span/probes scale as the
  recall target widens.
- `knn_cover.py` - seeds from a query's rank-1/2/5 true neighbor, expands
  that neighbor's own k-NN list, and reports what fraction of the query's
  true top-10 gets recovered - i.e., how much "free" recall comes from
  graph-adjacency to an already-found true neighbor.

Example (BIGANN-10M, matching the original exploratory runs):

```bash
uv run python tools/research/colocate.py \
  --base data/sift1m/base.npy --queries data/sift1m/queries.npy \
  --ground-truth data/sift1m/ground_truth.npy --dim 128

uv run python tools/research/knn_cover.py \
  --base data/bigann-10m/base10m.u8bin --base-format bigann_bin --base-dtype uint8 \
  --queries data/bigann-10m/query.u8bin --query-format bigann_bin --query-dtype uint8 \
  --ground-truth data/bigann-10m/gt10m.bin --gt-format bigann_bin \
  --num-queries 2000 --dim 128
```

These need `faiss-cpu` (not part of the core ann-suite runtime deps, since
they're offline analysis, not part of the benchmark loop):

```bash
uv sync --group research
```

(See `pyproject.toml`'s `research` optional dependency group, added to keep
`faiss-cpu` out of the default install for people who only run benchmarks.)

## Device I/O characterization

- `fio_characterize.sh <target-file-or-device> [output.tsv]` - sweeps
  block-size x jobs x iodepth randread fio jobs against a file (typically an
  already-built index file, so the read pattern hits the real filesystem/
  device) and writes a TSV of iops/bandwidth/latency/CPU per point. Useful
  for characterizing a new benchmark host's NVMe before trusting absolute
  I/O-latency numbers from it (device class matters - see the "Research
  workflow" section of the main README).

```bash
tools/research/fio_characterize.sh \
  indices/PipeANN/bigann-10m/R64_L128_pq32/pipeann_disk.index \
  results/fio/bigann10m_nvme.tsv
```

## Provenance

Ported 2026-09 from `~/research/py/{colocate.py,largek.py,knn_cover.py}` and
`~/research/fio/run.sh` on the isfcr benchmark host (uv project
`~/research/py`, using `faiss-cpu`). Original scripts hard-coded
`/home/isfcr/...` paths and fixed parameters inline; behavior is otherwise
unchanged (same algorithms, same default parameters), only the CLI/path
plumbing was generalized.
