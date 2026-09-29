# Research Workflow: ANN-on-SSD Latency-Under-Load Project

This documents where the migrated ANN-on-SSD research project's data,
indices, configs and results live inside `ann-suite`, and how to reproduce
each experiment from a config. The project itself studies PipeANN's
pipelined I/O search mode vs. plain DiskANN-style beam search under load,
on the isfcr research host (100.83.46.2).

## Where things live

### Datasets (`data/`)

Registered in `data/registry.yaml`, kept in their native
big-ann-benchmarks binary format (`.u8bin`/`.fbin`, not converted to
`.npy` - `ann_suite.datasets.loader.DatasetLoader` memmaps this format
natively when `point_type` is set):

| Dataset | Path | Notes |
|---|---|---|
| `bigann-10m` | `data/bigann-10m/{base10m.u8bin,query.u8bin,gt10m.bin}` | BIGANN (SIFT1B) 10M-vector prefix, 128D uint8, L2. Top-100 ground truth. |
| `bigann-100m` | `data/bigann-100m/{base100m.u8bin,query.u8bin,gt100m.bin}` | Same source file, 100M-vector prefix. `query.u8bin` is a symlink to `bigann-10m`'s copy (identical query set, saved as one physical file). |
| `deep-100m` | `data/deep-100m/{base100m.fbin,query.fbin}` | DEEP1B 100M-vector prefix, 96D float32, L2. **No ground truth yet** - see below. |

Reproduce/re-download any of these with `tools/research/prepare_bigann.py`
or `tools/research/prepare_deep.py` (see `prepare_example` in
`data/registry.yaml` for exact invocations, and the "Provenance" section
in [PIPEANN.md](./PIPEANN.md)).

**DEEP-100M ground truth is not yet computed.** The lead researcher had
not computed it at migration time. To add it:

```bash
uv run python tools/research/prepare_deep.py \
  --output-dir data/deep-100m --base-count 100000000 --skip-query \
  --pipeann-bin ~/research/PipeANN/build-ro/tests
```

This is a CPU-heavy brute-force k-NN computation over 100M x 96D vectors -
treat it as "heavy work" under the isfcr `~/research/BENCH_LOCK` protocol
(wait for the lock to clear before running it; it was intentionally **not**
run during migration while the lead's PiPNN-100M build held the lock).

### Prebuilt indices (`indices/`)

Registered as `build.prebuilt_path` + `build.required_files` in configs
(see below), physically at:

| Index | Path | R / L | Notes |
|---|---|---|---|
| BIGANN-10M R=32 | `indices/PipeANN/bigann-10m/R32_L128_pq32/` | 32 / 128 | |
| BIGANN-10M R=64 | `indices/PipeANN/bigann-10m/R64_L128_pq32/` | 64 / 128 | Baseline; matches the lead's standalone reference numbers. |
| BIGANN-10M R=128 | `indices/PipeANN/bigann-10m/R128_L192_pq32/` | 128 / 192 | |
| BIGANN-10M R=256 | `indices/PipeANN/bigann-10m/R256_L320_pq32/` | 256 / 320 | |

All four carry the on-disk PipeANN index (`idx_disk.index`,
`idx_pq_pivots.bin`, `idx_pq_compressed.bin`), a 1% in-memory entry-point
index (`idx_mem.index*`), and the sample files DiskANN's disk-index build
leaves behind (`idx_SAMPLE_RATE_0.01_*`, present only for R=64 - the
original build script only kept these for the baseline run).

`R=512` (`L=576`) is intentionally **not** included anywhere: that build
was aborted on isfcr and no index for it exists.

`indices/PipeANN/bigann-100m/` does not exist yet - the lead's BIGANN-100M
PipeANN index build was in progress (`~/data/idx/bigann100m_pipnn`) at
migration time and was left untouched. `~/data/idx/bigann100m` (an empty
directory reserved for a future build) was also left as-is; nothing to
migrate there yet.

Every prebuilt-index config uses `required_files` to fail loudly (rather
than at opaque search time) if a build was interrupted before its final
merge - see `BuildConfig.required_files` in `src/ann_suite/core/schemas.py`.

### Old-path symlinks (isfcr host only)

To keep the lead researcher's existing shell scripts (`~/data/*.sh`,
`~/research/py/*.py`) working unchanged, every moved file/directory has a
symlink left at its old path:

```
~/data/bigann/base10m.u8bin   -> ~/shivang/ann-suite/data/bigann-10m/base10m.u8bin
~/data/bigann/gt10m.bin       -> ~/shivang/ann-suite/data/bigann-10m/gt10m.bin
~/data/bigann/base100m.u8bin  -> ~/shivang/ann-suite/data/bigann-100m/base100m.u8bin
~/data/bigann/gt100m.bin      -> ~/shivang/ann-suite/data/bigann-100m/gt100m.bin
~/data/bigann/query.u8bin     -> ~/shivang/ann-suite/data/bigann-10m/query.u8bin
~/data/deep/base100m.fbin     -> ~/shivang/ann-suite/data/deep-100m/base100m.fbin
~/data/deep/query.fbin        -> ~/shivang/ann-suite/data/deep-100m/query.fbin
~/data/idx/bigann10m          -> ~/shivang/ann-suite/indices/PipeANN/bigann-10m/R64_L128_pq32
~/data/idx/bigann10m_R32      -> ~/shivang/ann-suite/indices/PipeANN/bigann-10m/R32_L128_pq32
~/data/idx/bigann10m_R128     -> ~/shivang/ann-suite/indices/PipeANN/bigann-10m/R128_L192_pq32
~/data/idx/bigann10m_R256     -> ~/shivang/ann-suite/indices/PipeANN/bigann-10m/R256_L320_pq32
```

`~/data/bigann/base1b.u8bin` (in-progress download) and
`~/data/idx/bigann100m_pipnn` (in-progress build) were **not** touched or
symlinked - they were never moved.

The move was done with `mv` within the same filesystem
(`/dev/nvme0n1p2`), so it was near-instant and no bytes were copied; sizes
were verified identical before/after for every file.

### Configs (`configs/`)

| Config | Reproduces | Notes |
|---|---|---|
| `pipeann_bigann10m_degree_build.yaml` | `~/data/build_deg.sh` + `~/data/build10m.sh` | Fresh-builds R in {32,64,128,256} at their paired L (128/128/192/320). R=512 excluded (aborted upstream). |
| `pipeann_bigann10m_prebuilt_degrees.yaml` | Same R sweep, search-only | Reuses the migrated prebuilt indices above via `build.prebuilt_path` instead of rebuilding - use this one unless you're specifically testing the build phase. |
| `pipeann_sqpoll_comparison.yaml` | isfcr SQPOLL on/off A/B test | Pipe mode W=32, 24 workers, BIGANN-10M R=64, rates 500/2000/4000 QPS. |
| `pipeann_closed_loop_lsweep.yaml` | Closed-loop 1-thread Ls x mem_L sweep | beam W8 (mode 0) vs pipe W32 (mode 2), Ls in {20,40,80,150} x mem_L in {0,10}. |
| `pipeann_ol4_static_width_sweep.yaml` | `~/research/runs/ol4.sh` | Open-loop static-width sweep: beamW4/W8, pipeW4/8/16/32, rates 500/2000/3500/4500 QPS. See the config's header comment for the `num_queries` reproduction caveat. |
| `pipeann_prebuilt_example.yaml` | n/a | Generic illustrative example (sift1m), not tied to isfcr data. |

All BIGANN-10M search-only configs above use `index_prefix: idx` (matching
the on-disk prefix these indices actually carry, from the original
`build_disk_index -x idx ...` invocations) and
`required_files: [idx_disk.index, idx_pq_pivots.bin, idx_pq_compressed.bin]`.

## Reproducing an experiment

On the isfcr research host, once `~/research/BENCH_LOCK` is clear:

```bash
cd ~/shivang/ann-suite
uv run ann-suite build --algorithm pipeann      # once, or after algorithm changes
uv run ann-suite run --config configs/pipeann_bigann10m_prebuilt_degrees.yaml
uv run ann-suite run --config configs/pipeann_sqpoll_comparison.yaml
uv run ann-suite run --config configs/pipeann_closed_loop_lsweep.yaml
uv run ann-suite run --config configs/pipeann_ol4_static_width_sweep.yaml   # x3 for "3 reps"
```

Results land under `results/<config-name>_<timestamp>/`. See
[EXPERIMENT_ANALYSIS.md](./EXPERIMENT_ANALYSIS.md) for the results schema
and [OPEN_LOOP.md](./OPEN_LOOP.md) for open-loop-specific metrics
(`achieved_qps`, p50/p99 latency, `machine_cpu_util`).

## Validating against the lead's standalone reference numbers

The lead's standalone (non-ann-suite) PipeANN runs on BIGANN-10M R=64
produced a specific set of open-loop and closed-loop reference numbers
(24 workers, `PIPEANN_SQPOLL=0`, K=10, medians of 3 reps). Reproduce and
compare them via `pipeann_ol4_static_width_sweep.yaml` (open-loop) and
`pipeann_closed_loop_lsweep.yaml` (closed-loop, 1-thread deterministic
checks). The per-point comparison table and any investigated
discrepancies are reported in the PR description for the branch that adds
this document (`research-migration`) - re-run the validation and update
that table if the reference numbers or algorithm code change materially.

## Analysis tools (`tools/research/`)

Ported from the isfcr host's `~/research/py/*.py`:

- `prepare_bigann.py`, `prepare_deep.py` - dataset download/prepare (see above).
- `colocate.py` - co-visitation-based graph node packing/reordering analysis.
- `largek.py` - large-K exact rerank behavior.
- `knn_cover.py` - k-NN coverage/reachability analysis.
- `fio_characterize.sh` - raw device I/O characterization (independent of any index).

These have their own `research` uv dependency group (`faiss-cpu`, kept out
of the default install): `uv sync --group research`. See
`tools/research/README.md` for per-script usage.
