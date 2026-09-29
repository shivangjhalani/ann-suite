# PipeANN

[PipeANN](https://github.com/thustorage/PipeANN) is a disk-based ANN index
(Vamana/DiskANN-style graph on SSD, PQ- or RaBitQ-compressed) with a
pipelined search mode that overlaps SSD I/O with distance computation. It
ships as pure C++ binaries (no Python bindings), so `library/algorithms/pipeann`
builds them from source and its `algorithm/runner.py` shells out to them,
following the same subprocess pattern as the SPANN algorithm.

## Build

```bash
uv run ann-suite build --algorithm pipeann
```

The image is built at a pinned upstream commit (`PIPEANN_COMMIT` build arg,
default `b9ec4ba`), in `-DREAD_ONLY_TESTS -DNO_MAPPING` (search-only) mode -
this disables the tag<->id mapping table and update paths, which is the
configuration upstream recommends for search benchmarking (the default build
routes reads through a refcounted page cache shared by concurrent beam-search
queries, which skews I/O counts otherwise).

On top of that pinned commit, the image applies a small "load-aware" patch
(`library/algorithms/pipeann/patches/load_aware.patch`) plus two new vendored
files (`vendor/io_governor.h`, `vendor/search_openloop.cpp`). These come from
a fork used for latency-under-load research on the isfcr benchmark host
(`~/research/PipeANN`, branch `load-aware`) and add:

- An optional, env-controlled I/O width governor (`PIPEANN_QSTAR`,
  `PIPEANN_WMIN` - see `search_args.qstar` / `search_args.wmin` below). Unset
  = stock PipeANN behavior.
- `PIPEANN_SQPOLL=0` to disable per-thread io_uring SQPOLL (`search_args.sqpoll:
  false`). With many search threads, SQPOLL pins a CPU core per thread even
  when the workload is I/O-bound, which can saturate the machine.
- `search_openloop`, a standalone open-loop (Poisson/closed arrival) search
  driver - see [OPEN_LOOP.md](./OPEN_LOOP.md).

## Vector dtype

Unlike the DiskANN/SPANN runners, PipeANN's runner does **not** take a
separate `vector_dtype`/`point_type` config value that has to be kept in sync
with the dataset by hand - it infers the element type (`uint8`/`int8`/`float`)
directly from the loaded `.npy` array's dtype, both at build and search time.

## Build parameters (`build.args`)

| Key | Default | Meaning |
|-----|---------|---------|
| `index_prefix` | `"pipeann"` | File prefix under the index directory (`<prefix>_disk.index`, etc.) |
| `R` | 96 | Max out-neighbors (graph degree) |
| `L` | 128 | Build-time candidate list size (Vamana `L`, or PiPNN `L1`) |
| `pq_bytes` | 32 | Bytes per PQ-compressed vector |
| `build_memory_gb` | 64 | Advisory build memory budget (GB) |
| `num_threads` | `os.cpu_count()` | Build parallelism |
| `nbr_type` | `"pq"` | `pq` (supports update), `rabitq`, or `rabitq{3-5}` (search-only) |
| `builder` | `"vamana"` | `"vamana"` (recommended) or `"pipnn"` (experimental; requires `L2 > 0`) |
| `L2` | 0 | PiPNN's second-level candidate count (`builder: pipnn` only) |
| `build_mem_index` | `true` | Also build an in-memory entry-point index from a 1% sample |
| `mem_sample_rate` | 0.01 | Sample rate for the in-memory entry-point index |
| `mem_R`, `mem_L`, `mem_alpha`, `mem_threads` | 32, 64, 1.2, `num_threads` | In-memory index build params |

`required_files` for a `prebuilt_path` typically looks like
`["<prefix>_disk.index", "<prefix>_pq_pivots.bin", "<prefix>_pq_compressed.bin"]`
for `nbr_type: pq` (see `configs/pipeann_prebuilt_example.yaml`).

## Search parameters (`search.args`)

| Key | Default | Meaning |
|-----|---------|---------|
| `index_prefix` | `"pipeann"` | Must match the build's prefix |
| `mode` | 2 | `0` DiskANN best-first beam search, `2` PipeANN pipelined (recommended), `3` CoroSearch. (`1`, Starling page search, needs a reordered index and isn't wired up here.) |
| `beam_width` | 32 | In-flight I/O width per query |
| `mem_L` | 0 | In-memory entry-point candidate list size; `0` skips it even if built |
| `Ls` | 100 | Search-time candidate list size (sweepable, like other algorithms) |
| `num_threads` | 1 | Worker threads for closed-loop search |
| `nbr_type` | `"pq"` | Must match how the index was built |
| `sqpoll` | `true` | `false` disables io_uring SQPOLL (`PIPEANN_SQPOLL=0`) |
| `qstar` | unset | I/O width governor cap (`PIPEANN_QSTAR`); unset = stock behavior |
| `wmin` | unset | Governor's minimum per-query width (`PIPEANN_WMIN`) |

See [OPEN_LOOP.md](./OPEN_LOOP.md) for `search.arrival`.

## Known limitations

- PipeANN's search binaries load the index and run the timed search in one
  process invocation, with no way to observe a separate "index load"
  timestamp from the outside. The runner reports the whole invocation as the
  search window (zero warmup), matching the SPANN runner's convention, rather
  than mislabeling real search I/O as warmup.
- `search.arrival.seed` is accepted by the schema but the vendored
  `search_openloop` binary uses a fixed internal RNG seed (`12345`) for its
  Poisson arrival schedule; true per-run seed control would need a further
  small patch to `vendor/search_openloop.cpp` (not done here - the C++ change
  couldn't be compile-verified in this environment).
- Mode `1` (Starling page search) is not wired up: it needs a reordered index
  built by upstream's separate Starling tooling.
- This algorithm has not been built or run end-to-end in this environment
  (see the PR description for what's untested).
