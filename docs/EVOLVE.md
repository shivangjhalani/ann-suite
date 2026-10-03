# Evolving disk-ANN designs with OpenEvolve

ann-suite is the evaluator for an OpenEvolve run that searches for disk-resident
ANN designs. OpenEvolve (on the researcher's machine, LLM = Claude Code CLI) writes
candidate programs; each one is built and searched here, in Docker, under the same
rules as the published baselines, and scored by how far it pushes past their
measured Pareto frontier.

| Piece | Where |
|---|---|
| Candidate harness (fixed) | `library/algorithms/evolved/` (`harness.py` = the API, `runner.py`) |
| Evaluation + scoring tool | `tools/evolve/evolve_bench.py`, `tools/evolve/frontier.py` |
| Settings (data, caps, box, guardrails) | `configs/evolve/bigann10m.yaml` |
| Baseline configs | `configs/evolve/baselines/*.yaml` |
| New baselines | `library/algorithms/starling/`, `library/algorithms/pageann/` (PageANN + LAANN) |
| Outputs | `results/evolve/` (`floors.json`, `frontier_bigann10m_q2k.json`, `candidates/<id>.json`) |

## Data

BIGANN (= ANN_SIFT1B) 10M prefix, uint8, L2; the first 2000 of the 10k public
queries (`/home/isfcr/data/bigann10m-q2k`). The 1M sanity stage uses the 1M prefix
with the first 1000 queries. `sift10m`/`sift1m` in `/home/isfcr/data` are the same
vectors stored as float32 (checked byte-for-byte); evolution uses the true uint8
bytes so page counts and DRAM are not inflated 4x.

Candidates see `/home/isfcr/evolve_data` (base + queries, hard links, **no ground
truth**). Ground truth is read only by the host-side scorer.

## A candidate program

One Python file (`evolve/disk_ann/initial_program.py` in the AI-Researcher repo is
the seed) defining `SEARCH_POINTS` (literal list, 1..8 operating points),
`build(ctx)` and `class Searcher` with `search(query, k, io)`. See the docstring of
`library/algorithms/evolved/algorithm/harness.py`. Images provide numpy, faiss,
numba and diskannpy.

## Measurement and integrity

Per search point (fresh container, page cache dropped by ann-suite):

- **recall@10**: recomputed on the host from the result ids the runner dumps
  (`<index>/results/<run_tag>_point_<i>.npz`).
- **pages/query**: counted by `QueryIO.read()` (4 KB O_DIRECT reads). The host
  rejects the point if the kernel's io.stat pages/query for the query window
  exceed `1.25 x harness + 2`, i.e. reads that bypassed the harness.
- **DRAM**: search-phase peak anonymous memory of the container minus the image
  floor (`floors.json`: a null program for the evolved image, idle Python for the
  C++ images). Cap: 640 MB + floor (0.5x the 1.28 GB raw data).
- Before the first query the runner fsyncs and evicts (`POSIX_FADV_DONTNEED`)
  every file under `/data`, `/tmp`, `/app`, ...; afterwards it rejects the point if
  `/dev/shm`, cgroup shmem or file-backed non-library mappings exceed 16 MB, or if
  result ids are out of range.
- **index size**: bytes under the index's `disk/` and `mem/`.
- CPU time per query is a guardrail only (Python; 200 ms/query).

Builds are not DRAM-capped (`build.memory_limit: none`, a per-phase override added
to ann-suite for this) and are cached by a hash of the program source without
`class Searcher` / `SEARCH_POINTS`, so search-only mutations skip the 10M build.

## Baselines and the frontier

`evolve_bench.py baselines configs/evolve/baselines/<system>.yaml` runs a system
and merges its points into `frontier_bigann10m_q2k.json`. Systems: SPANN (native
10M indices from `/home/isfcr/spann`), PipeANN (pipelined and beam search, PQ 10
and 32 B), DiskANN (diskannpy, uint8, PQ 10 and 34 B), Starling, PageANN and LAANN
(each at B = 0.1 and 0.32 GB, with and without page caching). Baseline pages/query
is the binaries' own O_DIRECT read count (`stats.io_reads`); SPANN's is io.stat for
the query window. All run under the same 785 MB cap, 8 pinned cores, 2000 queries.
`tools/evolve/run_baselines.sh` runs them in two lanes; the scored metrics are
per-container, so lanes may overlap (only QPS, which is not scored, is disturbed).

## Score

`tools/evolve/frontier.py`. Axes (all minimized): miss = 1 - recall, pages/query,
DRAM MB, index GB; each mapped to [0, 1] in log space over the box in
`configs/evolve/bigann10m.yaml` (recall >= 0.8, <= 2000 pages, <= 640 MB,
<= 40 GB).

- If the candidate's points add hypervolume to the baseline set:
  `combined_score = (HV(B u C) - HV(B)) / HV(B)` (> 0).
- Otherwise: `combined_score = -min_c max_b min_i (c_i - b_i)^+`, the uniform
  log-space improvement its best point still needs to escape domination (points
  outside the box are charged their distance to it).
- Failed sanity gate (best 1M recall < 0.5): `-5 - (1 - recall)`. Broken: `-10`.

MAP-Elites features reported for OpenEvolve: `dram_mb`, `rounds`.

## Running

```bash
uv run python tools/evolve/evolve_bench.py floors                 # once per image change
tools/evolve/run_baselines.sh                                      # hours; once
uv run python tools/evolve/evolve_bench.py candidate prog.py --id test1   # one candidate
```

`ANN_SUITE_SUDO_PASSWORD` (for cache drops) is read from
`~/.config/ann-suite/sudo_password` when unset.
