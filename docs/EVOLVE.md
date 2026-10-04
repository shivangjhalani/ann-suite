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
| Settings (data, budget cells, guardrails) | `configs/evolve/bigann10m.yaml` |
| Baseline configs | `configs/evolve/baselines/*.yaml` |
| New baselines | `library/algorithms/starling/`, `library/algorithms/pageann/` (PageANN + LAANN) |
| Outputs | `results/evolve/` (`floors.json`, `frontier_bigann10m_q2k.json`, `candidates/<id>.json`) |

## Data

BIGANN (= ANN_SIFT1B) 10M prefix, uint8, L2; the first 2000 of the 10k public
queries (`/home/isfcr/data/bigann10m-q2k`). The 1M sanity stage uses the 1M prefix
with the first 1000 queries. Queries 2000-3999 (`bigann10m-hidden`, built by
`make_heldout.py bigann10m-hidden` with the published big-ann-benchmarks ground
truth, which equals our brute force on queries 0-1999) are used only to validate
would-be records. `sift10m`/`sift1m` in `/home/isfcr/data` are the same
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

Candidate code is untrusted (machine-written and selected for score), so it runs
sandboxed. The evolved container runs as root with no network
(`container_user: root`, `network: none` on the algorithm); the runner
(`algorithm/runner.py`) starts the candidate (`algorithm/sandbox.py`) as `nobody`
and gives it file descriptors, not paths, for the base vectors and its own index
directory. `/data` is 0700 on the host, so the candidate cannot open queries,
other datasets or other programs, cannot read the runner's memory (different uid,
ptrace scope 1), and cannot download published ground truth. The runner keeps
the queries and sends them one at a time over a pipe, the next only after the
previous answer. An adversarial probe program (reading queries, listing `/data`,
`/proc/<parent>`, TCP and DNS, writing `/data`) was blocked on every attempt in
both build and search.

Per search point (fresh container, page cache dropped by ann-suite):

- **recall@10**: recomputed on the host from the result ids the runner dumps
  (`<index>/results/<run_tag>_point_<i>.npz`).
- **pages/query**: counted by `QueryIO.read()` (4 KB O_DIRECT reads) inside the
  sandbox, so not trusted alone: the scored value is
  `max(harness, kernel - 1)`, the kernel count being the container's io.stat read
  bytes taken by the runner right before the first and after the last query (it
  matches the harness within 0.05 pages/query for honest programs; ann-suite's
  own figure, from 100 ms samples, caught the end of large index loads). The host
  also rejects the point if the kernel count exceeds `1.25 x harness + 1`.
- **rounds/query** (`io.read()` calls): more than 64 fails the point, so pages
  cannot be cut by reading one page per SSD round trip.
- **DRAM**: search-phase peak anonymous memory of the container minus the image
  floor (`floors.json`: a null program for the evolved image, now ~86 MB for the
  runner + sandbox processes; idle Python for the C++ images). Budget: 640 MB
  (0.5x the 1.28 GB raw data), enforced by the score (a point above it is in no
  budget cell). The container limit (1 GB) leaves headroom so library code pages
  are not evicted and re-read during queries: at 730 MB, IVFADC+R-pq32 (455 MB
  anon) showed ~7 extra kernel pages/query from such refaults, which the I/O
  cross-check mistook for bypass reads.
- **page cache**: `io.read()` is O_DIRECT, so the container's page cache must not
  grow by more than 16 MB during the queries; growth means index data read with
  ordinary reads and kept as uncounted DRAM.
- Before the first query the runner fsyncs and evicts (`POSIX_FADV_DONTNEED`)
  every file under `/data`, `/tmp`, `/app`, ...; afterwards it rejects the point if
  `/dev/shm`, cgroup shmem or file-backed mappings outside root-owned library
  trees exceed 16 MB, or if result ids are out of range.
- **index size**: bytes under the index's `disk/` and `mem/`; above 10.24 GB (8x
  the raw vectors) every point fails, so disk is not traded without limit for
  fewer reads (a larger index usually lowers pages/query).
- CPU time per query is a guardrail only (Python; 200 ms/query).

Builds are not DRAM-capped (`build.memory_limit: none`, a per-phase override added
to ann-suite for this) and are cached by a hash of the program source without
`class Searcher` / `SEARCH_POINTS`, so search-only mutations skip the 10M build.

## Baselines and the frontier

`evolve_bench.py baselines configs/evolve/baselines/<system>.yaml` runs a system
and merges its points into `frontier_bigann10m_q2k.json`. Systems: SPANN (native
10M indices from `/home/isfcr/spann`), PipeANN (pipelined and beam search, PQ 10
and 32 B), DiskANN (diskannpy, uint8, PQ 10 and 34 B), Starling, PageANN and LAANN
(each at B = 0.1 and 0.32 GB, with and without page caching; PageANN also in its
low-memory mode, `baselines/pageann_lowmem.yaml`: PQ inline on disk, 0.07 / 0.15 GB
hot-PQ budget, nav graph over 50k / 200k sampled pages; 0.07 GB is the minimum its
page-graph builder accepts at 10M). Configurations whose search exceeded the cap
were OOM-killed and are absent by design. Baseline pages/query
is the binaries' own O_DIRECT read count (`stats.io_reads`); SPANN's is io.stat for
the query window. All run under the same 785 MB cap, 8 pinned cores, 2000 queries.
`tools/evolve/run_baselines.sh` runs them in two lanes; the scored metrics are
per-container, so lanes may overlap (only QPS, which is not scored, is disturbed).

**Reference designs.** Classic designs that no packaged system covers are
implemented as harness programs in `tools/evolve/reference/` and added with
`evolve_bench.py add-reference <prog> --name N --system S`, measured exactly like
candidates. Without them the evolver is credited for rediscovering textbook
methods (the first OpenEvolve mutation reproduced IVFADC+R and scored +0.28 on
index size alone). Current references: IVF-Flat on disk (= the seed, plus a deeper
nprobe sweep, `ivf_flat_disk_wide.py`), IVFADC+R with 32 / 16 / 8 B PQ in DRAM
(Jegou et al. 2011), IVFADC+R with 16 B or 8 B PQ codes on SSD
(`ivfadc_rerank_ssd_pq16.py` / `_pq8.py`, ~10-20 MB of DRAM: the 32 MB cells; pq16
reaches recall 0.919 at 292 pages/query and dominates pq8), and IVFADC with a
refinement code (Jegou et al., ICASSP 2011): 32 B + 16 B (`ivfadc_refine_32_16.py`,
~550 MB: the 640 MB cells, 0.907 at 11.5 pages, 0.963 at 15.2) and 8 B + 2 B
(`ivfadc_refine_8_2.py`, ~115-120 MB: the 128 MB cells, 0.906 at 94 pages, 0.942
at 132), each also with an OPQ rotation (Ge et al. 2013; `ivfadc_opq_refine_*.py`,
added when evolved programs used OPQ: about 2% fewer pages on BIGANN at the same
recall). Sweeps tune nprobe and rerank depth separately: for in-DRAM codes the
pages depend on rerank depth, not nprobe, so a fixed ratio between them
understates the reference (audit of 2026-10-04). DiskANN
B0.1 is also swept deeper (`baselines/diskann_b01_wide.yaml`, Ls up to 1000) to
reach 0.95 inside 128 MB. Add a reference whenever a run reports a cell as
uncovered. References must be the published design, not a simplification: the
IVFADC+R references first coded raw vectors instead of residuals from the list
centroid, and the first evolution run scored +1.46 largely on that gap (+0.45
against the residual references; pq8 went from recall 0.89 at 278 pages/query to
0.945 at 212). The same happened with the refinement code: run "main2" reached
+1.11 in the 640 MB cells with 32 + 16 B two-level residual codes and a page
micro-cluster layout; against the refinement reference (recall 0.948 at 15.1
pages/query vs Starling-B0.32's 27.9 at 0.95) that is +0.23. Rule: each cell's
references must include the strongest textbook design that fits its budget.

Measured caveat: PipeANN's search-phase anonymous memory is ~490 MB with 10 B PQ
(100 MB of codes), independent of thread count; DiskANN with the same PQ uses
~100 MB. The excess is unexplained; check before citing PipeANN DRAM numbers.

## Score

Being replaced (2026-10-05): this score counts only SSD pages per query and ignores
latency and CPU. The replacement, scoring throughput and latency in the same cells,
is specified in [EVOLVE_SCORE.md](EVOLVE_SCORE.md).

`tools/evolve/frontier.py`, budget cells in the style of big-ann-benchmarks:
fixed budgets, fixed accuracy targets, comparison with the best known method
under the same budget. Cells are (DRAM budget T in {32, 128, 640} MB) x (recall@10
target R in {0.90, 0.95}).

- In each cell, opponent pages = fewest pages/query any baseline (published
  systems and reference designs) needs to reach R with DRAM <= T; candidate pages
  likewise from the candidate's points. Pages at exactly R are interpolated
  log-log in (miss, pages) along each side's (recall, pages) Pareto front, never
  extrapolated, so a sweep point placed just above R gains nothing.
- `gain = log2(opponent pages / candidate pages)`; `combined_score` = the best
  gain over covered cells, floored at -4 (16x more pages).
- A cell without any baseline is **uncovered**: it gives no credit and is listed
  in the report (`uncovered`) so a reference can be added. An empty region means
  nobody tried, not that it is hard; the pilot showed that open-ended frontier
  hypervolume rewards exactly such regions (IVFADC+R with codes on SSD scored
  +0.68 on DRAM alone).
- DRAM jitter (3-6 MB between identical runs) is absorbed by a 6 MB margin,
  charged to the candidate and credited to the baselines.
- No candidate point reaches 0.90 within 640 MB: `-4 - recall shortfall`. Failed
  sanity gate (best 1M recall < 0.5): `-5 - (1 - recall)`. Broken: `-10`.

**Validation.** A candidate whose score would beat the record (best validated
score so far, `results/evolve/record_<name>.json`, at least 0) is searched again
on the same queries and on the hidden queries (same build); its score becomes
the minimum of the three, so neither measurement luck nor fitting to the scored
queries sets a record. Delete the record file when starting a new run.

**Re-scoring after the frontier changes.** A report keeps every measured point,
including the validation stages, and the score is a pure function of those points
and the frontier. So when a reference is added mid-run, `evolve_bench.py rescore
<ids> --write-record --validate` re-scores stored candidates exactly, resets the
record to the best validated score among them, and validates a new leader that
was never validated (one more benchmark run per leader). The OpenEvolve side
(`/home/isfcr/shivang/evolve`, `disk_ann/rescore_checkpoint.py`) applies the new
scores to a checkpoint, so a run resumes instead of restarting. Reports written
before 2026-10-04 lack validation points and re-score as unvalidated.

MAP-Elites features reported for OpenEvolve: `dram_mb`, `rounds`.

## Running

```bash
uv run python tools/evolve/evolve_bench.py floors                 # once per image change
tools/evolve/run_baselines.sh                                      # hours; once
uv run python tools/evolve/evolve_bench.py candidate prog.py --id test1   # one candidate
uv run python tools/evolve/make_heldout.py bigann10m-hidden        # hidden queries, once
uv run python tools/evolve/evolve_bench.py rescore <ids> --write-record --validate
```

OpenEvolve runs from `/home/isfcr/shivang/evolve` on this host (README there).

`ANN_SUITE_SUDO_PASSWORD` (for cache drops) is read from
`~/.config/ann-suite/sudo_password` when unset.
