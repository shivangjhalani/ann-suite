# Evolve score v2: throughput and latency (proposal, 2026-10-05)

Status: in use since 2026-10-05 (`score.version: 2` in configs/evolve/bigann10m.yaml),
after the SSD calibration, the one-thread baseline re-measurement and the model check
below. Score v1 is described in [EVOLVE.md](EVOLVE.md#score); this document says why
it was replaced, what replaces it, and how it was measured and validated.

## Why the current score is not enough

Score v1 compares **SSD pages per query** with the best known method in budget cells
(DRAM 32 / 128 / 640 MB x recall@10 0.90 / 0.95). Pages per query is a throughput
proxy for an I/O-bound server and nothing else. It ignores:

- **Latency.** Every dependent I/O round costs a device round trip (~80-100 us on
  NVMe), and CPU time adds to it. A design may take more rounds to save pages.
- **CPU per query.** Saving pages by spending CPU is free under v1. The evolved
  leaders of run "main2" spend 6-38 ms of CPU per query (Python harness) where the
  IVFADC+R references written in the same harness spend ~1-4 ms, and the C++
  baselines 0.2-1.5 ms. In the 640 MB cells, where designs read 10-14 pages per
  query, a real server is CPU-bound, so a page saved there is worth little.

Papers on disk-resident ANN (DiskANN, SPANN, Starling, PipeANN, PageANN) and
big-ann-benchmarks judge methods by latency and throughput at a recall target and a
memory budget, not by page counts. v2 scores those two.

## Approaches rejected

- **Wall-clock latency of candidates.** Candidates are Python; `io.read` issues one
  synchronous `preadv` per page, where real systems issue a round's reads in
  parallel. Wall time would measure the interpreter and the harness.
- **Latency numbers already in the baseline frontier.** They are derived from
  throughput (latency = threads / QPS; p50 = p99), with different thread counts per
  system, and rounds are missing for SPANN, PipeANN and Starling.
- **A weighted sum** such as a*pages + b*latency: the weights would decide the
  winner.
- **Open Pareto / hypervolume scores:** the pilot run gamed hypervolume by occupying
  a corner no baseline was measured in. Budget cells exist to prevent this.
- **Pareto selection in OpenEvolve:** it has none. Fitness is one `combined_score`;
  trade-off diversity comes from the MAP-Elites feature grid. So: one scalar score,
  plus features.

## Definition

Every operating point x (one SEARCH_POINT of a candidate, or one search setting of a
known method) has recall@10 R(x), peak DRAM D(x) and SSD pages per query P(x) as
today, plus two costs.

**Latency** (ms per query, single thread):

    T(x) = sum over the query's I/O rounds r of d(p_r)  +  c(x)

- `d(p)`: time for the SSD to complete a batch of p random 4 KB reads issued
  together (queue depth p), measured once on the benchmark host with fio
  (`results/evolve/ssd_model.json`; interpolated in p).
- `p_r`: distinct pages read in round r (the harness counts these exactly).
- `c(x)`: CPU time per query on one thread, excluding the time spent inside
  `io.read` (the device time is charged through `d`). numba is available to
  candidates, so loop-heavy code (graph traversal) runs at native speed rather than
  being penalised by the interpreter.
- Averaged over queries (mean latency). p99 is a later option.

**Throughput** (queries per second on the benchmark host):

    Q(x) = min( IOPS_max / P(x),  N_cores / c(x) )

- `IOPS_max`: the SSD's saturated random 4 KB read rate (fio, many jobs, deep queue).
- `N_cores` = 8: the performance cores (0-7 on isfcr) every search is pinned to,
  so c is measured on the cores it is scaled by. The 16 slower efficiency cores are
  left out (the user's choice, 2026-10-05).
- Whichever resource runs out first limits the server.

**Known methods.** For the C++ baselines (DiskANN, PipeANN, SPANN, Starling,
PageANN, LAANN), T is **measured**: their search is re-run with one search thread
on a cold cache, which is the real latency of their real asynchronous I/O and native
code. Their c is the search CPU per query measured from the container cgroup (already
recorded). Reference designs in the harness (IVF-Flat, IVFADC+R family) are
measured exactly like candidates.

**Cells and gain.** Cells are unchanged: DRAM tiers 32 / 128 / 640 MB (with the 6 MB
margin) x recall@10 targets 0.90 / 0.95. In a cell, each method's points that fit the
DRAM tier are interpolated to the target recall as today (log-log in miss rate
between the two adjacent points of its front); Q and T of an interpolated point come
from the same pair of points, so one operating point carries both. Then

    Q*  = best (highest) Q of any known method in the cell
    T*  = best (lowest)  T of any known method in the cell (possibly another method)
    gain(cell) = 1/2 * [ log2(Q_cand / Q*) + log2(T* / T_cand) ]

maximised over the candidate's operating points in the cell, and

    combined_score = max over cells that have a known method of gain(cell)

As in v1, cells without a known method give no credit, and a would-be record is
re-measured on the same and on hidden queries; the minimum counts.

**Reading the score.** +1 means that one design is, on the geometric average, 2x
beyond the best known throughput and the best known latency at the same DRAM budget
and recall, where those two bests may come from different methods. Being 2x better on
one and 2x worse on the other scores 0. The mean of logs needs no weights and does not
depend on units.

**Why it is hard to game.**
- More pages cost throughput (IOPS) and latency (`d` grows with p).
- One huge round costs device time in `d(p)`.
- Scanning everything in DRAM costs CPU in both T and Q.
- Many small rounds cost a round trip each.
- Empty corners give nothing (cells).

**MAP-Elites features:** `dram_mb` and log10 T of the best cell's operating point
(replacing raw `rounds`), so low-latency and low-memory designs keep their own niches.
The report carries T as `features.latency_ms`; the OpenEvolve evaluator takes its log10
(`log_latency`), because OpenEvolve bins a feature linearly between the smallest and
largest value seen.

## Known biases

- Candidates still pay Python per-call overhead in c, and the harness reads pages
  synchronously (only the model, not the score, sees them as parallel). Both
  understate candidates, so a reported gain is conservative.
- The device model has no software I/O overhead, which slightly flatters
  candidates. Validation below bounds this.
- c of the C++ baselines comes from the cgroup and includes I/O submission CPU; the
  candidates' c excludes the harness's read calls. Small next to d at one thread.

## Validation before use

1. **Model vs real latency.** For baselines whose rounds are known (DiskANN: hops x
   beam width; PageANN, LAANN: hops), the modelled T (from their pages per round and
   measured c) must agree with their measured single-thread latency within 25%. If
   not, the model (or `d`) is revised before any score is used.
2. **Stability.** Re-running a candidate must change the score by less than 0.02
   (v1: ~0.003); c is noisier than page counts.
3. **Spot checks.** Re-scoring the stored programs of run "main2" must rank them
   plausibly: a design that saves pages by scanning far more codes in DRAM must not
   rise.

## Measurements and validation results (2026-10-05)

- **SSD model** (`results/evolve/ssd_model.json`, fio on an existing 20 GB index
  file, median of 3): one round of p pages takes 93 us (p = 1), 116 us (4), 188 us
  (16), 396 us (64), 1.08 ms (256), 3.67 ms (1024); saturated rate 284k reads/s
  (8 jobs x queue depth 128). Monotonic, and the three runs agree within 1%. A first
  calibration on a freshly written file was erratic (the QLC drive folding its SLC
  cache) and was discarded.
- **Baselines at one search thread** on their existing indexes, after the SSD had
  been idle 10 minutes: all 15 baseline indexes (DiskANN, PageANN, LAANN,
  PipeANN, SPANN, Starling) and the 11 harness references. Points that exceeded the
  785 MB search cap (PageANN-B0.32, and the 10% page-cache settings) were killed
  as in the 8-thread campaign, so v1 and v2 cover the same methods. A first attempt
  rebuilt the indexes before searching (ann-suite reuses an index only within one
  run); `baselines --search-threads` now uses the existing index and refuses to build.
- **Model vs measured latency** (`check_latency_model.py`): modelled / measured
  median ratio 1.14 for DiskANN (17 points), 1.14 for PageANN (20), 1.15 for LAANN
  (16); worst point 1.44. Within the 25% tolerance. The model overestimates the
  real systems' latency, which favours the known methods over candidates.
- **Old reports' CPU:** the 85 reference points re-measured with runner-served reads
  spend 0.8-1.4 us less CPU per page than before (the read syscall), so
  `legacy_read_cpu_ms_per_page: 0.001`. Negligible next to candidates' CPU.

## Implementation plan

1. `tools/evolve/calibrate_ssd.py`: fio `randread`, 4 KB, O_DIRECT on the index
   device; batch latency for p = 1, 2, 4, ..., 1024 (submit p, wait for all p) and
   saturated IOPS. Writes `results/evolve/ssd_model.json`.
2. Harness (`library/algorithms/evolved`): the runner performs the reads. The
   candidate cannot open its disk files and `io.read()` sends one round's page ids
   over the pipe, so rounds and pages per round are counted by trusted code (once
   rounds cost latency, a self-reported count would be worth faking), and the
   candidate's CPU time excludes the reads. Per point the runner reports a
   histogram {pages in a round: rounds}; evolve_bench turns it into T with `d`.
3. Baselines: `evolve_bench.py baselines <config> --search-threads 1` re-runs each
   frontier search setting with one search thread on a cold cache, recording
   measured mean latency and cgroup CPU per query. The indexes are reused, so this
   is search time only (a few hours, exclusive use of the host). The harness
   reference designs are re-measured with `add-reference` (cached builds).
4. `tools/evolve/frontier.py`: v2 cells and gain; tests for interpolation, the
   two-objective gain, and the gaming cases above.
5. OpenEvolve evaluator: new features; the budget_cells artifact shows Q, T and c
   next to the opponent's.
6. Re-score stored candidates. Their reports hold mean rounds, pages and CPU per
   point (CPU includes the old per-page read calls, a small overcharge), so T is
   estimated with equal pages per round; new evaluations use exact per-round counts.
   Then re-score the main2 checkpoint (`rescore_checkpoint.py`) and resume.

## Not scored

Index size (cap 10 GB) and build time (about an hour at 10M) stay as limits. Updates
and deletions are not measured. Generalisation is checked by hidden queries and the
held-out DEEP and T2I sets, as in v1.
