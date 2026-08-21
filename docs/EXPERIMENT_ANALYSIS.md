# Experiment Analysis & Metrics Accuracy Report

**Date:** 2026-03-29
**Dataset:** sift-10k (10,000 base vectors, 1,000 queries, 128D, L2)
**Algorithms:** HNSW (hnswlib), DiskANN (diskannpy)
**Mode:** `batch_mode: false` (serial, per-query latency)
**Monitor interval:** 50ms

---

## 1. Experiments Run

### Experiment A: HNSW Parameter Sweep

Fixed build parameters (`M=16, ef_construction=200, num_threads=4`), sweeping `ef ∈ {10, 30, 50, 100, 200, 400}`.

| ef  | Recall | QPS      | Mean Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Build Time (s) |
|-----|--------|----------|--------------------|----------|----------|----------|-----------------|
| 10  | 0.8767 | 86,496.6 | 0.011              | 0.010    | 0.014    | 0.017    | 2.08            |
| 30  | 0.9834 | 35,020.1 | 0.027              | 0.025    | 0.040    | 0.066    | 1.98            |
| 50  | 0.9957 | 26,378.3 | 0.037              | 0.035    | 0.051    | 0.069    | 1.98            |
| 100 | 0.9997 | 16,750.6 | 0.059              | 0.058    | 0.071    | 0.083    | 1.96            |
| 200 | 0.9998 | 8,966.6  | 0.110              | 0.111    | 0.133    | 0.151    | 2.02            |
| 400 | 0.9999 | 4,999.9  | 0.198              | 0.197    | 0.245    | 0.308    | 1.94            |

### Experiment B: DiskANN Parameter Sweep

Fixed build parameters (`R=64, L=100, alpha=1.2`), sweeping `Ls ∈ {30, 50, 100, 200}` with `beam_width=2, num_threads=4, num_nodes_to_cache=0`.

| Ls  | Recall | QPS   | Mean Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Build Time (s) |
|-----|--------|-------|--------------------|----------|----------|----------|-----------------|
| 30  | 0.9989 | 695.5 | 1.436              | 1.437    | 1.709    | 1.890    | 5.64            |
| 50  | 0.9998 | 450.7 | 2.217              | 2.233    | 2.573    | 2.897    | 5.84            |
| 100 | 0.9998 | 239.9 | 4.166              | 4.190    | 4.758    | 5.129    | 5.87            |
| 200 | 0.9999 | 122.2 | 8.181              | 8.205    | 9.148    | 9.729    | 5.97            |

---

## 2. What's Working Correctly

### 2.1 Recall-QPS Tradeoff Curves

Both algorithms produce **monotonically correct** recall-vs-QPS tradeoffs:

- HNSW: Increasing `ef` from 10→400 improves recall from 0.877→0.9999 while QPS drops from 86.5k→5k. This is the expected behavior — `ef` controls the search beam width, and a wider beam finds more true neighbors at the cost of visiting more nodes.
- DiskANN: Increasing `Ls` from 30→200 improves recall from 0.999→0.9999 while QPS drops from 695→122. The diminishing recall returns at high Ls (already near 1.0 on this small dataset) are expected.

**Verdict:** Quality metrics are accurate and research-ready.

### 2.2 Latency Distributions

All runs show a consistent and plausible latency hierarchy:

- P50 < P95 < P99 in every run
- Tail ratios (P99/P50) range from 1.3x to 1.6x — normal for in-process ANN search
- HNSW latency scales approximately linearly with `ef` (0.011ms at ef=10 → 0.198ms at ef=400 ≈ 18x for 40x ef increase, sub-linear due to memory access patterns)
- DiskANN latency scales linearly with `Ls` (1.4ms at Ls=30 → 8.2ms at Ls=200 ≈ 5.7x for 6.7x Ls increase)

**Verdict:** Latency metrics from per-query (serial) mode are accurate and research-ready.

### 2.3 Warmup Phase Separation (DiskANN)

DiskANN warmup phase correctly captures index loading:

- `warmup_read_mb` ≈ 21–23 MB across all Ls values (index is ~19.4 MB on disk, so warmup reads the full index plus metadata/PQ tables)
- `warmup_duration_seconds` ≈ 0.19–0.21s (consistent index load time)
- `warmup.peak_rss_mb` ≈ 40 MB (index loaded into memory)

**Verdict:** Warmup/search phase separation works correctly for DiskANN.

### 2.4 DiskANN Block-Aligned Reads

DiskANN reports `avg_bytes_per_read_op = 4096` (exactly 4KB aligned reads), confirming that diskannpy uses proper block-aligned I/O for disk-based search. This is consistent with the DiskANN paper's design.

---

## 3. Issues Found

### 3.1 CRITICAL: Search Resource Metrics Use Full Container Window (Not Query Window)

**Affects:** HNSW and DiskANN resource metrics (CPU, Disk I/O)
**Root Cause:** A single design issue in the metrics pipeline

The `resources` object returned by `container_runner.run_phase()` covers the **entire container lifetime** (from container start to container exit), including:
- Python interpreter startup
- NumPy/library imports
- Dataset loading
- Index loading (warmup)
- Query execution (the actual benchmark)
- Container teardown

The warmup phase resources ARE correctly windowed — `container_runner.py` calls `self._cgroups_collector.get_summary(ws_dt, we_dt)` using warmup start/end timestamps. However, the main `resources` (used as `search_res` in the evaluator) is the **unwindowed full-container summary**.

**Evidence:**

```
HNSW ef=10:
  search.cpu_time_seconds = 0.046020  ← identical
  warmup.cpu_time_seconds = 0.046020  ← identical
  (Search took 0.012s wall time — can't have 0.046s CPU time)

HNSW ef=200:
  search.cpu_time_seconds = 0.050737
  warmup.cpu_time_seconds = 0.050693  ← nearly identical
  (Search took 0.112s wall time at 99.8% CPU — should be ~0.112s CPU time)
```

The search CPU time is reporting the same (or nearly same) values as warmup CPU time because both are looking at the full container cgroups window. The warmup window happens to be close to the full container window for these short runs.

**Impact:** All rate-based search resource metrics (CPU%, IOPS, throughput) are diluted by non-search time, and for short-running HNSW benchmarks, are essentially measuring container overhead rather than algorithm behavior.

**Downstream effects** of this single root cause:
- Issue 3.2 (ghost write I/O on HNSW)
- Issue 3.3 (CPU time leaking warmup into search)
- Issue 3.4 (DiskANN burst IOPS artifact)

---

### 3.2 HNSW Reports Write I/O During Search (Ghost Writes)

**Affects:** HNSW `search.disk_io.avg_write_iops`, `search.disk_io.total_pages_written`

| ef  | Write IOPS | Pages Written | Write MB |
|-----|------------|---------------|----------|
| 10  | 7,525.2    | 588           | 2.30     |
| 30  | 3,046.7    | 588           | 2.30     |
| 50  | 2,294.9    | 588           | 2.30     |
| 100 | 1,457.3    | 588           | 2.30     |
| 200 | 0.0        | 0             | 0.00     |
| 400 | 0.0        | 0             | 0.00     |

HNSW is a pure in-memory algorithm. Zero disk writes should occur during search. The constant 588 pages (2.30 MB) across ef=10–100 is **container overlay filesystem writes** during startup (Python imports, library loading, temp files). These writes happen before the query window but are included because the resource window covers the full container.

**Why ef=200/400 show zero:** For longer-running searches, the cgroups delta computation happened to not include the startup writes (timing-dependent artifact).

**This issue is resolved by fixing Issue 3.1** (query-window resource windowing).

---

### 3.3 HNSW `cpu_time_per_query_ms` Is Unreliable

**Affects:** HNSW `search.cpu_time_per_query_ms`

| ef  | cpu/query (ms) | mean_latency (ms) | Ratio (should be ~1.0x) |
|-----|----------------|--------------------|-------------------------|
| 10  | 0.046          | 0.011              | 4.36x                   |
| 30  | 0.073          | 0.027              | 2.66x                   |
| 50  | 0.083          | 0.037              | 2.26x                   |
| 100 | 0.085          | 0.059              | 1.45x                   |
| 200 | 0.051          | 0.110              | 0.46x                   |
| 400 | 0.151          | 0.198              | 0.76x                   |

For a single-threaded in-memory search, `cpu_time_per_query` should approximately equal `mean_latency_ms` (ratio ≈ 1.0). The ratios range from 0.46x to 4.36x, showing the metric is dominated by non-query CPU time (container startup, imports, index loading).

**This issue is resolved by fixing Issue 3.1.**

---

### 3.4 DiskANN p95/max Read IOPS Are Constant Across All Parameters

**Affects:** DiskANN `search.disk_io.p95_read_iops`, `search.disk_io.max_read_iops`

| Ls  | Avg Read IOPS | P95 Read IOPS | Max Read IOPS | Burst Ratio (max/avg) |
|-----|---------------|---------------|---------------|-----------------------|
| 30  | 797.0         | 29,004.6      | 29,013.3      | 36.4x                 |
| 50  | 523.2         | 29,638.3      | 29,921.0      | 57.2x                 |
| 100 | 297.2         | 28,638.8      | 31,169.1      | 104.9x                |
| 200 | 155.9         | 26,748.5      | 30,805.2      | 197.6x                |

The p95 and max IOPS are nearly identical (~29,000–31,000) regardless of `Ls`. This is because the p95/max values are computed from **per-interval samples that include the warmup phase** (index loading generates a large burst of sequential reads). The warmup burst dominates the tail statistics.

Average IOPS correctly decreases with increasing Ls (longer search time, same total I/O), but p95/max are meaningless.

**This issue is resolved by fixing Issue 3.1.**

---

### 3.5 Insufficient Cgroups Samples for HNSW

**Affects:** All HNSW resource metrics

| ef  | Search Duration (s) | Sample Count | Sampling Interval |
|-----|---------------------|--------------|--------------------|
| 10  | 0.012               | 2            | 50ms               |
| 30  | 0.029               | 2            | 50ms               |
| 50  | 0.038               | 2            | 50ms               |
| 100 | 0.060               | 2            | 50ms               |
| 200 | 0.112               | 2            | 50ms               |
| 400 | 0.200               | 4            | 50ms               |

With a 50ms polling interval, a 12ms search phase yields at most 0-1 samples within the actual query window. Even the full container window only captures 2 samples. Rate-based metrics (IOPS, CPU%, throughput) derived from 2 data points are statistically meaningless.

**Root cause:** sift-10k is too small for HNSW — the search completes faster than the monitoring can sample. This is partially a dataset issue and partially fixable with query repetition (see Suggestion S3).

---

### 3.6 `max_latency_ms` Is Always `None`

**Affects:** All algorithms, `latency.max_ms`

Neither `hnsw/algorithm/runner.py` nor `diskann/algorithm/runner.py` computes or emits `max_latency_ms` in their output JSON. The `LatencyMetrics` schema supports it, the `results.json` format reserves a slot for it, but the field is always `None`.

**Evidence:**
```
HNSW ef=10:   max_ms = None
DiskANN Ls=30: max_ms = None
(all 10 runs: max_ms = None)
```

---

### 3.7 DiskANN `pages_per_query` Does Not Reflect True Disk-Bound Behavior

**Affects:** DiskANN disk I/O metrics on sift-10k

| Ls  | Pages/Query | Total Read (MB) | Index Size (MB) | Warmup Read (MB) | Major Faults/Query |
|-----|-------------|-----------------|-----------------|-------------------|--------------------|
| 30  | 1.146       | 4.48            | 19.4            | 21.3              | 0.0                |
| 50  | 1.161       | 4.54            | 19.4            | 23.3              | 0.0                |
| 100 | 1.239       | 4.84            | 19.4            | 23.2              | 0.0                |
| 200 | 1.276       | 4.98            | 19.4            | 22.8              | 0.0                |

Pages/query is nearly flat (1.1–1.3) despite Ls varying by 6.7x. The explanation:

1. The warmup phase reads ~21–23 MB (the full index)
2. The index is only 19.4 MB — it fits entirely in OS page cache after warmup
3. `major_faults_per_query = 0.0` confirms zero actual disk reads during search
4. `file_cache_avg_mb ≈ 5.8 MB` shows file-backed pages are cached

The cgroups `io.stat` reports I/O operations that were **issued to the block device**, but since the page cache serves all reads, the reported I/O is residual metadata or readahead — not algorithm-driven disk access.

**This is not a code bug.** It's a fundamental limitation of benchmarking disk-based algorithms on datasets that fit in RAM. See Suggestion S4.

---

### 3.8 Per-Device I/O Summary Always Dropped

**Affects:** DiskANN `per_device_summary`

All DiskANN runs trigger the sanity check and drop per-device data:

```
Per-device read (127.1 MB) exceeds aggregate total (4.5 MB) by >2x.
Per-device read (787.5 MB) exceeds aggregate total (5.0 MB) by >2x.
```

The per-device counters from the cgroups collector are reporting **cumulative** values, not deltas relative to the search window. The aggregate totals (computed as deltas) are correct, but per-device values are raw cumulative reads since container start.

---

## 4. Suggested Fixes

### S1: Window Search Resources to Query Timestamps (Fixes 3.1, 3.2, 3.3, 3.4)

**Priority:** CRITICAL — resolves 4 issues with one change
**Files:** `src/ann_suite/runners/container_runner.py`, `src/ann_suite/core/schemas.py` (ContainerResult)

The warmup phase already has correct windowing:

```python
# container_runner.py (existing code, lines 646-661)
if warmup_start and warmup_end:
    warmup_res = self._cgroups_collector.get_summary(ws_dt, we_dt)
    warmup_resources_obj = self._build_resource_summary(warmup_res, ...)
```

Apply the same pattern for the query window:

1. After parsing the container output, extract `query_start_timestamp` and `query_end_timestamp`
2. Call `self._cgroups_collector.get_summary(qs_dt, qe_dt)` to get a query-windowed resource summary
3. Attach it to `ContainerResult` as a new field `query_resources: ResourceSummary | None`
4. In `evaluator._aggregate_results`, use `search_result.query_resources` (falling back to `search_result.resources` if unavailable) as `search_res` for all search-phase metric computation

**Result:** All search-phase resource metrics (CPU, disk I/O, IOPS, throughput) are scoped to the actual query execution window. Container startup, imports, index loading, and teardown are excluded.

---

### S2: Emit `max_latency_ms` From Algorithm Runners (Fixes 3.6)

**Priority:** LOW — one-line fix per runner
**Files:** `library/algorithms/hnsw/algorithm/runner.py`, `library/algorithms/diskann/algorithm/runner.py`

In the non-batch branch of each runner, after computing percentiles from `latencies_sorted`:

```python
max_latency = latencies_sorted[-1]
```

In the batch branch:

```python
max_latency = None  # Not measurable in batch mode
```

Add `"max_latency_ms": max_latency` to the return dict. The evaluator already reads this field:

```python
# evaluator.py line 849 (existing)
max_ms=search_output.get("max_latency_ms"),
```

No changes needed in the evaluator or schemas.

---

### S3: Add `query_rounds` to Ensure Sufficient Measurement Duration (Fixes 3.5)

**Priority:** MEDIUM — makes HNSW resource metrics reliable on small datasets
**Files:** `src/ann_suite/core/schemas.py` (SearchConfig), both algorithm runners

Add a `query_rounds: int = 1` field to `SearchConfig`. When `query_rounds > 1`:

1. The runner executes the full query set N times in a loop
2. **Latency and recall** are computed from **round 1 only** (avoids cache-warm bias)
3. **QPS** is computed as `total_queries_all_rounds / total_time_all_rounds`
4. **Resource metrics** benefit from the longer measurement window (more cgroups samples)

The runner changes are minimal — wrap the existing search call in a loop:

```python
# Pseudocode for runner change
all_round_latencies = []
query_start_timestamp = now()
start_time = perf_counter()
for round_idx in range(query_rounds):
    indices, distances, latencies = index.search(queries, k=k, ef=ef, ...)
    if round_idx == 0:
        first_round_indices = indices
        first_round_latencies = latencies
total_time = perf_counter() - start_time
query_end_timestamp = now()

# Use first round for latency/recall, total for QPS
qps = (n_queries * query_rounds) / total_time
recall = compute_recall(first_round_indices, ground_truth, k)
```

This is cleaner than lowering the monitor interval (which has OS overhead limits) or artificially inflating query counts.

---

### S4: Use Larger-Than-Cache Datasets for DiskANN Benchmarks (Fixes 3.7)

**Priority:** MEDIUM — experiment design, not code
**No code changes required**

sift-10k produces a 19.4 MB index that fits entirely in the OS page cache. To observe true disk-bound behavior with DiskANN:

**Option A: Use sift-128-euclidean (1M vectors).** The full dataset is already downloaded. The index will be ~2 GB, well beyond what fits in the 1 GB memory limit configured for DiskANN containers. Update the DiskANN config to use:

```yaml
datasets:
  - name: sift-128-euclidean
    base_path: sift-128-euclidean/base.npy
    query_path: sift-128-euclidean/queries.npy
    ground_truth_path: sift-128-euclidean/ground_truth.npy
    distance_metric: L2
    dimension: 128
```

**Option B: Restrict container memory.** Set `memory_limit: "30m"` to force the 19.4 MB index to be evicted from page cache during search. This is artificial but works for validating disk I/O metrics.

**Option C: Drop caches between runs.** Set `search.warmup.drop_caches_before: true` —
the suite now drops the OS page cache before each search phase automatically
(requires root or sudo; set the `ANN_SUITE_SUDO_PASSWORD` env var for passworded
sudo). Manual alternative:

```bash
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

> [!IMPORTANT]
> **Index reuse interacts with page cache warmth.** Since `build.reuse_index` defaults to
> `true`, only the first search sweep point follows a fresh index build; later points may
> see a warmer page cache from prior searches, slightly inflating QPS and deflating
> measured I/O across the sweep. For cold-cache comparisons either set
> `build.reuse_index: false`, use larger-than-cache datasets (Option A), or drop caches
> manually between runs.

---

### S5: Fix Per-Device I/O Delta Computation or Remove It (Fixes 3.8)

**Priority:** LOW — feature is currently non-functional
**Files:** `src/ann_suite/monitoring/cgroups_collector.py` (delta logic), or remove from `evaluator.py`

Two options:

**Option A (recommended): Remove `per_device_summary`.** It has never successfully populated in any run. Delete the computation block from `evaluator._aggregate_results` (lines ~732–759) and the `per_device_summary` field from `DiskIOMetrics`. Fewer lines, no dead code in results.

**Option B: Fix delta computation.** The collector needs to store per-device cumulative counters at the query window start timestamp and subtract them from the query window end values, the same way it handles aggregate I/O deltas. This requires changes to the cgroups collector's `get_summary()` method to track per-device baselines.

---

## 5. Summary

| Issue | Severity | Fix | Effort |
|-------|----------|-----|--------|
| 3.1 Search resources not windowed | CRITICAL | S1: Query-window `get_summary()` | Medium |
| 3.2 HNSW ghost write I/O | CRITICAL | Resolved by S1 | — |
| 3.3 HNSW CPU time leak | CRITICAL | Resolved by S1 | — |
| 3.4 DiskANN burst IOPS artifact | HIGH | Resolved by S1 | — |
| 3.5 Insufficient HNSW samples | MEDIUM | S3: `query_rounds` | Medium |
| 3.6 `max_ms` always None | LOW | S2: One-line runner fix | Trivial |
| 3.7 DiskANN page cache masking | MEDIUM | S4: Larger dataset | None (experiment design) |
| 3.8 Per-device summary broken | LOW | S5: Remove or fix deltas | Low |

**Key takeaway:** A single fix (S1: query-window resource windowing) resolves 4 of the 8 issues. The quality metrics (recall, QPS, latency percentiles) are already accurate and research-ready. The resource metrics (CPU, disk I/O) need the windowing fix before they can be trusted.
