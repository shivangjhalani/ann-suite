# Metrics System Review Plan

## Scope
- Collectors: cgroups v2 and eBPF
- Orchestration and merge logic
- Post-processing and phase refinement

## Findings (Fact-Checked)
- eBPF metrics are dropped when query-window refinement runs.
  - Reason: eBPF merge occurs first, then `get_summary()` re-aggregates cgroups-only samples and rebuilds resources.
  - Impact: search metrics ignore eBPF for runs that report query timestamps.

- eBPF merge is partial, leaving tail metrics and per-device summaries inconsistent.
  - Reason: merge replaces totals/avg IOPS/latency totals only; tail metrics and per-device summaries remain from cgroups samples.
  - Impact: totals and tail metrics can diverge when eBPF captures I/O missed by cgroups.

- eBPF collector produces no samples.
  - Reason: `_maybe_snapshot_sample()` is a stub and never appends to `_samples`.
  - Impact: no eBPF time-series samples, no eBPF-derived tail metrics, sample_count stays cgroups-only.

- eBPF read/write classification is simplistic and can misclassify operations.
  - Reason: uses `req->cmd_flags & 1` to classify read/write, which does not cover all op types.
  - Impact: read/write counts and service-time totals can be skewed by non-read/write ops.

## Where eBPF Is Preferable
- Memory-mapped index access and page-fault driven reads.
  - Reason: cgroups `io.stat` can under-report mmap-driven disk activity; eBPF block-layer tracing captures it.

- High-variance or bursty I/O.
  - Reason: eBPF captures per-request events without sampling loss; cgroups sampling can miss spikes on short runs.

- Precise service-time distributions.
  - Reason: eBPF can emit per-op timing; cgroups provides only aggregate `rusec/wusec` totals.

## Cgroups Metrics That Must Be Collected via eBPF
- Block I/O bytes and ops (`io.stat` rbytes/wbytes/rios/wios) MUST be sourced from eBPF, not cgroups.
  - Reason: cgroups can miss mmap-driven disk activity; eBPF sees block-layer requests.
  - Implementation: derive bytes/ops from block-layer request events (e.g., tracepoints `block:block_rq_issue` and `block:block_rq_complete` or kprobes on `blk_account_io_start`/`blk_account_io_done`). Existing BCC tools like `biolatency`/`biosnoop` use these block-layer hooks and provide a reference pattern for counting ops and bytes per request.
  - Reliability/efficiency tradeoff: eBPF is more accurate but has higher overhead and permissions requirements.

- I/O service time totals (`io.stat` rusec/wusec) MUST be sourced from eBPF, not cgroups.
  - Reason: eBPF provides per-request timings and can build accurate distributions, not just totals.
  - Implementation: compute service time using request issue → completion deltas from the same block-layer hooks; this is the same approach described for `biolatency`/`biosnoop` (issue/complete latency).
  - Reliability/efficiency tradeoff: eBPF is more precise but needs sampling/aggregation to keep overhead controlled.

## Cgroups Metrics That Should Remain Cgroups-Based
- CPU time and throttling (`cpu.stat` usage_usec, nr_throttled, throttled_usec).
  - Reason: cgroups already provides exact accounting for the container; eBPF adds complexity without clear accuracy gains.

- Memory usage and cache stats (`memory.current`, `memory.stat`).
  - Reason: cgroups exposes the authoritative container memory accounting; eBPF cannot replace this cleanly.

- PSI I/O pressure (`io.pressure`).
  - Reason: PSI is kernel-provided and not derivable from eBPF events alone.

## Recommended Follow-Ups
- [x] Preserve eBPF I/O when refining to query windows, or re-aggregate eBPF using the same window.
- [x] Implement eBPF sample snapshots so tail metrics can be computed consistently.
- [x] Correct eBPF read/write op classification to handle non-READ/WRITE request types.

## Best Fix Strategy (Clean + Robust) — IMPLEMENTED

### ✅ Define a single source-of-truth for I/O after any time-window refinement
- Reason: avoids overwriting eBPF with cgroups when `query_start_timestamp`/`query_end_timestamp` are used.
- Implementation: Added `EBPFCollector.get_summary(start_dt, end_dt)` method. In `container_runner.py`, query-window refinement now re-aggregates both cgroups AND eBPF for the same window, then merges via `_merge_ebpf_io()`.

### ✅ Make I/O tail metrics consistent with the chosen source
- Reason: current merge updates totals but leaves p95/max derived from cgroups samples.
- Implementation: `_merge_ebpf_io()` now computes p95/max tail metrics from eBPF samples if available (>=2 samples); otherwise sets them to `None` for transparency. Tail metrics always match the I/O source.

### ✅ Implement periodic sampling inside EBPFCollector
- Reason: required for per-interval tail metrics and debug samples parity with cgroups.
- Implementation: Added `_collect_sample()`, `_maybe_snapshot_sample()`, initial sample in `start()`, final sample in `stop()`. Samples include monotonic_time for accurate interval computation.

### ✅ Fix eBPF read/write classification to handle more than a 1-bit op flag
- Reason: REQ_OP uses a wider mask; non-read/write ops can be misclassified.
- Implementation: BPF_PROGRAM_KPROBE now uses `REQ_OP_MASK (0xFF)` to extract op type, only emits events for `REQ_OP_READ (0)` and `REQ_OP_WRITE (1)`. Other ops (DISCARD, FLUSH, ZONE_*) are ignored.

## Refactor Assessment (Facts Only)
- The metrics system is modular (collectors, runner, evaluator) and readable.
- The issues found are localized to eBPF collection/merge paths and do not imply systemic messiness.
- A full refactor is not justified by the current findings; targeted fixes in collectors and merge logic are sufficient.
