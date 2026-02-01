# Disk-Based ANN Metrics Upgrade TODO

## Goal
Make ann-suite produce diagnostic-grade disk I/O metrics and insight for disk-based ANN algorithms (DiskANN/SPANN-style), using cgroups v2 as the primary data source.

## Milestones
- [x] M1: Extend collector samples with I/O latency proxies, PSI, and memory fault stats.
- [ ] M2: Aggregate new metrics per phase with query-window time base and compute tail metrics.
- [x] M3: Expand output schema (JSON/CSV) and docs to include new metrics and interpretation.
- [x] M4: Add tests for parsing/aggregation of new metrics.

## Tasks

### Collector (cgroups v2) extensions
- [x] Parse additional io.stat fields when present (e.g., rusec, wusec) and store in samples.
- [x] Track per-device io.stat in a minimal form (top device by read bytes or small map).
- [x] Parse io.pressure (PSI) and store totals for some/full.
- [x] Parse memory.stat (pgmajfault, pgfault, file, file_mapped, active_file, inactive_file).
- [x] Parse cpu.stat throttling (nr_throttled, throttled_usec) when available.
- [ ] Add interval deltas collection for tail metrics (p95/max of read IOPS/MBps/service time).

### Aggregation + phase metrics
- [ ] Compute avg read/write size bytes/op and avg service time ms/op (when rusec/wusec available).
- [ ] Compute I/O stall percentages from PSI totals for query window and warmup.
- [ ] Compute major faults per query/sec, and cache size avg/peak during search.
- [ ] Add tail metrics (p95/max) for IOPS/MBps/service time from interval deltas.
- [ ] Preserve current behavior for missing metrics (default 0/None).

### Schema + output
- [x] Extend DiskIOMetrics with new fields (service time, bytes/op, PSI, tail metrics, per-device top).
- [x] Add MemoryMetrics or a new sub-structure for cache/major-fault stats.
- [x] Ensure to_summary_dict and to_flat_dict include new metrics and remain backward compatible.

### Documentation
- [x] Update METRICS.md with new fields, sources, formulas, and interpretation guidance.
- [x] Document kernel variability/optional fields and how missing data is handled.

### Tests
- [x] Add/extend unit tests for io.stat parsing with extra keys.
- [x] Add tests for PSI parsing and delta-to-percent conversion.
- [x] Add tests for memory.stat parsing and major-fault rates.
- [x] Add tests for tail metric computations.

## Remaining Work Summary

### What remains
- [ ] Complete M2 aggregation for new metrics (service time ms/op, PSI stall %, major faults per query/sec, cache stats, tail metrics from interval deltas).

### Why it remains
- Aggregation in `evaluator.py` currently has placeholders for PSI stall %, tail IOPS, and per-device summary because the aggregation pipeline does not yet consume the new collector fields and compute interval deltas.
- ResourceSummary does not yet carry new collector result fields through to the evaluator layer, so new metrics are not wired end-to-end.

### What is required to complete it
- Extend `ResourceSummary` to include: read/write usec totals, PSI deltas, pgmajfault/pgfault deltas, peak file cache bytes, throttling deltas, and top device summary.
- Map new `CollectorResult` fields into `ResourceSummary` in `container_runner.py` for build/search/warmup.
- Use those fields in `evaluator.py` to compute:
  - `search_avg_read_service_time_ms` / `search_avg_write_service_time_ms`
  - `search_io_stall_percent` (PSI) for query window and warmup
  - `search_major_faults_per_query`, `search_major_faults_per_second`, and cache size metrics
  - tail metrics (p95/max IOPS/MBps/service time) from per-interval deltas
- Add/update tests for ResourceSummary wiring and evaluator aggregation.

## Assigned Subagent Notes (to be filled after implementation)
- Collector: Implemented in base collector types and cgroups collector (extended io.stat parsing, per-device stats, PSI, memory.stat, CPU throttling, aggregated deltas). Updated __init__ exports. Files: src/ann_suite/monitoring/base.py, src/ann_suite/monitoring/cgroups_collector.py, src/ann_suite/monitoring/__init__.py.
- Aggregation + schema: Extended schemas (DiskIOMetrics, MemoryMetrics, LatencyMetrics) and evaluator aggregation placeholders for new metrics. Files: src/ann_suite/core/schemas.py, src/ann_suite/evaluator.py.
- Docs + tests: Updated METRICS.md with new metric definitions and added comprehensive tests for new fields/parsing in tests/test_cgroups_collector.py.
