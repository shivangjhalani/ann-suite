# Open-loop (arrival-rate) search mode

By default, every algorithm's search phase runs "closed-loop": the harness
(or, for `batch_mode`, the algorithm itself) issues the next query as soon as
the previous one completes. That measures throughput and per-query service
time, but not what users actually experience under load: when queries arrive
faster than the system can drain them, they queue, and closed-loop benchmarks
are blind to that queueing delay.

Open-loop mode issues queries on an **arrival schedule** instead - a fixed
worker pool serves a FIFO queue of queries that arrive per a Poisson process
(or all "arrive" at t=0, for a closed-loop-equivalent sanity check run
through the same code path) - and reports **latency = completion time -
scheduled arrival time**, so queueing delay is included in every percentile.

Semantics match the reference implementation, PipeANN's
`tests/search_openloop.cpp` (vendored as
`library/algorithms/pipeann/vendor/search_openloop.cpp`; see
[PIPEANN.md](./PIPEANN.md)):

- M/G/T FIFO queue: Poisson arrivals (or closed), a fixed pool of `T` worker
  threads, first-in-first-out service.
- `latency = completion − scheduled_arrival` (queueing included);
  `service_time = completion − dequeue` (pure per-query search time, no
  queueing) is also reported.
- The first `warmup_fraction` (default 10%) of issued queries are dropped
  from reported percentiles as steady-state warm-up.
- Timing happens in the algorithm's own process (ideally a C++ driver, for
  microsecond-latency accuracy), not in the Python harness - Python-side
  wall-clock timing at that resolution is unreliable.

## Configuration

```yaml
search:
  args:
    Ls: 50
    num_threads: 8
  arrival:
    mode: poisson            # or "closed"
    rate_qps: [500, 1000, 2000, 4000]   # a list sweeps -> one point per rate
    num_queries: 50000
    seed: 12345
    warmup_fraction: 0.1
    num_workers: 8            # defaults to search.args.num_threads
    raw_dump: false            # true -> per-query raw dump written to /results
```

`ArrivalConfig` lives on `SearchConfig.arrival`
(`src/ann_suite/core/schemas.py`). A list-valued `rate_qps` expands like any
other swept search arg: `search_sweep_params()` fans every search-args
combination out across every rate, each becoming its own benchmark point
(see `src/ann_suite/evaluator.py::ARRIVAL_RATE_SWEEP_KEY`).

## Results

When `search.arrival` is set, the algorithm container is expected to include
an `"open_loop"` object in its search output (alongside the ordinary `qps`/
`recall`/`mean_latency_ms`/etc. fields, populated with the same steady-state
numbers for backward compatibility with tooling that doesn't know about
open-loop mode). The evaluator reads it back onto
`BenchmarkResult.open_loop` (a loosely-typed passthrough dict, since fields
are algorithm/driver-dependent) and it round-trips through
`to_summary_dict()`/`results.json`:

```json
{
  "achieved_qps": 943.2,
  "arrival_rate_qps": 1000.0,
  "num_queries": 50000,
  "num_workers": 8,
  "warmup_fraction": 0.1,
  "latency_ms": {"mean": 0.81, "p50": 0.70, "p90": 1.50, "p99": 3.01, "p999": 8.82},
  "service_time_ms": {"mean": 0.65, "p99": 2.20},
  "ios_per_query": 41.2,
  "device_read_iops": 15342.0,
  "device_avg_read_latency_ms": 0.052,
  "device_util": 0.812,
  "machine_cpu_util": 0.734,
  "raw_dump_path": null
}
```

A healthy sweep looks like: `achieved_qps` tracking `arrival_rate_qps` almost
1:1 at low rates, latency percentiles roughly flat and close to the service
time, until some rate where the queue can no longer drain - at that point
`achieved_qps` plateaus (device/CPU utilization saturates) and latency
percentiles climb sharply (the "knee"). `device_read_iops` /
`device_avg_read_latency_ms` / `machine_cpu_util` are sourced from
`/sys/block/<dev>/stat` and `/proc/stat` deltas over the search window (see
`src/ann_suite/monitoring/base.py::read_device_io_totals`,
`read_machine_cpu_ticks`) - the same fix applied to the general
`disk_io.search_avg_read_service_time_ms` metric (previously always
0/None on kernels whose cgroup `io.stat` has no `rusec`/`wusec` field).

## Per-algorithm support

- **PipeANN**: native support via the vendored `search_openloop` C++ binary
  (preferred - microsecond-accurate timing). See
  [PIPEANN.md](./PIPEANN.md#known-limitations) for its one caveat (fixed
  internal RNG seed).
- **DiskANN**: best-effort Python-side fallback
  (`library/algorithms/diskann/algorithm/runner.py::run_openloop_search`),
  documented as such - see its docstring for caveats (scheduling jitter,
  unverified concurrent-search thread-safety).
- **SPANN / HNSW**: not wired up. `search.arrival` is accepted by the schema
  for any algorithm, but a container that doesn't understand the `"arrival"`
  key in its search config will simply run its ordinary closed-loop search
  (the extra key is ignored) - it will not produce an `"open_loop"` result
  and `BenchmarkResult.open_loop` will be `None`.
