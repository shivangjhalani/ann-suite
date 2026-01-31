# Metrics Reference

This document provides comprehensive documentation of all metrics collected by the ANN Benchmarking Suite, how they're measured, and the underlying collection architecture.

---

## Requirements

> [!IMPORTANT]
> **cgroups v2 is required** for running benchmarks. The suite will fail at startup if cgroups v2 is not available.

### Verifying cgroups v2

```bash
# Check if cgroups v2 is enabled (should show available controllers)
cat /sys/fs/cgroup/cgroup.controllers
# Expected output: cpuset cpu io memory hugetlb pids rdma misc
```

### Enabling cgroups v2

Most modern Linux distributions (Ubuntu 21.10+, Fedora 31+, Debian 11+, Arch Linux) use cgroups v2 by default.

For older systems or hybrid setups:

1. **Add kernel parameter** to GRUB:
   ```bash
   # Edit /etc/default/grub
   GRUB_CMDLINE_LINUX="systemd.unified_cgroup_hierarchy=1"

   # Update GRUB and reboot
   sudo update-grub
   sudo reboot
   ```

2. **Docker configuration**: Ensure Docker is configured to use cgroups v2:
   ```json
   // /etc/docker/daemon.json
   {
     "exec-opts": ["native.cgroupdriver=systemd"]
   }
   ```
   Then restart Docker: `sudo systemctl restart docker`

---

## Overview

The suite automatically collects **structured metrics** in five categories:

| Category | Priority | Source | Description |
|----------|----------|--------|-------------|
| **Disk I/O** | CRITICAL | cgroups v2 `io.stat` | IOPS and throughput for disk-based algorithms |
| **CPU** | HIGH | cgroups v2 `cpu.stat` | CPU time and utilization |
| **Memory** | HIGH | cgroups v2 `memory.current` | Peak and average RSS |
| **Latency** | HIGH | Algorithm container output | Query latency percentiles |
| **Quality** | HIGH | Algorithm container output | Recall and QPS |

> [!NOTE]
> All metrics are collected from cgroups v2, which provides the most accurate kernel-level metrics.

### Understanding "Warmup" Terminology

> [!IMPORTANT]
> **"Warmup" has two distinct meanings in this suite:**
>
> | Term | What it means | When it happens |
> |------|---------------|-----------------|
> | **Warmup Phase** | Time/resources to **load the index** from disk into memory | Always happens automatically before queries |
> | **Cache Warmup Queries** | Optional untimed queries to warm OS page cache | Only if `cache_warmup_queries > 0` in config |
>
> The `warmup` section in `results.json` reports the **warmup phase** (index loading), not cache warmup queries.
> Even with `cache_warmup_queries: 0`, the warmup phase metrics will show the index loading time/resources.

---

## Collector Architecture

The suite uses `CgroupsV2Collector` exclusively for metrics collection:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BaseCollector (Abstract)                        │
│  start(container_id) → stop() → CollectorResult                        │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │CgroupsV2      │
                  │Collector      │
                  │               │
                  │ Source:       │
                  │ /sys/fs/cgroup│
                  │               │
                  │ IOPS: True    │
                  │ Accuracy: ★★★ │
                  └───────────────┘
                    (Required)
```

### CgroupsV2Collector Features

| Feature | Details |
|---------|---------|
| **IOPS Source** | `rios`/`wios` from `io.stat` (true operation count) |
| **CPU Time** | `usage_usec` from `cpu.stat` |
| **Memory** | `memory.current` |
| **Block I/O** | `rbytes`/`wbytes` from `io.stat` |
| **Overhead** | Low (direct filesystem reads) |
| **Availability** | Linux with cgroups v2 (required) |

> [!NOTE]
> The suite reads actual I/O operation counts directly from the kernel, providing accurate IOPS measurements without approximation.

---

## Structured Metrics Schemas

### CPUMetrics

Metrics are separated by phase (build → warmup → search) for accurate analysis.

```python
class CPUMetrics:
    # Build phase
    build_cpu_time_seconds: float   # CPU time during build phase
    build_peak_cpu_percent: float   # Peak CPU utilization during build %

    # Warmup phase (index loading during search container startup)
    warmup_cpu_time_seconds: float    # CPU time during index warmup/load phase
    warmup_peak_cpu_percent: float    # Peak CPU utilization during warmup phase %

    # Search phase (primary focus for benchmarking)
    search_cpu_time_seconds: float  # CPU time during search phase
    search_avg_cpu_percent: float   # Average CPU utilization during search %
    search_peak_cpu_percent: float  # Peak CPU utilization during search phase %

    # Per-query metric (stable across different runs)
    search_cpu_time_per_query_ms: float  # CPU time per query (ms)
```

**How it's measured:**

| Metric | Source | Formula |
|--------|--------|---------|
| `*_cpu_time_seconds` | cgroups v2 `cpu.stat` | `usage_usec / 1_000_000` |
| `*_avg_cpu_percent` | Computed | `(cpu_time / wall_clock_time) × 100` |
| `*_peak_cpu_percent` | Per-interval max | `max(interval_cpu_time / interval_duration × 100)` |
| `search_cpu_time_per_query_ms` | Computed | `(search_cpu_time_seconds × 1000) / total_queries` |

> [!NOTE]
> All CPU metrics are sourced from cgroups v2. This ensures consistency with kernel-level accounting
> and avoids timing artifacts often seen in container runtimes.

> [!TIP]
> **`search_cpu_time_per_query_ms`** is the most stable metric for comparing algorithms across
> different hardware configurations, as it normalizes for query count and is unaffected by
> wall-clock variations.

---

### MemoryMetrics

Metrics are separated by phase (build → warmup → search) for accurate analysis.

```python
class MemoryMetrics:
    # Build phase
    build_peak_rss_mb: float   # Peak RSS during build phase in MB

    # Warmup phase (index loading during search container startup)
    warmup_peak_rss_mb: float    # Peak RSS during index warmup/load phase in MB

    # Search phase (query execution)
    search_peak_rss_mb: float  # Peak RSS during search phase in MB
    search_avg_rss_mb: float   # Average RSS during search phase in MB
```

**How it's measured:**
- Read from `memory.current` (cgroups v2)
- Samples filtered to exclude zeros (container stopped)
- Converted from bytes to megabytes: `bytes / (1024 × 1024)`

**Phase Boundaries:**
- **Build phase**: Entire build container lifetime
- **Warmup phase**: From container start to `warmup_end_timestamp` (if provided by algorithm)
- **Search phase**: From `query_start_timestamp` to `query_end_timestamp` (if provided)

> [!NOTE]
> If the algorithm container doesn't report timestamps, load and search phases are combined.
> For accurate phase-separated metrics, implement timestamp reporting in your algorithm runner.

---

### DiskIOMetrics (CRITICAL)

```python
class DiskIOMetrics:
    # Warmup phase I/O (index loading)
    warmup_read_mb: float         # Total MB read during warmup/index load phase
    warmup_write_mb: float        # Total MB written during warmup phase

    # CRITICAL: Search phase IOPS metrics
    search_avg_read_iops: float          # Average read operations per second
    search_avg_write_iops: float         # Average write operations per second

    # CRITICAL: Search phase Throughput
    search_avg_read_throughput_mbps: float   # Read throughput in MB/s
    search_avg_write_throughput_mbps: float  # Write throughput in MB/s

    # HIGH: Search phase page-level metrics (standardized 4KB pages)
    search_total_read_mb: float          # Total MB read during search phase
    search_total_pages_read: int         # Total 4KB pages read
    search_total_pages_written: int      # Total 4KB pages written
    search_pages_per_query: float | None # Average pages read per query

    # Metadata
    physical_block_size: int      # Detected physical block size of storage device
    sample_count: int             # Number of samples collected for I/O metrics
```

**How IOPS is calculated:**

| Collector | Method |
|-----------|--------|
| **CgroupsV2** | `IOPS = Δ(rios or wios) / Δtime` — **True operation count** |

> [!NOTE]
> **For in-memory algorithms (HNSW, etc.):** The `pages_per_query` metric includes I/O from
> index loading at the start of the search phase, not just query execution. For true in-memory
> algorithms, actual query execution has zero disk I/O, but the index must first be loaded
> from disk into RAM. If the index was recently built (same benchmark run), most of it may
> be in the Linux page cache, resulting in low but non-zero I/O values.

---

### LatencyMetrics

```python
class LatencyMetrics:
    mean_ms: float  # Mean query latency
    p50_ms: float   # Median (50th percentile)
    p95_ms: float   # 95th percentile
    p99_ms: float   # 99th percentile (tail latency)
```

**Source:** Reported by the algorithm container in its JSON output. The container measures these internally.

---

### Quality Metrics

```python
recall: float | None    # Recall@k (0.0 to 1.0)
qps: float | None       # Queries per second
```

**Source:** Reported by the algorithm container. Containers compute:
- `recall` = fraction of true k-nearest neighbors found
- `qps` = `total_queries / total_time_seconds`

---

## Raw Sample Collection

Each collector samples at a configurable interval (default: 100ms). Each sample contains:

```python
@dataclass
class CollectorSample:
    timestamp: datetime
    memory_usage_bytes: int     # Current RSS in bytes
    cpu_time_ns: int            # CPU time in nanoseconds
    blkio_read_bytes: int       # Cumulative bytes read
    blkio_write_bytes: int      # Cumulative bytes written
    blkio_read_ops: int         # Cumulative read operations (cgroups only)
    blkio_write_ops: int        # Cumulative write operations (cgroups only)
```

> [!TIP]
> Samples with `memory_usage_bytes == 0` are filtered out during aggregation, as they indicate the container has stopped.

---

## IOPS Calculation Methodology

### CgroupsV2Collector (Accurate)

Reads directly from `/sys/fs/cgroup/.../io.stat`:

```
8:0 rbytes=12345 wbytes=67890 rios=100 wios=50 ...
```

```python
# True IOPS from operation counts
read_ops_delta = last_sample.blkio_read_ops - first_sample.blkio_read_ops
avg_read_iops = read_ops_delta / duration_seconds
```

---

## Block Size Detection

The physical block size is detected for metadata reporting, but **IOPS is computed from true operation counts** (`rios`/`wios` from cgroups v2 `io.stat`), not approximated from block size. Page metrics use a **standardized 4KB page size** for cross-system comparability, regardless of physical block size.

```python
def get_system_block_size() -> int:
    """Detect from /sys/block/*/queue/physical_block_size"""
    # Checks: sda, nvme0n1, etc. (skips loop, ram, dm-*)
    # Falls back to 4096 bytes if detection fails
```

| Device Type | Typical Block Size |
|-------------|-------------------|
| HDD | 512 bytes |
| SSD/NVMe | 4096 bytes (4KB) |
| Default fallback | 4096 bytes |

---

## Configuration

### Sampling Interval

Configure in your benchmark YAML:

```yaml
monitor_interval_ms: 100  # Range: 50-1000ms
```

| Setting | Trade-off |
|---------|-----------|
| Lower (50ms) | More samples, higher overhead, better for short runs |
| Higher (1000ms) | Fewer samples, lower overhead, may miss fast containers |

---

## Console Output Example

```
╭────────────────────── HNSW on sift-10k ──────────────────────╮
│   Algorithm                  HNSW                            │
│   Dataset                    sift-10k                        │
│   Recall@k                   0.9997                          │
│   QPS                        19,594.7                        │
│   Build Time                 2.21s                           │
│                                                              │
│   Latency                                                    │
│     Mean                     0.050 ms                        │
│     P50                      0.049 ms                        │
│     P95                      0.061 ms                        │
│     P99                      0.087 ms                        │
│                                                              │
│   Build Phase Resources                                      │
│     Peak RAM                 128.5 MB                        │
│     CPU Time                 2.34 s                          │
│     Peak CPU                 380.0%                          │
│                                                              │
│   Search Phase Resources                                     │
│     Peak RAM                 35.4 MB                         │
│     Avg RAM                  32.1 MB                         │
│     CPU Time                 0.51 s                          │
│     Avg CPU                  150.0%                          │
│     Peak CPU                 200.0%                          │
│     CPU Time/Query           0.051 ms                        │
│                                                              │
│   Disk I/O (Search Phase)                                    │
│     Read IOPS                1500.0                          │
│     Write IOPS               50.0                            │
│     Pages Read               150,000                         │
│                                                              │
│   Hyperparameters                                            │
│     build.M                  16                              │
│     search.ef                100                             │
╰──────────────────────────────────────────────────────────────╯
```

---

## For Accurate Disk I/O Metrics

1. **Volume Mounting**: Index must be written to `/data/index/` (host-mounted volume)
2. **cgroups v2**: Ensure cgroups v2 is enabled (default on modern Linux)
3. **Clear Page Cache** (optional, for reproducibility):
   ```bash
   echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

---

## Time Bases

Rate metrics (IOPS, throughput, CPU%) require a time denominator. ANN Suite uses explicit time bases stored in `TimeBases`:

```python
class TimeBases:
    container_duration_seconds: float  # Total container wall time (start to exit)
    sample_span_seconds: float         # First to last resource sample
    warmup_duration_seconds: float | None   # Index loading time (if reported)
    query_duration_seconds: float | None  # Query execution time (if reported)
    query_start_timestamp: str | None     # ISO-8601 when queries started
    query_end_timestamp: str | None       # ISO-8601 when queries ended
```

| Time Base | Description | Use Case |
|-----------|-------------|----------|
| `container_duration_seconds` | Total container wall time | Operational cost, billing |
| `sample_span_seconds` | First to last resource sample | Diagnostic only |
| `warmup_duration_seconds` | Index loading/warmup time | Warmup cost analysis |
| `query_duration_seconds` | Query execution time only | **Primary for algorithm comparison** |

### Which Time Base is Used?

| Metric | Time Base | Notes |
|--------|-----------|-------|
| `search_avg_read_throughput_mbps` | `query_duration_seconds` (preferred) | Falls back to `duration_seconds` |
| `search_avg_write_throughput_mbps` | `query_duration_seconds` (preferred) | Falls back to `duration_seconds` |
| `search_avg_read_iops` | `query_duration_seconds` (preferred) | Computed from `total_read_ops / io_time_base` |
| `search_avg_write_iops` | `query_duration_seconds` (preferred) | Computed from `total_write_ops / io_time_base` |
| `*_avg_cpu_percent` | Sample span | Clamped to max CPUs × 100 |
| `search_cpu_time_per_query_ms` | N/A (absolute) | CPU time / num_queries |

> [!IMPORTANT]
> For accurate rate metrics, algorithm containers **should** report `query_start_timestamp` and
> `query_end_timestamp`. Without these, rates are computed over the entire container lifetime,
> which includes index loading overhead.

---

## BenchmarkResult Schema

All metrics are aggregated into a structured `BenchmarkResult`:

```python
class BenchmarkResult:
    run_id: UUID                    # Unique identifier for log correlation
    algorithm: str
    dataset: str
    timestamp: datetime

    # Structured metrics
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk_io: DiskIOMetrics          # CRITICAL for disk-based algorithms
    latency: LatencyMetrics

    # Quality
    recall: float | None
    qps: float | None

    # Build summary
    total_build_time_seconds: float | None
    index_size_bytes: int | None

    # Configuration
    hyperparameters: dict[str, Any]
```

Results can be flattened for CSV/DataFrame export via `to_flat_dict()`.

---

## Troubleshooting

### cgroups v2 Not Available

If you see the error:
```
RuntimeError: cgroups v2 is required for metrics collection but is not available.
```

1. **Check if cgroups v2 is mounted**:
   ```bash
   cat /sys/fs/cgroup/cgroup.controllers
   # Should show: cpuset cpu io memory hugetlb pids rdma misc
   ```

2. **Enable cgroups v2** (if not enabled):
   - See the [Requirements](#requirements) section above for detailed setup instructions

3. **Check permissions**:
   ```bash
   # Ensure you can read cgroup files
   ls -la /sys/fs/cgroup/
   ```

### Container cgroup Path Not Found

If you see the error:
```
RuntimeError: Could not find cgroup path for container <id>.
```

1. **Verify the container is running**:
   ```bash
   docker ps
   ```

2. **Check cgroup path exists**:
   ```bash
   # For container ID abc123...
   ls /sys/fs/cgroup/system.slice/docker-abc123*.scope/
   ```

3. **Verify Docker is using cgroups v2**:
   ```bash
   docker info | grep "Cgroup Driver"
   # Should show: Cgroup Driver: systemd
   ```

### Zero or Missing Metrics

| Symptom | Cause | Solution |
|---------|-------|----------|
| `Peak RAM: 0.0 MB` | Container exited too fast | Increase sample interval or check container logs |
| `Read IOPS: 0.0` | Index in overlay filesystem | Ensure index writes to `/data/index/` volume |
| All zeros | Container failed to start | Check Docker logs and container exit code |

### Verifying cgroups v2

```bash
# Check if cgroups v2 is enabled
cat /sys/fs/cgroup/cgroup.controllers

# Should show: cpuset cpu io memory hugetlb pids rdma misc
```

### Finding Container cgroup Path

```bash
# For container ID abc123...
ls /sys/fs/cgroup/system.slice/docker-abc123*.scope/
```
