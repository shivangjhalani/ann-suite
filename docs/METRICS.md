# Metrics Reference

This document provides comprehensive documentation of all metrics collected by the ANN Benchmarking Suite, how they're measured, and the underlying collection architecture.

---

## Overview

The suite automatically collects **structured metrics** in five categories:

| Category | Priority | Source | Description |
|----------|----------|--------|-------------|
| **Disk I/O** | CRITICAL | cgroups v2 `io.stat` | IOPS and throughput for disk-based algorithms |
| **CPU** | HIGH | cgroups v2 `cpu.stat` / Docker stats | CPU time and utilization |
| **Memory** | HIGH | Docker stats API | Peak and average RSS |
| **Latency** | HIGH | Algorithm container output | Query latency percentiles |
| **Quality** | HIGH | Algorithm container output | Recall and QPS |

> [!NOTE]
> All metrics are collected automatically. The suite prefers **cgroups v2** when available for more accurate I/O metrics.

---

## Collector Architecture

The suite uses a **modular collector architecture** with three implementations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BaseCollector (Abstract)                        │
│  start(container_id) → stop() → CollectorResult                        │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│CgroupsV2      │ │DockerStats    │ │ResourceMonitor│
│Collector      │ │Collector      │ │(Legacy)       │
│               │ │               │ │               │
│ Source:       │ │ Source:       │ │ Source:       │
│ /sys/fs/cgroup│ │ Docker API    │ │ Docker API    │
│               │ │               │ │               │
│ IOPS: True    │ │ IOPS: Approx  │ │ IOPS: Approx  │
│ Accuracy: ★★★ │ │ Accuracy: ★★  │ │ Accuracy: ★★  │
└───────────────┘ └───────────────┘ └───────────────┘
    (Preferred)      (Fallback)         (Legacy)
```

### Collector Comparison

| Feature | CgroupsV2Collector | DockerStatsCollector | ResourceMonitor |
|---------|-------------------|---------------------|-----------------|
| **IOPS Source** | `rios`/`wios` from `io.stat` | Approximated from bytes | Approximated from bytes |
| **IOPS Accuracy** | True operations count | `bytes / block_size` | `bytes / block_size` |
| **CPU Time** | `usage_usec` from `cpu.stat` | Not available | Calculated from delta |
| **Memory** | `memory.current` | Docker stats | Docker stats |
| **Block I/O** | `rbytes`/`wbytes` from `io.stat` | `blkio_stats` | `blkio_stats` |
| **Overhead** | Low (filesystem reads) | Medium (API calls) | Medium (API calls) |
| **Availability** | Linux with cgroups v2 | Docker running | Docker running |

> [!IMPORTANT]
> **For accurate IOPS measurements**, ensure cgroups v2 is enabled. The `CgroupsV2Collector` reads actual I/O operation counts directly from the kernel, while Docker-based collectors approximate IOPS from byte counts.

---

## Structured Metrics Schemas

### CPUMetrics

```python
class CPUMetrics:
    cpu_time_total_seconds: float  # Total CPU time (user + system) from cgroups
    avg_cpu_percent: float         # Average CPU utilization %
    peak_cpu_percent: float        # Peak CPU utilization %
```

**How it's measured:**
- `cpu_time_total_seconds`: Read from cgroups v2 `cpu.stat` → `usage_usec` (converted from microseconds)
- `avg_cpu_percent`: `(cpu_time / wall_clock_time) × 100`
- `peak_cpu_percent`: Maximum CPU% across all sample intervals

---

### MemoryMetrics

```python
class MemoryMetrics:
    peak_rss_mb: float  # Peak RSS (Resident Set Size) in MB
    avg_rss_mb: float   # Average RSS in MB
```

**How it's measured:**
- Read from `memory.current` (cgroups) or `memory_stats.usage` (Docker)
- Samples filtered to exclude zeros (container stopped)
- Converted from bytes to megabytes: `bytes / (1024 × 1024)`

---

### DiskIOMetrics (CRITICAL)

```python
class DiskIOMetrics:
    # CRITICAL: IOPS metrics
    avg_read_iops: float          # Average read operations per second
    avg_write_iops: float         # Average write operations per second
    
    # CRITICAL: Throughput
    avg_read_throughput_mbps: float   # Read throughput in MB/s
    avg_write_throughput_mbps: float  # Write throughput in MB/s
    
    # HIGH: Page-level metrics (4KB pages)
    total_pages_read: int         # Total 4KB pages read
    total_pages_written: int      # Total 4KB pages written
    pages_per_query: float | None # Average pages read per query
```

**How IOPS is calculated:**

| Collector | Method |
|-----------|--------|
| **CgroupsV2** | `IOPS = Δ(rios or wios) / Δtime` — **True operation count** |
| **Docker Stats** | `IOPS ≈ Δbytes / block_size / Δtime` — Approximation |

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
    cpu_percent: float          # CPU utilization (Docker only)
    cpu_time_ns: int            # CPU time in nanoseconds (cgroups only)
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

### DockerStatsCollector (Approximate)

Estimates IOPS from byte deltas and block size:

```python
# Approximate IOPS from bytes
block_size = get_system_block_size()  # Typically 4096
read_ops_estimate = read_bytes_delta / block_size
avg_read_iops = read_ops_estimate / duration_seconds
```

---

## Block Size Detection

The system block size is used for IOPS approximation:

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
│   Resources                                                  │
│     Peak RAM                 35.4 MB                         │
│     Avg CPU                  150.0%                          │
│                                                              │
│   Disk I/O                                                   │
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

### Zero or Missing Metrics

| Symptom | Cause | Solution |
|---------|-------|----------|
| `Peak RAM: 0.0 MB` | Container exited too fast | Increase sample interval or check container logs |
| `Read IOPS: 0.0` | Index in overlay filesystem | Ensure index writes to `/data/index/` volume |
| `IOPS: N/A` | cgroups v2 not available | DockerStatsCollector will approximate; verify cgroups v2 setup |
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
