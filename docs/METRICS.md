# Metrics Reference

This document explains all metrics collected by the ANN Benchmarking Suite and how they're measured.

## Overview

The suite automatically collects **structured metrics** in five categories:

| Category | Priority | Source |
|----------|----------|--------|
| **Disk I/O** | CRITICAL | cgroups v2 `io.stat` |
| **CPU** | HIGH | cgroups v2 `cpu.stat` / Docker stats |
| **Memory** | HIGH | Docker stats API |
| **Latency** | HIGH | Algorithm container output |
| **Algorithm-Specific** | HIGH | Algorithm container output |

> **Note**: All metrics are collected automatically. The suite uses **cgroups v2** when available for more accurate I/O metrics.

---

## Structured Metrics Schemas

### CPUMetrics

```python
class CPUMetrics:
    cpu_time_total_seconds: float  # Total CPU time (user + system)
    avg_cpu_percent: float         # Average CPU utilization %
    peak_cpu_percent: float        # Peak CPU utilization %
```

### MemoryMetrics

```python
class MemoryMetrics:
    peak_rss_mb: float  # Peak RSS (Resident Set Size) in MB
    avg_rss_mb: float   # Average RSS in MB
```

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

### LatencyMetrics

```python
class LatencyMetrics:
    mean_ms: float  # Mean query latency
    p50_ms: float   # Median (50th percentile)
    p95_ms: float   # 95th percentile
    p99_ms: float   # 99th percentile (tail latency)
```

---

## Metric Collection

### CgroupsV2Collector (Preferred)

When available, the suite reads directly from cgroups v2 for accurate I/O:

```
/sys/fs/cgroup/system.slice/docker-{container_id}.scope/
├── io.stat      → rbytes, wbytes, rios, wios
├── cpu.stat     → usage_usec
└── memory.current
```

**Advantages**:
- Kernel-level accuracy
- Per-operation tracking (IOPS)
- Low overhead

### Docker Stats API (Fallback)

Falls back to Docker stats when cgroups v2 is unavailable:

```python
docker.containers.get(id).stats(stream=True)
```

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

1. **Volume Mounting**: Index must be written to `/data/index/` (host-mounted)
2. **cgroups v2**: Ensure cgroups v2 is enabled (default on modern Linux)
3. **Clear Page Cache** (optional):
   ```bash
   echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

---

## BenchmarkResult Schema

All metrics are aggregated into a structured `BenchmarkResult`:

```python
class BenchmarkResult:
    algorithm: str
    dataset: str
    timestamp: datetime
    
    # Structured metrics
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk_io: DiskIOMetrics          # CRITICAL
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
