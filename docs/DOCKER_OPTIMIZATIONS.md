# Docker Optimization Reference

This document details the Docker runtime configurations `ann-suite` uses so containerized runs match bare-metal performance.

## Measured Overhead (DiskANN, BIGANN-10M, 2026-09-29)

`tools/docker_overhead/run.py` compares three arms on one prebuilt index (DiskANN C++
R=100/L=100, commit 78256bb), the same 10k queries, P-cores 0-7, W=2, no node cache,
and a dropped page cache before every point, over Ls in {10..200} x {1, 8} threads,
3 interleaved repeats:

| Arm | What it is |
| :-- | :-- |
| suite | `ann-suite run` (container, cgroup monitoring on) |
| native | the same runner + same diskannpy build, run directly on the host |
| cpp | DiskANN's C++ `search_disk_index` |

Result: recall and I/Os per query are identical across arms at every point. QPS
suite/native geo-mean is **1.008** (range 0.98-1.06) and cpp/native **1.016**
(0.98-1.12), both inside the repeat-to-repeat spread (median 3-4%, max 18%). Docker
and ann-suite's monitoring add no measurable search cost, and the Python runner in
batch mode matches the C++ driver. Raw data: `results/docker_overhead/2026-09-29_15-50-06/`
on the isfcr host. Re-run after changing hardware, kernel, or the runner.

Extended with `tools/docker_overhead/run_diskann_extra.py` (same index/setup): serial
per-query latency percentiles, Poisson open-loop search, and search under a memory cap
(Docker `--memory` vs. an equivalent cgroup limit natively). All three also land at
ratio ~1.0 (0.96-1.06). Note: DiskANN reads via O_DIRECT, so the memory-cap test mostly
exercises cgroup-limit *parity*, not real cache-pressure behavior (nothing to evict).

## Native-arm pinning bug: `taskset` vs. cgroup cpuset (found via PipeANN)

`tools/docker_overhead/run_pipeann_spann.py` runs the same check for PipeANN
(BIGANN-10M, R=64/L=128/pq=32, pipe_search/SQPOLL mode) and SPANN. The first PipeANN
run showed suite (Docker) **2.7-3.7x faster** than native at every Ls and repeat -
the opposite of overhead, and large enough to be a methodology bug, not a real effect.

**Cause:** every native arm pinned CPUs with `taskset -c <cpus>` (a raw
`sched_setaffinity` mask), while Docker's `--cpuset-cpus` is backed by the cpuset
*cgroup* controller - a different kernel mechanism that also changes how the CFS load
balancer treats the pinned threads. DiskANN's search (num_threads == pinned cores)
never exposed this. PipeANN's pipe_search mode does: with SQPOLL, each of the 8
worker threads gets its own busy-polling kernel thread, so 16 runnable threads
contend for 8 cores - exactly the case where cgroup-cpuset load balancing and raw
affinity-mask balancing diverge. `/proc/<pid>/task/*/stat` during a native run showed
all 16 threads persistently `R` (running/runnable); under Docker's cpuset several
polling threads properly sat `S` (idle) between bursts.

**Confirmed on the live binary** (same index, same config, Ls=100, 8 threads):

| Pinning mechanism | QPS |
| :-- | :-- |
| `taskset -c 0-7` | 304 |
| `systemd-run --scope -p AllowedCPUs=0-7` (cgroup cpuset) | ~2000 |
| Docker `--cpuset-cpus 0-7`, same moment | ~2000 |

memlock ulimit (Docker's container default is 8 MB vs. the host shell's ~7.8 GB) was
also checked and ruled out - forcing native down to 8 MB changed nothing.

**Fix:** `tools/docker_overhead/cpuset.py` gives every native arm a
`systemd-run --scope --user -p AllowedCPUs=<cpus>` prefix instead of `taskset`,
matching Docker's actual mechanism. Re-running PipeANN's full 3-repeat sweep with the
fix: closed-loop ratio 0.96-1.24 (4 Ls values), open-loop 1.00 (3 rates) - both back
in the normal noise band. **Takeaway: always pin a native comparison process with a
cgroup cpuset, not `taskset`, whenever it can spawn more runnable threads than pinned
cores** (io_uring SQPOLL, thread pools sized independently of a `num_threads` config,
etc.) - `taskset` alone will understate that arm's real throughput.

## Summary of Optimizations

| Feature | Setting | Purpose | Impact on Benchmark |
| :--- | :--- | :--- | :--- |
| **Networking** | `network_mode="host"` | Bypasses Docker's bridge network (NAT). | Eliminates network latency overhead (<1ms). Critical for high-throughput queries. |
| **Shared Memory** | `shm_size="2g"` | Increases `/dev/shm` from default 64MB. | Prevents crashes in libraries like FAISS/OMP that use heavy IPC/shared memory. |
| **Syscalls** | `seccomp=unconfined` | Disables syscall filtering. | Enables advanced I/O (e.g., `io_uring`, large `mmap`) used by state-of-the-art disk algorithms. |
| **CPU Affinity** | `cpuset_cpus` | Restricts the container to specific cores. | Controls placement; it does not cap aggregate CPU time. |
| **NUMA Pinning** | `cpuset_mems` | Pins memory placement to the same node(s) as the pinned CPUs. | Removes cross-NUMA traffic noise; CPU and page cache stay local. |
| **CPU Limit** | `nano_cpus` | Hard cap on aggregate CPU usage (CFS quota). | ⚠️ CFS throttling can inflate p95/p99 latency; prefer affinity. |
| **Memory Limit** | `mem_limit` | Hard cap on container RAM. | Enforces strict resource constraints; prevents swap thrashing during large builds. |

---

## Detailed Explanations

### 1. Host Networking (`network_mode="host"`)
By default, Docker uses a "bridge" network which creates a virtual ethernet adapter and uses NAT (Network Address Translation) to route traffic. While secure, this introduces a measurable CPU and latency overhead for every packet.
*   **Without Optimization**: Queries must pass through the kernel's NAT table, adding microseconds of latency per query.
*   **With Optimization**: The container shares the host's network stack directly. `localhost` inside the container is `localhost` on the host. Performance is effectively identical to a bare-metal process.

### 2. Large Shared Memory (`shm_size="2g"`)
Many high-performance numerical libraries (like Intel MKL, OpenBLAS, and FAISS) utilize shared memory for inter-process communication (IPC) or temporary storage during parallel operations.
*   **The Issue**: Docker defaults `/dev/shm` to 64MB.
*   **The Fix**: We explicitly raise this to 2GB. This ensures that large-scale index builds or highly parallel searches do not crash with `Bus error` or `SIGSEGV` due to running out of shared memory segments.

### 3. Unconfined Seccomp Profile (`security_opt=["seccomp=unconfined"]`)
`seccomp` (Secure Computing mode) is a Linux kernel feature used by Docker to filter which system calls a container can make.
*   **The Issue**: The default Docker profile blocks many strictly "safe" but "uncommon" syscalls. Modern high-performance disk I/O libraries (like `liburing` for async I/O) often rely on newer syscalls that might be blocked.
*   **The Fix**: Setting `seccomp=unconfined` allows the algorithm to use the full range of Linux kernel system calls. This is essential for disk-based algorithms (like DiskANN) that need to squeeze every ounce of IOPS from an NVMe drive.

### 4. CPU Affinity (`cpuset_cpus`) + NUMA Memory Pinning (`cpuset_mems`)
OS schedulers constantly move processes between cores to balance heat and load. This "migration" wipes CPU caches (L1/L2) and adds noise. On hybrid CPUs (e.g. the isfcr host's Core Ultra 9 285K: P-cores 0-7, E-cores 8-23) it also mixes core types, so pin to one core type.
*   **The Setting**: `cpu_affinity="0-3"` restricts placement to those logical CPUs.
*   **NUMA**: On multi-socket / multi-NUMA-node hosts, restricting CPUs alone is not enough. The kernel may still place the container's memory (page cache, heap) on a *different* node than the pinned CPUs, forcing every access across the NUMA interconnect — a real, measurable penalty for both in-memory (HNSW) and disk-based (DiskANN) workloads.
*   **The Optimization**: When `cpu_affinity` is set, `ann-suite` reads the host NUMA topology (`/sys/devices/system/node/node*/cpulist`) and sets `cpuset_mems` to the node(s) the affinity cores belong to. CPU and memory are therefore pinned together, so all allocations stay local to the working cores. This is applied automatically; no extra config is required.
*   **Best practice**: Pin all cores of a single NUMA node (e.g., `"0-15"` on a 2-socket host) for deterministic, low-noise results. Avoid spanning two nodes unless you specifically want to benchmark cross-node traffic. On single-node / non-NUMA systems this is a no-op and safe.

### 5. CPU Limit (`nano_cpus`) — CFS Throttling Caveat
`cpu_limit` (cores) maps to Docker's `nano_cpus`, which the CFS scheduler enforces as a CPU quota.
*   **The Caveat**: CFS caps CPU in ~100ms accounting windows. A busy workload periodically hits the quota and is **throttled** (paused) until the next window, injecting small bursts of stalls.
*   **Impact on Benchmarks**: For steady-state query workloads, throttling can inflate **p95/p99 tail latency** and add noise to QPS. It does **not** affect recall.
*   **Recommendation**: For latency-sensitive ANN research, prefer `cpu_affinity` (NUMA-pinned cores, no throttling) over `cpu_limit`. If you must cap usage (e.g., to mimic a target core budget), combine affinity + limit and watch the `CPUThrottlingMetrics` (`nr_throttled` / `throttled_percent`) in your results to quantify the noise. When no limit is configured, throttling counters are always 0.

## Reproducing These Results
These settings are applied automatically by the `ContainerRunner`; no config is needed.

To re-measure overhead on a host (needs the DiskANN image and a prebuilt index; see the
constants at the top of `run.py`):

```bash
tools/docker_overhead/setup_native_diskannpy.sh     # same diskannpy, built on the host
ANN_SUITE_SUDO_PASSWORD=... uv run python tools/docker_overhead/run.py --repeats 3
```
