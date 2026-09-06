# Docker Optimization Reference

This document details the specific Docker runtime configurations used by `ann-suite` to ensure "research-grade" performance and reproducibility. These settings mimic bare-metal performance while maintaining container isolation.

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
OS schedulers constantly move processes between cores to balance heat and load. This "migration" wipes CPU caches (L1/L2), causing massive performance implementations.
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
All these optimizations are applied automatically by the `ContainerRunner`. You do not need to manually configure them. They are baked into the Python runner logic to ensuring that `ann-suite run` is always a valid scientific measurement.
