# Docker Optimization Reference

This document details the specific Docker runtime configurations used by `ann-suite` to ensure "research-grade" performance and reproducibility. These settings mimic bare-metal performance while maintaining container isolation.

## Summary of Optimizations

| Feature | Setting | Purpose | Impact on Benchmark |
| :--- | :--- | :--- | :--- |
| **Networking** | `network_mode="host"` | Bypasses Docker's bridge network (NAT). | Eliminates network latency overhead (<1ms). Critical for high-throughput queries. |
| **Shared Memory** | `shm_size="2g"` | Increases `/dev/shm` from default 64MB. | Prevents crashes in libraries like FAISS/OMP that use heavy IPC/shared memory. |
| **Syscalls** | `seccomp=unconfined` | Disables syscall filtering. | Enables advanced I/O (e.g., `io_uring`, large `mmap`) used by state-of-the-art disk algorithms. |
| **CPU Pinning** | `cpuset_cpus` | Pins container to specific cores. | Prevents OS scheduler jitter and ensures fair comparison. |
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

### 4. CPU Affinity (`cpuset_cpus`)
OS schedulers constantly move processes between cores to balance heat and load. This "migration" wipes CPU caches (L1/L2), causing massive performance implementations.
*   **The Fix**: We allow users to specific `cpu_limit="0-3"`. Docker uses cgroups to essentially "wire" the container to those specific physical cores. The standard OS scheduler can no longer migrate the process to other sockets/cores, ensuring consistent cache hit rates.

## Reproducing These Results
All these optimizations are applied automatically by the `ContainerRunner`. You do not need to manually configure them. They are baked into the Python runner logic to ensuring that `ann-suite run` is always a valid scientific measurement.
