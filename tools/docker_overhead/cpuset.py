"""Shared CPU-pinning helper for the Docker-overhead "native" arms.

Do NOT use `taskset -c <cpus>` to pin a native comparison process: `taskset`
sets a raw sched_setaffinity() mask, which restricts *where* threads may run
but does not put them in the isolated scheduling domain a cgroup cpuset does.
Docker's `--cpuset-cpus` is backed by the cpuset cgroup controller, which
also affects how the CFS load balancer treats the threads.

For workloads that run at or below the number of pinned cores (e.g. DiskANN
with num_threads == cpus), this difference doesn't show up. For workloads
that oversubscribe the pinned cores with more busy/runnable threads than
CPUs (e.g. PipeANN's io_uring SQPOLL mode: num_threads worker threads *plus*
one busy-polling kernel thread per worker), taskset-restricted threads load
balance far worse under contention than cgroup-cpuset-restricted ones -
measured on this host as a 6-7x throughput regression for PipeANN's
pipe_search mode at 8 threads on 8 pinned cores (304 QPS under taskset vs.
~2000 QPS under an equivalent cgroup cpuset, with Docker's own container
--cpuset-cpus giving the same ~2000 QPS). See
docs/DOCKER_OPTIMIZATIONS.md for the full writeup.

`native_cmd_prefix()` gives every native arm the same cgroup-cpuset mechanism
Docker uses, via a user-level systemd transient scope (no sudo required).
"""

from __future__ import annotations


def native_cmd_prefix(cpus: str) -> list[str]:
    """Command prefix that confines the wrapped command to `cpus` via a cgroup
    cpuset (systemd scope), the same mechanism Docker's --cpuset-cpus uses -
    unlike `taskset`, which only sets a raw affinity mask. See module docstring.
    """
    return ["systemd-run", "--scope", "--user", "-p", f"AllowedCPUs={cpus}", "--"]
