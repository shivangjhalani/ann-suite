"""eBPF collector for high-precision block I/O tracing.

This collector uses eBPF (via BCC) to trace block layer I/O request issue and completion
events (block_rq_issue, block_rq_complete). It provides:
1.  Accurate I/O metrics for memory-mapped files (which cgroups v2 io.stat misses).
2.  High-resolution service time / latency histograms (future).
3.  Filtering by Cgroup ID to attribute I/O to specific containers.

Requires:
- Root privileges or CAP_SYS_ADMIN.
- Kernel headers.
- BCC installed.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from bcc import BPF
except ImportError as e:
    import logging

    logging.getLogger(__name__).error(f"Failed to import BPF from bcc: {e}")
    BPF = None
except Exception as e:
    import logging

    logging.getLogger(__name__).error(f"Unexpected error importing BPF from bcc: {e}")
    BPF = None

from ann_suite.monitoring.base import BaseCollector, CollectorResult, CollectorSample

logger = logging.getLogger(__name__)


class EBPFCollector(BaseCollector):
    """Collector that uses eBPF to trace block I/O metrics.

    Hooks into kernel tracepoints:
    - block:block_rq_issue
    - block:block_rq_complete

    Matches requests by pointer/dev/sector to calculate accurate service time (latency)
    and volume (bytes). Filters events by the container's cgroup ID (inode).
    """

    # BPF Program to trace block I/O
    # We filter by cgroup_id in kernel space for efficiency.
    BPF_PROGRAM = r"""
    #include <uapi/linux/ptrace.h>
    #include <linux/blkdev.h>
    #include <linux/blk-mq.h>

    // Key to track in-flight requests
    struct key_t {
        struct request *req;
    };

    // Value to store start timestamp
    struct val_t {
        u64 ts;
    };

    // Output event to userspace
    struct data_t {
        u64 delta_us;
        u64 bytes;
        u64 sector;
        u64 rwflag;
        u32 dev_major;
        u32 dev_minor;
    };

    // Maps
    BPF_HASH(start, struct key_t, struct val_t);
    BPF_PERF_OUTPUT(events);

    // Variable to filter by cgroup id (populated by python)
    // 0 means trace all (or logic handled in python, but better here)
    // We use a map or global variable. Array map is standard for config.
    BPF_ARRAY(filter_cgid, u64, 1);

    TRACEPOINT_PROBE(block, block_rq_issue) {
        u64 id = bpf_get_current_cgroup_id();
        int key_idx = 0;
        u64 *target_cgid = filter_cgid.lookup(&key_idx);

        // Filter: if target_cgid is set (>0) and doesn't match current, skip
        if (target_cgid && *target_cgid != 0 && *target_cgid != id) {
            return 0;
        }

        struct key_t key = {};
        key.req = (struct request *)args->rwbs; // Using args available in tracepoint

        // Note: tracepoint args vary by kernel.
        // standard args for block_rq_issue: dev, sector, nr_sector, bytes, rwbs, comm, cmd
        // But we need the 'struct request *' to match with completion.
        // Actually, typical tracepoint definition has 'struct request *rq' or cast via args.
        // Let's use the pointer address as the key, which is common in bcc tools (biolatency).
        // For block_rq_issue, args->dev is dev_t. We rely on the implicit (struct request *) argument
        // if accessible, OR we assume args->dev + context.
        // A robust way in BCC for tracepoints is slightly tricky if we need the struct request pointer
        // matching exactly.
        // However, 'biolatency' tool uses kprobes on blk_account_io_start/done or similar.
        // Tracepoints are more stable API-wise.
        // block_rq_issue args: dev, sector, nr_sector, bytes, rwbs, comm, cmd
        // No request pointer in standard tracepoint args?
        // Let's check /sys/kernel/debug/tracing/events/block/block_rq_issue/format
        // usually it doesn't have the pointer exposed cleanly as a field we can map 1:1 to completion
        // WITHOUT a bit of hacking or using kprobes.

        // Strategy switch: Use kprobes on `blk_mq_start_request` (issue) and `blk_account_io_done` (completion)?
        // Or stick to tracepoints if we can match.
        // biolatency.py uses:
        //   kprobe:blk_account_io_start / kprobe:blk_mq_start_request
        //   kprobe:blk_account_io_done / kprobe:__blk_account_io_done
        // Let's use kprobes for access to 'struct request *'.
        return 0;
    }
    """

    # improved BPF program using kprobes for reliable struct request access
    BPF_PROGRAM_KPROBE = r"""
    #include <uapi/linux/ptrace.h>
    #include <linux/blkdev.h>
    #include <linux/blk-mq.h>

    #ifndef REQ_OP_MASK
    #define REQ_OP_MASK ((1 << 8) - 1)
    #endif
    #define REQ_OP_READ 0
    #define REQ_OP_WRITE 1

    struct start_req_t {
        u64 ts;
        u64 cgroup_id;
    };

    struct data_t {
        u64 delta_us;
        u64 bytes;
        u64 rwflag; // 0=read, 1=write
        u32 dev_major;
        u32 dev_minor;
    };

    BPF_HASH(start, struct request *, struct start_req_t);
    BPF_PERF_OUTPUT(events);
    BPF_ARRAY(filter_cgid, u64, 1);

    // KPROBE: Trace request start (Issue)
    // Matches: blk_account_io_start, blk_mq_start_request
    void trace_req_start(struct pt_regs *ctx, struct request *req) {
        u64 id = bpf_get_current_cgroup_id();
        int key_idx = 0;
        u64 *target_cgid = filter_cgid.lookup(&key_idx);

        // Filter: if target matches 0 (all) or current matches target
        if (target_cgid && *target_cgid != 0 && *target_cgid != id) {
            return;
        }

        struct start_req_t val = {};
        val.ts = bpf_ktime_get_ns();
        val.cgroup_id = id;

        start.update(&req, &val);
    }

    // KPROBE: Trace request completion
    // Matches: blk_account_io_done
    void trace_req_done(struct pt_regs *ctx, struct request *req) {
        struct start_req_t *stp;
        u64 ts;

        stp = start.lookup(&req);
        if (!stp) {
            return; // Missed start or filtered out
        }

        // Extract op type using REQ_OP_MASK (low 8 bits)
        u32 op = req->cmd_flags & REQ_OP_MASK;

        // Only emit events for READ and WRITE operations
        // Ignore DISCARD (3), FLUSH (2), ZONE_* ops, etc.
        if (op != REQ_OP_READ && op != REQ_OP_WRITE) {
            start.delete(&req);
            return;
        }

        // Calculate delta
        ts = bpf_ktime_get_ns();
        u64 delta_us = (ts - stp->ts) / 1000;

        // Get details
        struct data_t data = {};
        data.delta_us = delta_us;
        data.bytes = req->__data_len;
        data.rwflag = op; // 0 for read, 1 for write

        start.delete(&req);

        events.perf_submit(ctx, &data, sizeof(data));
    }
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self._interval_seconds = max(0.05, min(1.0, interval_ms / 1000))
        self._container_cgroup_id: int | None = None
        self._bpf: Any = None
        self._samples: list[CollectorSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._last_snapshot_time: float | None = None

        # Internal Aggregators
        self._aggr_lock = threading.Lock()
        self._read_bytes = 0
        self._write_bytes = 0
        self._read_ops = 0
        self._write_ops = 0
        self._read_us = 0
        self._write_us = 0

    @property
    def name(self) -> str:
        return "ebpf_block"

    def is_available(self) -> bool:
        """Check if eBPF/BCC is available and we have permissions."""
        # Simple check: if BPF imported successfully, we might be good.
        # Runtime permission issues are handled in start().
        return BPF is not None

    def start(self, container_id: str) -> None:
        """Start collecting metrics for a container using eBPF.

        Args:
            container_id: Docker container ID.
        """
        if self._running:
            return

        # 1. Resolve Cgroup ID (Inode)
        self._cgroup_path = self._find_cgroup_path(container_id)
        if self._cgroup_path and self._cgroup_path.exists():
            try:
                # Inode number is the cgroup ID seen by bpf_get_current_cgroup_id()
                self._container_cgroup_id = os.stat(self._cgroup_path).st_ino
                logger.debug(
                    f"Resolved cgroup {self._cgroup_path} to ID {self._container_cgroup_id}"
                )
            except OSError as e:
                logger.warning(f"Failed to get inode for cgroup {self._cgroup_path}: {e}")
                raise RuntimeError(f"Failed to resolve cgroup ID: {e}") from e
        else:
            raise RuntimeError(
                f"Could not find cgroup path for container {container_id}. EBPF needs valid cgroup."
            )

        # 2. Init BPF
        try:
            self._bpf = BPF(text=self.BPF_PROGRAM_KPROBE)

            # Attach Kprobes
            # 1. Issue/Start Tracepoint
            start_fn_name = "trace_req_start"
            start_candidates = [b"blk_account_io_start", b"blk_mq_start_request"]
            attached_start = False

            for sym in start_candidates:
                if self._bpf.get_kprobe_functions(sym):
                    try:
                        self._bpf.attach_kprobe(event=sym, fn_name=start_fn_name)
                        logger.debug(f"Attached kprobe to {sym.decode()} for request start")
                        attached_start = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed to attach to {sym.decode()}: {e}")

            if not attached_start:
                raise RuntimeError(
                    f"Could not find any suitable start kprobes (tried: {[s.decode() for s in start_candidates]})"
                )

            # 2. Completion/Done Tracepoint
            done_fn_name = "trace_req_done"
            done_candidates = [
                b"blk_account_io_done",  # Standard (older kernels or when not inlined)
                b"__blk_account_io_done",  # Underscored variant
                b"blk_mq_complete_request",  # Global symbol (stable API, newer kernels)
                b"blk_mq_end_request",  # Fallback
            ]
            attached_done = False

            for sym in done_candidates:
                if self._bpf.get_kprobe_functions(sym):
                    try:
                        self._bpf.attach_kprobe(event=sym, fn_name=done_fn_name)
                        logger.debug(f"Attached kprobe to {sym.decode()} for request completion")
                        attached_done = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed to attach to {sym.decode()}: {e}")

            if not attached_done:
                raise RuntimeError(
                    f"Could not find any suitable completion kprobes (tried: {[s.decode() for s in done_candidates]})"
                )

            # Set Filter
            if self._container_cgroup_id:
                filter_array = self._bpf["filter_cgid"]
                filter_array[0] = ctypes.c_ulonglong(self._container_cgroup_id)

        except Exception as e:
            logger.error(f"Failed to initialize eBPF: {e}")
            if self._bpf:
                self._bpf.cleanup()
                self._bpf = None
            raise RuntimeError(f"eBPF initialization failed: {e}") from e

        # 3. Start polling loop
        self._running = True
        self._start_time = time.monotonic()
        self._samples = []

        # Open perf buffer
        self._bpf["events"].open_perf_buffer(self._handle_event)

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started eBPF block I/O tracing for cgroup {self._container_cgroup_id}")

        # Collect initial sample
        self._collect_sample()

    def _handle_event(self, cpu: int, data: Any, size: int) -> None:
        """Callback for perf buffer events."""
        event = self._bpf["events"].event(data)

        with self._aggr_lock:
            if event.rwflag == 1:  # REQ_OP_WRITE
                self._write_bytes += event.bytes
                self._write_ops += 1
                self._write_us += event.delta_us
            else:  # REQ_OP_READ (0)
                self._read_bytes += event.bytes
                self._read_ops += 1
                self._read_us += event.delta_us

    def _monitor_loop(self) -> None:
        """Background loop to poll perf buffer and take samples."""
        while self._running and self._bpf:
            try:
                self._bpf.perf_buffer_poll(timeout=100)  # 100ms timeout

                # Periodically snapshot samples?
                # Ideally, we sample at interval_ms.
                # Since poll blocks, we need to track time.
                # Actually, simpler: verify if enough time passed to create a 'sample' object
                # for parity with CgroupsCollector.
                # But since we aggregate continuously in handle_event, we can just snapshot
                # the *current total* as a sample every interval.
                self._maybe_snapshot_sample()

            except Exception as e:
                logger.error(f"eBPF monitor loop error: {e}")
                time.sleep(0.1)

    def _collect_sample(self) -> None:
        """Collect a sample with current aggregated counters."""
        now_mono = time.monotonic()
        with self._aggr_lock:
            sample = CollectorSample(
                timestamp=datetime.now(UTC),
                monotonic_time=now_mono,
                blkio_read_bytes=self._read_bytes,
                blkio_write_bytes=self._write_bytes,
                blkio_read_ops=self._read_ops,
                blkio_write_ops=self._write_ops,
                blkio_read_usec=self._read_us,
                blkio_write_usec=self._write_us,
            )
        with self._lock:
            self._samples.append(sample)
        self._last_snapshot_time = now_mono

    def _maybe_snapshot_sample(self) -> None:
        """Periodically snapshot the current aggregated counters."""
        now = time.monotonic()
        if self._last_snapshot_time is None:
            self._last_snapshot_time = now
            return
        if now - self._last_snapshot_time >= self._interval_seconds:
            self._collect_sample()

    def stop(self) -> CollectorResult:
        """Stop collecting and return metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        # Collect final sample
        self._collect_sample()

        if self._bpf:
            self._bpf.cleanup()
            self._bpf = None

        # Build Result using the shared aggregation logic
        return self._aggregate_samples()

    def get_summary(
        self,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
    ) -> CollectorResult:
        """Get aggregated I/O metrics, optionally filtered by time window.

        This allows re-aggregation for query windows, consistent with CgroupsV2Collector.

        Args:
            start_timestamp: Start of the window (inclusive)
            end_timestamp: End of the window (inclusive)

        Returns:
            Aggregated I/O metrics for the specified window
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return CollectorResult()

        if start_timestamp or end_timestamp:
            filtered_samples = []
            for s in samples:
                if start_timestamp and s.timestamp < start_timestamp:
                    continue
                if end_timestamp and s.timestamp > end_timestamp:
                    continue
                filtered_samples.append(s)

            if len(filtered_samples) < 2 and len(samples) >= 2:
                logger.debug(
                    f"Time window filtering left {len(filtered_samples)} eBPF samples "
                    f"(original: {len(samples)}). Using nearest bracketing samples."
                )
                bracketed = self._find_bracketing_samples(samples, start_timestamp, end_timestamp)
                if len(bracketed) >= 2:
                    samples = bracketed
            else:
                samples = filtered_samples

        return self._aggregate_samples(samples)

    def _find_bracketing_samples(
        self,
        samples: list[CollectorSample],
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
    ) -> list[CollectorSample]:
        """Find samples that bracket the requested time window."""
        before_sample: CollectorSample | None = None
        after_sample: CollectorSample | None = None

        for s in samples:
            if (
                start_timestamp
                and s.timestamp <= start_timestamp
                and (before_sample is None or s.timestamp > before_sample.timestamp)
            ):
                before_sample = s
            if (
                end_timestamp
                and s.timestamp >= end_timestamp
                and (after_sample is None or s.timestamp < after_sample.timestamp)
            ):
                after_sample = s

        if before_sample is None and samples:
            before_sample = samples[0]
        if after_sample is None and samples:
            after_sample = samples[-1]

        if before_sample and after_sample and before_sample is not after_sample:
            return [before_sample, after_sample]
        return []

    def _aggregate_samples(self, samples: list[CollectorSample] | None = None) -> CollectorResult:
        """Aggregate samples into a CollectorResult.

        If samples is None, uses all collected samples. The result contains
        I/O deltas computed from first to last sample in the list.
        """
        if samples is None:
            with self._lock:
                samples = list(self._samples)

        if not samples:
            return CollectorResult()

        # Sort by monotonic time
        samples = sorted(
            samples,
            key=lambda s: s.monotonic_time if s.monotonic_time > 0 else s.timestamp.timestamp(),
        )

        first_sample = samples[0]
        last_sample = samples[-1]

        # Duration from monotonic time
        if first_sample.monotonic_time > 0 and last_sample.monotonic_time > 0:
            duration = last_sample.monotonic_time - first_sample.monotonic_time
        else:
            duration = (last_sample.timestamp - first_sample.timestamp).total_seconds()
        duration = max(duration, 0.001)  # Avoid division by zero

        # Compute deltas
        read_bytes_delta = max(0, last_sample.blkio_read_bytes - first_sample.blkio_read_bytes)
        write_bytes_delta = max(0, last_sample.blkio_write_bytes - first_sample.blkio_write_bytes)
        read_ops_delta = max(0, last_sample.blkio_read_ops - first_sample.blkio_read_ops)
        write_ops_delta = max(0, last_sample.blkio_write_ops - first_sample.blkio_write_ops)
        read_usec_delta = max(0, last_sample.blkio_read_usec - first_sample.blkio_read_usec)
        write_usec_delta = max(0, last_sample.blkio_write_usec - first_sample.blkio_write_usec)

        return CollectorResult(
            total_read_bytes=read_bytes_delta,
            total_write_bytes=write_bytes_delta,
            total_read_ops=read_ops_delta,
            total_write_ops=write_ops_delta,
            total_read_usec=read_usec_delta,
            total_write_usec=write_usec_delta,
            avg_read_iops=read_ops_delta / duration if duration > 0 else 0.0,
            avg_write_iops=write_ops_delta / duration if duration > 0 else 0.0,
            duration_seconds=duration,
            samples=samples,
            sample_count=len(samples),
        )

    def _find_cgroup_path(self, container_id: str) -> Path | None:
        # Reuse logic from CgroupsV2Collector or CgroupsUtils?
        # Ideally we refactor find_cgroup_path to a util.
        # For now, duplicate standard patterns or import from cgroups_collector?
        # Let's import to avoid duplication if possible, or just copy-paste for safety.
        # Actually in cgroups_collector it's an instance method.
        # Quick copy of patterns:
        patterns = [
            "/sys/fs/cgroup/system.slice/docker-{container_id}.scope",
            "/sys/fs/cgroup/docker/{container_id}",
            "/sys/fs/cgroup/system.slice/docker-{container_id}.scope/container",  # sometimes here
        ]
        for p in patterns:
            path = Path(p.format(container_id=container_id))
            if path.exists():
                return path

        # Fallback search
        cgroup_base = Path("/sys/fs/cgroup")
        system_slice = cgroup_base / "system.slice"
        if system_slice.exists():
            for child in system_slice.iterdir():
                if container_id[:12] in child.name and child.is_dir():
                    return child
        return None
