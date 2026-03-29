"""eBPF collector for research-grade block I/O tracing.

This collector uses eBPF (via BCC) to trace block layer I/O request issue and completion
events. It provides metrics that cgroups cannot:

1. Per-operation latency histograms (p50, p95, p99, max)
2. Request size distribution
3. Access pattern analysis (sequential vs random)

DESIGN: Device-based filtering instead of cgroup-based
-------------------------------------------------------
Container-attributed I/O is problematic at the block layer because buffered I/O
is submitted by kernel workers (kworker), not the container process. This collector
uses DEVICE-BASED filtering: it traces ALL I/O to the storage device used by the
container, during the container's lifetime. This is accurate for benchmarks running
on dedicated storage.

For I/O volume metrics (total bytes, ops), use CgroupsCollector which has accurate
writeback accounting. This collector provides LATENCY and ACCESS PATTERN metrics only.

Requires:
- Root privileges or CAP_SYS_ADMIN
- Kernel headers
- BCC installed
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from bcc import BPF
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import BPF from bcc: {e}")
    BPF = None
except Exception as e:
    logging.getLogger(__name__).error(f"Unexpected error importing BPF from bcc: {e}")
    BPF = None

logger = logging.getLogger(__name__)


@dataclass
class LatencyHistogram:
    """Latency histogram with log2 buckets."""

    buckets: dict[int, int] = field(default_factory=dict)  # bucket_index -> count
    total_count: int = 0
    total_us: int = 0

    def add(self, latency_us: int) -> None:
        """Add a latency sample."""
        # Log2 bucket (1us, 2us, 4us, ..., up to ~1s)
        if latency_us <= 0:
            bucket = 0
        else:
            bucket = latency_us.bit_length() - 1
        self.buckets[bucket] = self.buckets.get(bucket, 0) + 1
        self.total_count += 1
        self.total_us += latency_us

    def percentile(self, p: float) -> int | None:
        """Get latency at given percentile (0-100)."""
        if self.total_count == 0:
            return None
        target = int(self.total_count * p / 100)
        cumulative = 0
        for bucket in sorted(self.buckets.keys()):
            cumulative += self.buckets[bucket]
            if cumulative >= target:
                # Return upper bound of bucket
                return 1 << bucket
        return None

    def mean(self) -> float | None:
        """Get mean latency in microseconds."""
        if self.total_count == 0:
            return None
        return self.total_us / self.total_count

    def max(self) -> int | None:
        """Get approximate max latency (upper bound of highest bucket)."""
        if not self.buckets:
            return None
        return 1 << max(self.buckets.keys())


@dataclass
class EBPFMetrics:
    """Metrics collected via eBPF."""

    # Per-operation latency
    read_latency: LatencyHistogram = field(default_factory=LatencyHistogram)
    write_latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    # Request counts (for cross-validation with cgroups)
    read_ops: int = 0
    write_ops: int = 0
    read_bytes: int = 0
    write_bytes: int = 0

    # Request size tracking
    read_sizes: list[int] = field(default_factory=list)
    write_sizes: list[int] = field(default_factory=list)


class EBPFCollector:
    """Collector that uses eBPF for per-operation I/O latency and access patterns.

    Uses device-based filtering (not cgroup-based) to capture ALL I/O to the
    target storage device during the container's lifetime.
    """

    # BPF Program with device-based filtering and latency tracking
    BPF_PROGRAM = r"""
    #include <uapi/linux/ptrace.h>
    #include <linux/blkdev.h>
    #include <linux/blk-mq.h>
    #include <linux/genhd.h>

    #ifndef REQ_OP_MASK
    #define REQ_OP_MASK ((1 << 8) - 1)
    #endif
    #define REQ_OP_READ 0
    #define REQ_OP_WRITE 1

    struct start_req_t {
        u64 ts;
        u32 bytes;
    };

    struct data_t {
        u64 delta_us;
        u32 bytes;
        u8 rwflag;  // 0=read, 1=write
    };

    // Track in-flight requests by request pointer
    BPF_HASH(start, struct request *, struct start_req_t);

    // Perf buffer for completed I/O events
    BPF_PERF_OUTPUT(events);

    // Target device (major:minor) - set from userspace
    BPF_ARRAY(target_dev_major, u32, 1);
    BPF_ARRAY(target_dev_minor, u32, 1);

    // Helper to get device from request
    static inline int get_req_dev(struct request *req, u32 *major, u32 *minor) {
        struct gendisk *disk = NULL;
        bpf_probe_read_kernel(&disk, sizeof(disk), &req->q->disk);
        if (!disk) {
            return -1;
        }
        bpf_probe_read_kernel(major, sizeof(*major), &disk->major);
        bpf_probe_read_kernel(minor, sizeof(*minor), &disk->first_minor);
        return 0;
    }

    // KPROBE: Trace request start
    int trace_req_start(struct pt_regs *ctx, struct request *req) {
        u32 major = 0, minor = 0;

        // Get device info
        if (get_req_dev(req, &major, &minor) < 0) {
            return 0;
        }

        // Filter by target device
        int key = 0;
        u32 *target_major = target_dev_major.lookup(&key);
        u32 *target_minor = target_dev_minor.lookup(&key);

        if (target_major && target_minor) {
            if (*target_major != 0 && (*target_major != major || *target_minor != minor)) {
                return 0;  // Not our target device
            }
        }

        // Record start time and size
        struct start_req_t val = {};
        val.ts = bpf_ktime_get_ns();
        bpf_probe_read_kernel(&val.bytes, sizeof(val.bytes), &req->__data_len);

        start.update(&req, &val);
        return 0;
    }

    // KPROBE: Trace request completion
    int trace_req_done(struct pt_regs *ctx, struct request *req) {
        struct start_req_t *stp;

        stp = start.lookup(&req);
        if (!stp) {
            return 0;  // Missed start or filtered out
        }

        // Extract op type
        u32 cmd_flags = 0;
        bpf_probe_read_kernel(&cmd_flags, sizeof(cmd_flags), &req->cmd_flags);
        u32 op = cmd_flags & REQ_OP_MASK;

        // Only process READ and WRITE
        if (op != REQ_OP_READ && op != REQ_OP_WRITE) {
            start.delete(&req);
            return 0;
        }

        // Calculate latency
        u64 ts = bpf_ktime_get_ns();
        u64 delta_us = (ts - stp->ts) / 1000;

        // Emit event
        struct data_t data = {};
        data.delta_us = delta_us;
        data.bytes = stp->bytes;
        data.rwflag = (op == REQ_OP_WRITE) ? 1 : 0;

        start.delete(&req);
        events.perf_submit(ctx, &data, sizeof(data));

        return 0;
    }
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self._interval_seconds = max(0.05, min(1.0, interval_ms / 1000))
        self._bpf: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None

        # Target device
        self._target_major: int = 0
        self._target_minor: int = 0

        # Metrics
        self._metrics = EBPFMetrics()

    @property
    def name(self) -> str:
        return "ebpf_block"

    def is_available(self) -> bool:
        """Check if eBPF/BCC is available."""
        return BPF is not None

    def _resolve_device(self, path: Path) -> tuple[int, int]:
        """Resolve a path to its underlying block device (major, minor).

        Args:
            path: Path to resolve (e.g., /home/shivang/...)

        Returns:
            Tuple of (major, minor) for the block device
        """
        # Get device ID from path
        stat_result = os.stat(path)
        dev = stat_result.st_dev

        # Extract major/minor
        major = os.major(dev)
        minor = os.minor(dev)

        logger.debug(f"Resolved {path} to device {major}:{minor}")
        return major, minor

    def start(self, data_path: Path | str) -> None:
        """Start collecting I/O metrics for the given data path.

        Args:
            data_path: Path to the data directory (used to determine target device)
        """
        if self._running:
            return

        if not self.is_available():
            raise RuntimeError("BCC/eBPF is not available")

        # Resolve target device
        data_path = Path(data_path)
        if not data_path.exists():
            raise RuntimeError(f"Data path does not exist: {data_path}")

        self._target_major, self._target_minor = self._resolve_device(data_path)
        logger.info(
            f"eBPF tracing I/O on device {self._target_major}:{self._target_minor} "
            f"(resolved from {data_path})"
        )

        # Initialize BPF
        try:
            self._bpf = BPF(text=self.BPF_PROGRAM)

            # Attach kprobes
            start_fn = "trace_req_start"
            done_fn = "trace_req_done"

            # Try different symbols for request start
            start_attached = False
            for sym in [b"blk_mq_start_request", b"blk_account_io_start"]:
                if self._bpf.get_kprobe_functions(sym):
                    try:
                        self._bpf.attach_kprobe(event=sym, fn_name=start_fn)
                        logger.debug(f"Attached kprobe to {sym.decode()}")
                        start_attached = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed to attach to {sym.decode()}: {e}")

            if not start_attached:
                raise RuntimeError("Could not attach to any block I/O start function")

            # Try different symbols for request completion
            done_attached = False
            for sym in [
                b"blk_mq_end_request",
                b"__blk_mq_end_request",
                b"blk_account_io_done",
            ]:
                if self._bpf.get_kprobe_functions(sym):
                    try:
                        self._bpf.attach_kprobe(event=sym, fn_name=done_fn)
                        logger.debug(f"Attached kprobe to {sym.decode()}")
                        done_attached = True
                        break
                    except Exception as e:
                        logger.debug(f"Failed to attach to {sym.decode()}: {e}")

            if not done_attached:
                raise RuntimeError("Could not attach to any block I/O completion function")

            # Set target device in BPF maps
            import ctypes

            major_map = self._bpf["target_dev_major"]
            minor_map = self._bpf["target_dev_minor"]
            major_map[ctypes.c_int(0)] = ctypes.c_uint(self._target_major)
            minor_map[ctypes.c_int(0)] = ctypes.c_uint(self._target_minor)

            logger.debug(f"Set eBPF target device to {self._target_major}:{self._target_minor}")

        except Exception as e:
            logger.error(f"Failed to initialize eBPF: {e}")
            if self._bpf:
                self._bpf.cleanup()
                self._bpf = None
            raise RuntimeError(f"eBPF initialization failed: {e}") from e

        # Reset metrics
        self._metrics = EBPFMetrics()

        # Start polling
        self._running = True
        self._start_time = time.monotonic()

        # Open perf buffer
        self._bpf["events"].open_perf_buffer(self._handle_event)

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started eBPF block I/O tracing for device {self._target_major}:{self._target_minor}")

    def _handle_event(self, cpu: int, data: Any, size: int) -> None:
        """Callback for perf buffer events."""
        event = self._bpf["events"].event(data)

        with self._lock:
            if event.rwflag == 1:  # Write
                self._metrics.write_ops += 1
                self._metrics.write_bytes += event.bytes
                self._metrics.write_latency.add(event.delta_us)
                self._metrics.write_sizes.append(event.bytes)
            else:  # Read
                self._metrics.read_ops += 1
                self._metrics.read_bytes += event.bytes
                self._metrics.read_latency.add(event.delta_us)
                self._metrics.read_sizes.append(event.bytes)

    def _monitor_loop(self) -> None:
        """Background loop to poll perf buffer."""
        while self._running and self._bpf:
            try:
                self._bpf.perf_buffer_poll(timeout=100)
            except Exception as e:
                logger.error(f"eBPF monitor loop error: {e}")
                time.sleep(0.1)

    def stop(self) -> EBPFMetrics:
        """Stop collecting and return metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if self._bpf:
            self._bpf.cleanup()
            self._bpf = None

        with self._lock:
            return self._metrics

    def get_metrics(self) -> EBPFMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            # Return a copy
            return EBPFMetrics(
                read_latency=self._metrics.read_latency,
                write_latency=self._metrics.write_latency,
                read_ops=self._metrics.read_ops,
                write_ops=self._metrics.write_ops,
                read_bytes=self._metrics.read_bytes,
                write_bytes=self._metrics.write_bytes,
                read_sizes=list(self._metrics.read_sizes),
                write_sizes=list(self._metrics.write_sizes),
            )
