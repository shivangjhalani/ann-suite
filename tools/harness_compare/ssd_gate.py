"""SSD-state gate for native search benchmarks.

On this QLC drive the same index file gives 3-5x different QPS depending on flash state, and during the
2026-10-20 100M session QD64 probes fell from ~136-161 to 15-34 kIOPS in the middle of a run (two of the
24 points were 28-41% slow). The gate does two things:

  pre()   probe before a point; if it is below `ratio` x the reference for that file, wait and re-probe
          (the dips were transient), up to `max_waits` times.
  post()  probe after a point; a point whose pre or post probe is below the gate is `suspect` and the
          caller re-runs it.

The reference for a file is the median of its recent accepted pre-probes (first 3 probes seed it), so a
whole session in a slower-but-stable state is not rejected, only transient dips are.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType


class SsdGate:
    def __init__(
        self,
        host: ModuleType,
        ratio: float = 0.7,
        wait_s: float = 20.0,
        max_waits: int = 4,
        fio_seconds: int = 2,
        history: int = 15,
        seed_probes: int = 3,
    ) -> None:
        self.host = host
        self.ratio = ratio
        self.wait_s = wait_s
        self.max_waits = max_waits
        self.seed_probes = seed_probes
        host.FIO_SECONDS = fio_seconds  # host_state reads this module global at call time
        self._ref: dict[Path, deque[float]] = defaultdict(lambda: deque(maxlen=history))

    def _probe(self, target: Path) -> dict:
        return self.host.probe_device(target)

    def reference(self, target: Path) -> float | None:
        values = self._ref[target]
        return statistics.median(values) if values else None

    def _seed(self, target: Path) -> None:
        while len(self._ref[target]) < self.seed_probes:
            kiops = self._probe(target).get("probe_qd64_kiops")
            if kiops is None:
                raise RuntimeError(f"SSD probe unavailable for {target} (is fio installed?)")
            self._ref[target].append(kiops)

    def pre(self, target: Path) -> dict:
        """Probe before a point, waiting out transient dips. Result has gate_ok/gate_waits/ref_kiops."""
        self._seed(target)
        ref = self.reference(target) or 0.0
        waits = 0
        while True:
            probe = self._probe(target)
            kiops = probe.get("probe_qd64_kiops", 0.0)
            ok = kiops >= self.ratio * ref
            if ok or waits >= self.max_waits:
                break
            waits += 1
            time.sleep(self.wait_s)
        if ok:
            self._ref[target].append(kiops)
        return {**probe, "gate_ok": ok, "gate_waits": waits, "ref_kiops": round(ref, 1)}

    def post(self, target: Path) -> dict:
        """Probe after a point; ok=False means the drive was degraded at the end of the run."""
        ref = self.reference(target) or 0.0
        probe = self._probe(target)
        return {**probe, "ok": probe.get("probe_qd64_kiops", 0.0) >= self.ratio * ref}
