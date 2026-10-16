"""Matched-recall comparison of design v2 (run_v2_e2e.sh) with Starling, from
results/evolve/e2e. Our side is a parameter grid: its Pareto points (recall vs 1-thread
mean latency, recall vs 8-thread QPS) are chosen on repetition 1 and the same points are
reported from repetition 2, so picking the best of noisy points does not flatter us.
Starling's Ls sweep is used as measured in each repetition. Values at a recall target
are log-miss interpolated between the two points around it (never extrapolated).
  python tools/evolve/e2e/compare_v2.py [--reps A,B] DATASET TARGET [TARGET ...]
DATASET: bigann | deep | t2i | t2ihi (extended T2I sweep) or VARIANT-DATASET (e.g.
noopq-deep: tools/evolve/ablation/abl_v2_noopq.py, run_v2_ablation.sh).
"""

import json
import math
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[3] / "results/evolve/e2e"
STAR = {
    "bigann": ["Starling-B0.32", "Starling-B0.40"],
    "deep": ["Starling-B0.32-deep", "Starling-B0.40-deep"],
    "t2i": ["Starling-B0.32-t2i", "Starling-B0.40-t2i"],
    "t2ihi": ["Starling-B0.32-t2i", "Starling-B0.40-t2i"],
    "100m": ["Starling-B4.0-100m"],
}


def load(name: str, algo: str | None = None) -> list[dict]:
    p = D / name
    if not p.exists():
        return []
    txt = p.read_text()
    rows = (
        [json.loads(x) for x in txt.splitlines() if x.strip()]
        if name.endswith("l")
        else json.loads(txt)
    )
    return [r for r in rows if algo is None or r["algorithm"] == algo]


def pareto(rows: list[dict], key: str, lower_better: bool) -> list[dict]:
    """Points not dominated in (recall up, key down/up)."""
    out, best = [], None
    for r in sorted(rows, key=lambda r: -r["recall"]):
        v = r[key]
        if best is None or (v < best if lower_better else v > best):
            out.append(r)
            best = v
    return sorted(out, key=lambda r: r["recall"])


def at(rows: list[dict], target: float, key: str) -> float | None:
    rows = sorted(rows, key=lambda r: r["recall"])
    for lo, hi in zip(rows, rows[1:], strict=False):
        if lo["recall"] <= target <= hi["recall"] and hi["recall"] > lo["recall"]:

            def m(r: float) -> float:
                return math.log(max(1e-4, 1 - r))

            f = (m(lo["recall"]) - m(target)) / (m(lo["recall"]) - m(hi["recall"]))
            return lo[key] + f * (hi[key] - lo[key])
    return None


def same_params(rows: list[dict], chosen: list[dict]) -> list[dict]:
    keys = {json.dumps(r["params"], sort_keys=True) for r in chosen}
    return [r for r in rows if json.dumps(r["params"], sort_keys=True) in keys]


def cell(vals: list[float | None], width: int, prec: int) -> str:
    return " ".join(f"{'n/a':>{width}}" if v is None else f"{v:{width}.{prec}f}" for v in vals)


def main() -> None:
    args = sys.argv[1:]
    r1, r2 = 1, 2
    if args[0] == "--reps":
        r1, r2 = (int(x) for x in args[1].split(","))
        args = args[2:]
    ds, targets = args[0], [float(t) for t in args[1:]]
    base = ds.split("-")[-1]
    o1 = {t: load(f"v2-{ds}-t{t}-r{r1}.jsonl") for t in (1, 8)}
    o2 = {t: load(f"v2-{ds}-t{t}-r{r2}.jsonl") for t in (1, 8)}
    lat_pts = pareto(o1[1], "lat_mean_ms", True)
    qps_pts = pareto(o1[8], "qps", False)
    print(f"{ds}: Pareto points chosen on rep 1 (latency): {[r['params'] for r in lat_pts]}")
    for t in targets:
        print(
            f"== {ds} recall {t}: 1-thr mean ms (r1 r2) | 1-thr p99.9 ms | 8-thr QPS (r1 r2) | pages | CPU ms"
        )
        rows = []
        lat = [at(lat_pts, t, "lat_mean_ms"), at(same_params(o2[1], lat_pts), t, "lat_mean_ms")]
        p99 = [at(lat_pts, t, "lat_p999_ms"), at(same_params(o2[1], lat_pts), t, "lat_p999_ms")]
        qps = [at(qps_pts, t, "qps"), at(same_params(o2[8], qps_pts), t, "qps")]
        pg = [at(same_params(o2[1], lat_pts), t, "pages")]
        cpu = [at(same_params(o2[1], lat_pts), t, "cpu_ms")]
        rows.append((f"ours v2 {ds}", lat, p99, qps, pg, cpu))
        for s in STAR[base]:  # variant runs may have their own Starling files (v2star-<ds>)
            a = [load(f"v2star-{ds}-t1-r{r}.json", s) for r in (r1, r2)]
            b = [load(f"v2star-{ds}-t8-r{r}.json", s) for r in (r1, r2)]
            if not any(a + b):
                continue
            rows.append(
                (
                    s,
                    [at(x, t, "lat_mean_ms") for x in a],
                    [at(x, t, "lat_p99_ms") for x in a],  # upstream p99.9, labelled p99
                    [at(x, t, "qps") for x in b],
                    [at(a[1], t, "pages")],
                    [at(a[1], t, "cpu_ms")],
                )
            )
        for name, lat, p99, qps, pg, cpu in rows:
            print(
                f"   {name:20} {cell(lat, 6, 3)} | {cell(p99, 6, 3)} | {cell(qps, 6, 0)} | "
                f"{cell(pg, 5, 1)} | {cell(cpu, 5, 3)}"
            )
    print("p99.9: ours from e2e_bench; Starling's is upstream p99.9 (ann-suite stores it as p99).")


if __name__ == "__main__":
    main()
