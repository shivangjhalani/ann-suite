"""Amortized cache cost analysis over a cache-fraction sweep's results.csv.

Steady-state cost models are monotone in cache size: more cache, fewer page
reads, strictly better. Charging for cache *population* (the warmup phase)
changes the answer -- the optimal cache size becomes a function of query
budget. This computes break-even query volumes and the optimal cache size per
budget from an already-completed sweep.

Usage:
    python tools/cache_amortization_analysis.py RESULTS_CSV --dataset-size 100000000
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def load_sweep(path: str, dataset_size: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                cache_n = float(r["hp_search_num_nodes_to_cache"] or 0)
                rows.append(
                    {
                        "fraction": cache_n / dataset_size,
                        "cache_n": cache_n,
                        "pages": float(r["disk_io_search_pages_per_query"]),
                        "latency_ms": float(r["latency_mean_ms"]),
                        "warmup_s": float(r["time_warmup_duration_seconds"]),
                    }
                )
            except (ValueError, TypeError, KeyError):
                continue
    return sorted(rows, key=lambda x: x["fraction"])


def fit_latency_model(rows: list[dict], exclude: set[float]) -> tuple[float, float, float]:
    """Fit latency_ms = intercept + slope * pages_per_query.

    Returns (intercept, slope, max_abs_residual_pct). A large residual means the
    linear decomposition is breaking down -- typically at a fully-cached point
    where I/O goes to zero and compute dominates -- and the break-even numbers
    derived from it should not be trusted.
    """
    usable = [r for r in rows if r["cache_n"] not in exclude]
    pages = np.array([r["pages"] for r in usable])
    latency = np.array([r["latency_ms"] for r in usable])
    slope, intercept = np.polyfit(pages, latency, 1)
    predicted = intercept + slope * pages
    residual_pct = np.abs(predicted - latency) / latency * 100
    return intercept, slope, float(residual_pct.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv")
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument(
        "--exclude-cache-n",
        type=float,
        nargs="*",
        default=[],
        help="Cache sizes to exclude from the latency fit (known-contaminated points)",
    )
    args = parser.parse_args()

    rows = load_sweep(args.results_csv, args.dataset_size)
    if len(rows) < 3:
        raise SystemExit(f"need at least 3 sweep points, parsed {len(rows)}")

    intercept, slope, max_residual = fit_latency_model(rows, set(args.exclude_cache_n))
    print(f"latency_ms = {intercept:.4f} + {slope:.6f} * pages_per_query")
    print(f"max |residual| = {max_residual:.2f}%")
    if max_residual > 10:
        print("  WARNING: linear latency model fits poorly; break-even numbers below are unreliable")

    baseline = rows[0]
    baseline_lat = intercept + slope * baseline["pages"]

    print(f"\n{'fraction':>10} {'cache_n':>12} {'extra_warmup_s':>15} "
          f"{'ms_saved/query':>15} {'break_even_queries':>19}")
    for r in rows[1:]:
        extra_warmup = r["warmup_s"] - baseline["warmup_s"]
        saved_ms = baseline_lat - (intercept + slope * r["pages"])
        break_even = extra_warmup * 1000.0 / saved_ms if saved_ms > 1e-9 else float("inf")
        print(f"{r['fraction']:>10.5f} {r['cache_n']:>12,.0f} {extra_warmup:>15.1f} "
              f"{saved_ms:>15.3f} {break_even:>19,.0f}")

    print(f"\n{'query_budget':>14} {'optimal_cache_n':>17} {'optimal_fraction':>17} "
          f"{'total_seconds':>15} {'saving_vs_no_cache':>19}")
    for budget in [10**e for e in range(3, 9)]:
        best = min(
            rows,
            key=lambda r: r["warmup_s"] + budget * (intercept + slope * r["pages"]) / 1000.0,
        )
        best_cost = best["warmup_s"] + budget * (intercept + slope * best["pages"]) / 1000.0
        no_cache_cost = baseline["warmup_s"] + budget * baseline_lat / 1000.0
        saving = (1 - best_cost / no_cache_cost) * 100
        print(f"{budget:>14,} {best['cache_n']:>17,.0f} {best['fraction']:>17.5f} "
              f"{best_cost:>15,.1f} {saving:>18.1f}%")


if __name__ == "__main__":
    main()
