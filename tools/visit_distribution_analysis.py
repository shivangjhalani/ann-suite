"""Analyse a DiskANN visit-count trace: the H(N) curve and the selection-policy gap.

Input is the raw uint32 dump produced by visit_trace.patch (one count per node id,
little-endian), captured over a real query workload.

Computes:
  * H(N), the visit-frequency Lorenz curve -- the fraction of node accesses a cache
    of the top-N nodes would serve. Per the cost decomposition this is the only
    unmodelled term: pages_per_query(N) = A * (1 - H(N)) is otherwise an identity.
  * The oracle gap: how much hit rate DiskANN's own sample-based selection leaves
    on the table versus ranking by real-workload visit counts. Needs the selection
    list dumped by cache_list_dump.patch.

Usage:
    python tools/visit_distribution_analysis.py TRACE.bin [--cache-list LIST.bin]
"""

from __future__ import annotations

import argparse

import numpy as np


def lorenz(counts: np.ndarray, points: list[int]) -> dict[int, float]:
    """H(N) for each N in points, using the oracle (descending-count) ordering."""
    order = np.argsort(counts)[::-1]
    cum = np.cumsum(counts[order].astype(np.int64))
    total = cum[-1]
    return {n: float(cum[min(n, len(cum)) - 1]) / total for n in points if n >= 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--cache-list", help="Node-id list from cache_list_dump.patch")
    parser.add_argument("--queries", type=int, default=10000)
    args = parser.parse_args()

    counts = np.fromfile(args.trace, dtype=np.uint32)
    total = int(counts.sum())
    touched = int((counts > 0).sum())

    print(f"nodes:          {len(counts):,}")
    print(f"total visits:   {total:,}  ({total/args.queries:.4f} per query -- this is A)")
    print(f"nodes touched:  {touched:,} ({touched/len(counts)*100:.4f}% of the graph)")
    print(f"max visits:     {counts.max():,} on a single node")

    points = [n for n in [10**e for e in range(0, 9)] if n <= len(counts)]
    points += [n for n in [5 * 10**e for e in range(0, 8)] if n <= len(counts)]
    points = sorted(set(points))

    h = lorenz(counts, points)
    uniform = 1.0 / len(counts)
    print(f"\n{'N':>12} {'fraction':>10} {'H(N)':>9} {'lift':>10} {'marginal/node':>14} {'vs random':>10}")
    prev_n, prev_h = 0, 0.0
    for n in points:
        frac = n / len(counts)
        marginal = (h[n] - prev_h) / (n - prev_n)
        print(f"{n:>12,} {frac:>10.6f} {h[n]:>9.4f} {h[n]/frac:>9.1f}x "
              f"{marginal:>14.3e} {marginal/uniform:>9.2f}x")
        prev_n, prev_h = n, h[n]

    if not args.cache_list:
        return

    selected = np.fromfile(args.cache_list, dtype=np.uint32)
    print(f"\nDiskANN selection list: {len(selected):,} node ids")
    print(f"\n{'N':>12} {'H_diskann':>11} {'H_oracle':>10} {'captured':>10} {'gap':>9}")
    for n in [p for p in points if p <= len(selected)]:
        chosen = selected[:n]
        h_diskann = float(counts[chosen].astype(np.int64).sum()) / total
        h_oracle = h[n]
        captured = h_diskann / h_oracle if h_oracle > 0 else float("nan")
        print(f"{n:>12,} {h_diskann:>11.4f} {h_oracle:>10.4f} {captured*100:>9.1f}% "
              f"{h_oracle-h_diskann:>9.4f}")
    print("\ncaptured = fraction of the achievable hit rate DiskANN's sample-based")
    print("selection actually gets. 100% would mean its ranking matches the oracle.")


if __name__ == "__main__":
    main()
