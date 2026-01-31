"""Results module - Result storage and aggregation."""

from __future__ import annotations

from ann_suite.results.storage import ResultsStorage, load_results, store_results

__all__ = ["ResultsStorage", "store_results", "load_results"]
