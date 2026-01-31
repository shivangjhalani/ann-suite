"""Result storage and aggregation utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from ann_suite.core.schemas import BenchmarkResult

logger = logging.getLogger(__name__)


class ResultsStorage:
    """Storage manager for benchmark results.

    Handles saving results in multiple formats (JSON, CSV, Parquet)
    and loading results for analysis.
    """

    def __init__(self, results_dir: Path) -> None:
        """Initialize storage with results directory."""
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: list[BenchmarkResult],
        run_name: str | None = None,
        formats: list[str] | None = None,
    ) -> Path:
        """Save benchmark results to files.

        Args:
            results: List of BenchmarkResult objects
            run_name: Optional name for this benchmark run
            formats: Output formats (json, csv, parquet). Default: all

        Returns:
            Path to the results directory for this run
        """
        if formats is None:
            formats = ["json", "csv"]

        # Create run directory with timestamp to prevent overwrites
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_name:
            run_name = f"{run_name}_{timestamp}"
        else:
            run_name = f"benchmark_{timestamp}"
        run_dir = self.results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Save in requested formats
        if "json" in formats:
            self._save_json(results, run_dir / "results.json")
        if "csv" in formats:
            df.to_csv(run_dir / "results.csv", index=False)
        if "parquet" in formats:
            df.to_parquet(run_dir / "results.parquet", index=False)

        # Save raw results with full details
        self._save_detailed_json(results, run_dir / "results_detailed.json")

        logger.info(f"Saved results to {run_dir}")
        return run_dir

    def load(self, run_name: str | None = None) -> list[BenchmarkResult]:
        """Load benchmark results.

        Args:
            run_name: Specific run to load, or None for latest

        Returns:
            List of BenchmarkResult objects
        """
        if run_name is None:
            # Find latest run
            runs = sorted(self.results_dir.iterdir(), reverse=True)
            if not runs:
                raise FileNotFoundError("No benchmark runs found")
            run_dir = runs[0]
        else:
            run_dir = self.results_dir / run_name

        # Try to load detailed JSON first
        detailed_path = run_dir / "results_detailed.json"
        if detailed_path.exists():
            return self._load_detailed_json(detailed_path)

        # Fall back to regular JSON
        json_path = run_dir / "results.json"
        if json_path.exists():
            return self._load_json(json_path)

        raise FileNotFoundError(f"No results found in {run_dir}")

    def load_dataframe(self, run_name: str | None = None) -> pd.DataFrame:
        """Load results as a pandas DataFrame.

        Args:
            run_name: Specific run to load, or None for latest

        Returns:
            DataFrame with benchmark results
        """
        results = self.load(run_name)
        return self._results_to_dataframe(results)

    def _results_to_dataframe(self, results: list[BenchmarkResult]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows = [r.to_flat_dict() for r in results]
        return pd.DataFrame(rows)

    def _save_json(self, results: list[BenchmarkResult], path: Path) -> None:
        """Save results as phase-structured JSON for human readability."""
        data = [r.to_summary_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_detailed_json(self, results: list[BenchmarkResult], path: Path) -> None:
        """Save results with full details including samples.

        Uses compact JSON (no indent) to reduce file size, especially for
        results with many samples or large parameter sets.
        """
        data = [r.model_dump(mode="json") for r in results]
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"), default=str)

    def _load_json(self, path: Path) -> list[BenchmarkResult]:
        """Load results from simplified JSON."""
        with open(path) as f:
            data = json.load(f)
        return [BenchmarkResult.model_validate(r) for r in data]

    def _load_detailed_json(self, path: Path) -> list[BenchmarkResult]:
        """Load results from detailed JSON."""
        with open(path) as f:
            data = json.load(f)
        return [BenchmarkResult.model_validate(r) for r in data]


def store_results(
    results: list[BenchmarkResult],
    results_dir: Path,
    run_name: str | None = None,
) -> Path:
    """Convenience function to store results.

    Args:
        results: List of BenchmarkResult objects
        results_dir: Directory to store results
        run_name: Optional name for this benchmark run

    Returns:
        Path to the results directory
    """
    storage = ResultsStorage(results_dir)
    return storage.save(results, run_name)


def load_results(results_dir: Path, run_name: str | None = None) -> list[BenchmarkResult]:
    """Convenience function to load results.

    Args:
        results_dir: Directory containing results
        run_name: Specific run to load, or None for latest

    Returns:
        List of BenchmarkResult objects
    """
    storage = ResultsStorage(results_dir)
    return storage.load(run_name)
