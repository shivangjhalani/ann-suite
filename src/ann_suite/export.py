"""Export benchmark results into a single flat JSON file for the web dashboard.

The dashboard consumes one self-contained array of flat result rows (one row per
algorithm x dataset x hyperparameter combination), each tagged with the run name
it came from. This keeps the HTML page dependency-free and able to render every
collected experiment in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

from ann_suite.core.schemas import BenchmarkResult
from ann_suite.results.storage import ResultsStorage


def _enumerate_run_dirs(results_dir: Path) -> list[Path]:
    """Return run directories (containing result files) sorted oldest-first.

    Mirrors the discovery logic in ResultsStorage.load but returns all runs
    instead of only the latest, so the dashboard can offer a run selector.
    """
    results_dir = Path(results_dir).resolve()
    if not results_dir.is_dir():
        return []
    runs = sorted(
        (
            path
            for path in results_dir.iterdir()
            if path.is_dir()
            and ((path / "results_detailed.json").exists() or (path / "results.json").exists())
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return runs


def _is_scalar(value: object) -> bool:
    """Return True for values that serialize cleanly into a CSV cell / JSON scalar."""
    return value is None or isinstance(value, (str, int, float, bool))


def _flatten_record(record: dict[str, object]) -> dict[str, object]:
    """Flatten list/dict leaf values into a JSON-safe scalar form.

    Standard `to_flat_dict()` already emits hp_* and stats_* keys, but some
    values may themselves be lists or dicts. We keep scalars as-is and JSON-encode
    complex leaves under a `_raw` suffix so the dashboard can still inspect them.
    """
    flat: dict[str, object] = {}
    for key, value in record.items():
        if _is_scalar(value):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, default=str)
            flat[f"{key}_raw"] = value
    return flat


def to_dashboard_rows(results: list[BenchmarkResult], run_name: str) -> list[dict[str, object]]:
    """Convert one run's results into flat, JSON-serializable dashboard rows."""
    rows: list[dict[str, object]] = []
    for result in results:
        row = result.to_flat_dict()
        row["run_name"] = run_name
        rows.append(_flatten_record(row))
    return rows


def export_dashboard_json(results_dir: Path, output_path: Path | None = None) -> Path:
    """Export all runs under ``results_dir`` into a single flat JSON array.

    Args:
        results_dir: Directory containing timestamped run subdirectories.
        output_path: Destination file. Defaults to ``<results_dir>/dashboard_data.json``.

    Returns:
        Path to the written JSON file.
    """
    results_dir = Path(results_dir).resolve()
    output_path = (output_path or results_dir / "dashboard_data.json").resolve()

    storage = ResultsStorage(results_dir)
    all_rows: list[dict[str, object]] = []

    for run_dir in _enumerate_run_dirs(results_dir):
        run_name = run_dir.name
        try:
            results = storage.load(run_name)
        except Exception:
            # A malformed run should not block the rest of the dashboard.
            continue
        all_rows.extend(to_dashboard_rows(results, run_name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    return output_path
