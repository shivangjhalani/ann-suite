"""Dataset registry resolution.

The registry is a declarative YAML manifest of available datasets, kept as a
data file (``library/datasets/registry.yaml``) so adding a dataset is an edit,
not a code change. This module is the single place that knows how to locate and
read it, so no other code has to guess at filesystem paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Default location of the registry relative to the project root.
DEFAULT_REGISTRY = Path("library/datasets/registry.yaml")

#: Default output directory for prepared datasets (matches BenchmarkConfig.data_dir).
DEFAULT_DATA_DIR = Path("./data")


def resolve_registry_path(registry_path: Path | None = None) -> Path:
    """Resolve the registry path to a single, stable location.

    Resolution order:
    1. An explicit ``registry_path`` argument.
    2. ``library/datasets/registry.yaml`` relative to the project root (found by
       walking up from this file).

    Raises:
        FileNotFoundError: If no registry can be located.
    """
    if registry_path is not None:
        if registry_path.exists():
            return registry_path.resolve()
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    # Walk up from this file to the project root and look for the default path.
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / DEFAULT_REGISTRY
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Registry not found. Expected {DEFAULT_REGISTRY} at the project root, "
        "or pass an explicit registry path."
    )


def load_registry(registry_path: Path | None = None) -> dict[str, Any]:
    """Load the dataset registry as a dict.

    Args:
        registry_path: Optional explicit path; resolved via :func:`resolve_registry_path`.

    Returns:
        The registry contents (empty dict if the ``datasets`` key is absent).
    """
    path = resolve_registry_path(registry_path)
    logger.debug("Loading dataset registry from %s", path)
    with path.open(encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        logger.warning("Registry %s is empty or malformed; treating as empty", path)
        return {}
    return result


def _expand_available(datasets: dict[str, Any]) -> list[str]:
    """Return dataset names plus their subset names."""
    available: list[str] = []
    for name, config in datasets.items():
        available.append(name)
        subsets = config.get("subsets")
        if isinstance(subsets, dict):
            available.extend(subsets.keys())
    return available


def find_dataset(
    name: str, registry_path: Path | None = None
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Resolve a dataset name (top-level or subset) to its configuration.

    Args:
        name: Dataset name, or a subset name.
        registry_path: Optional explicit registry path.

    Returns:
        Tuple of ``(parent_name, parent_config, subset_config)``. ``subset_config``
        is ``None`` when ``name`` is a top-level dataset.

    Raises:
        ValueError: If the dataset name is not in the registry.
    """
    registry = load_registry(registry_path)
    datasets: dict[str, Any] = registry.get("datasets", {})

    if name in datasets:
        return name, datasets[name], None

    for ds_name, ds_config in datasets.items():
        subsets = ds_config.get("subsets")
        if isinstance(subsets, dict) and name in subsets:
            return ds_name, ds_config, subsets[name]

    available = _expand_available(datasets)
    raise ValueError(f"Unknown dataset: {name}. Available: {available}")


def list_datasets(registry_path: Path | None = None) -> None:
    """Print available datasets and their subsets."""
    registry = load_registry(registry_path)
    datasets: dict[str, Any] = registry.get("datasets", {})

    print("Available datasets:")
    print("-" * 60)
    for name, config in datasets.items():
        print(f"  {name}")
        print(f"    {config.get('description', 'No description')}")
        print(f"    Dimension: {config.get('dimension')}, Metric: {config.get('distance_metric')}")

        subsets = config.get("subsets")
        if isinstance(subsets, dict):
            print("    Subsets:")
            for sub_name, sub_config in subsets.items():
                print(f"      - {sub_name} ({sub_config.get('base_count')} vectors)")
        print()
