"""Configuration loading and saving utilities.

Supports YAML and JSON configuration files with schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ann_suite.core.schemas import BenchmarkConfig


def load_config(path: Path | str) -> BenchmarkConfig:
    """Load and validate a benchmark configuration file.

    Args:
        path: Path to YAML or JSON configuration file

    Returns:
        Validated BenchmarkConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
        pydantic.ValidationError: If config is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    with open(path, encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")

    return BenchmarkConfig.model_validate(data)
