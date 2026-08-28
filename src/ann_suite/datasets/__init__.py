"""Datasets module - dataset loading, downloading, and preparation."""

from __future__ import annotations

from ann_suite.datasets.download import download_file
from ann_suite.datasets.ground_truth import compute_ground_truth
from ann_suite.datasets.loader import DatasetLoader, load_dataset
from ann_suite.datasets.prepare import prepare_dataset
from ann_suite.datasets.registry import (
    DEFAULT_DATA_DIR,
    find_dataset,
    list_datasets,
    load_registry,
    resolve_registry_path,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DatasetLoader",
    "compute_ground_truth",
    "download_file",
    "find_dataset",
    "list_datasets",
    "load_dataset",
    "load_registry",
    "prepare_dataset",
    "resolve_registry_path",
]
