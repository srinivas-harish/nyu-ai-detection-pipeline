"""
Dataset manifest and compiled output schema.
Versioned; no examples silently dropped.
"""

from typing import Any, TypedDict

MANIFEST_SCHEMA_VERSION = "v1"


class FilterLogEntry(TypedDict):
    """One filter reason and count (no silent drops)."""
    reason: str
    count: int


class Manifest(TypedDict, total=False):
    """
    manifest.json in a compiled dataset folder.
    schema_version: required (e.g. v1).
    filter_log: required; every exclusion reason with count.
    """
    schema_version: str
    dataset_name: str
    created_utc: str
    source_paths: list[str]
    model_ids: list[str]
    prompt_ids: list[str]
    filter_log: list[FilterLogEntry]
    train_count: int
    valid_count: int
    train_path: str
    valid_path: str
    split_ratio: float
    split_seed: int
