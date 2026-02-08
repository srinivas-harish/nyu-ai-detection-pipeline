"""Dataset compiler: generation JSONL -> manifest + train/valid splits."""

from authinfra.datasets.compiler import compile_dataset, dataset_summary_counts, load_manifest
from authinfra.datasets.schema import MANIFEST_SCHEMA_VERSION, Manifest

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "compile_dataset",
    "dataset_summary_counts",
    "load_manifest",
]
