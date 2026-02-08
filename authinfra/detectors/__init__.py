"""Detector components."""

from authinfra.detectors.baseline import (
    ensure_downloaded,
    get_model_id,
    run_inference,
)

__all__ = ["ensure_downloaded", "get_model_id", "run_inference"]
