"""
Schemas for generation pipeline output.
Prefer explicit structure over ad-hoc dicts.
"""

from typing import Any, TypedDict


class PromptEntry(TypedDict):
    """Single prompt in the registry."""
    id: str
    text: str
    version: str


class GenerationResult(TypedDict, total=False):
    """Result of one generate() call. All fields optional for partial failure."""
    output_text: str
    error: str
    model_id: str
    prompt_id: str
    prompt_version: str
    runtime_sec: float
    input_token_count: int
    output_token_count: int


class JSONLLine(TypedDict, total=False):
    """
    One line of the generation JSONL artifact.
    job_id, timestamp_utc, prompt_id, prompt_version, model_id required for valid lines.
    error set on failure; output_text set on success (or placeholder in dry-run).
    """
    job_id: str
    timestamp_utc: str
    chunk_index: int
    prompt_id: str
    prompt_version: str
    prompt_text: str
    model_id: str
    input_token_count: int
    output_text: str | None
    output_token_count: int | None
    runtime_sec: float | None
    error: str | None
    dry_run: bool
