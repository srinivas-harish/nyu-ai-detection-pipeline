"""Generation pipeline: prompt registry, adapters, chunking, job runner."""

from authinfra.generation.adapters import BaseAdapter, DryRunAdapter, get_adapter
from authinfra.generation.chunking import chunk_text, token_count
from authinfra.generation.prompts import get_prompt, get_registry_version, list_prompt_ids
from authinfra.generation.runner import run_generation
from authinfra.generation.schema import GenerationResult, JSONLLine, PromptEntry

__all__ = [
    "BaseAdapter",
    "DryRunAdapter",
    "GenerationResult",
    "JSONLLine",
    "PromptEntry",
    "chunk_text",
    "get_adapter",
    "get_prompt",
    "get_registry_version",
    "list_prompt_ids",
    "run_generation",
    "token_count",
]
