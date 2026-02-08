"""
Provider-agnostic generation adapters.
Base class + dry-run (no API) + stubs. Partial failure is normal.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from authinfra.generation.schema import GenerationResult


class BaseAdapter(ABC):
    """Provider-agnostic adapter. Subclass per provider."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier for this adapter/model (e.g. openai/gpt-4o-mini)."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        text: str,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Run one generation. Return schema with output_text or error.
        Caller handles partial failure; capture errors explicitly.
        """
        pass


class DryRunAdapter(BaseAdapter):
    """No API calls. Returns placeholder output. For testing and schema validation."""

    def __init__(self, model_id: str = "dry-run"):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        text: str,
        **kwargs: Any,
    ) -> GenerationResult:
        start = time.perf_counter()
        # Placeholder only; no network
        tc = len(text.split())  # approximate
        out: GenerationResult = {
            "model_id": self._model_id,
            "output_text": f"[DRYRUN] model={self._model_id} input_tokens≈{tc}",
            "runtime_sec": round(time.perf_counter() - start, 6),
            "input_token_count": tc,
            "output_token_count": 0,
        }
        return out


class OpenAIAdapter(BaseAdapter):
    """Stub. Not wired to API. Override generate() when integrating."""

    def __init__(self, model_id: str = "openai/gpt-4o-mini"):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        text: str,
        **kwargs: Any,
    ) -> GenerationResult:
        return {
            "model_id": self._model_id,
            "error": "OpenAI adapter is a stub; not connected to API",
            "runtime_sec": 0.0,
        }


class AnthropicAdapter(BaseAdapter):
    """Stub. Not wired to API."""

    def __init__(self, model_id: str = "anthropic/claude-sonnet"):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        text: str,
        **kwargs: Any,
    ) -> GenerationResult:
        return {
            "model_id": self._model_id,
            "error": "Anthropic adapter is a stub; not connected to API",
            "runtime_sec": 0.0,
        }


class GeminiAdapter(BaseAdapter):
    """Stub. Not wired to API."""

    def __init__(self, model_id: str = "gemini/gemini-2.5-flash-lite"):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        text: str,
        **kwargs: Any,
    ) -> GenerationResult:
        return {
            "model_id": self._model_id,
            "error": "Gemini adapter is a stub; not connected to API",
            "runtime_sec": 0.0,
        }


def get_adapter(name: str) -> BaseAdapter:
    """Resolve adapter by name. Dry-run is default/safe."""
    n = (name or "").strip().lower()
    if n in ("dry-run", "dry_run", "dryrun"):
        return DryRunAdapter()
    if n in ("openai", "gpt", "chatgpt"):
        return OpenAIAdapter()
    if n in ("anthropic", "claude"):
        return AnthropicAdapter()
    if n in ("gemini", "google"):
        return GeminiAdapter()
    return DryRunAdapter(model_id=name or "dry-run")
