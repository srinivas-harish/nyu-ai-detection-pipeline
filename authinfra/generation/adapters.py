"""
Provider-agnostic generation adapters.
Base class + dry-run (no API) + stubs + Gemini 3 Pro. Partial failure is normal.
API keys read from api_keys.md locally only; never logged or transmitted.
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
    """
    Gemini 3 Pro adapter. Reads GEMINI_API_KEY from api_keys.md (local only).
    Explicit params: temperature, top_p, max_output_tokens. Retries on rate limit.
    """

    def __init__(
        self,
        model_id: str = "gemini-3-pro",
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
        max_retries: int = 3,
    ):
        self._model_id = model_id
        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries

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
        keys = __import__("authinfra.utils.keys", fromlist=["load_api_keys"]).load_api_keys()
        api_key = (keys.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            return {
                "model_id": self._model_id,
                "error": "GEMINI_API_KEY not found in api_keys.md (local only)",
                "runtime_sec": round(time.perf_counter() - start, 6),
            }
        full_prompt = f"{prompt}\n\n---\n\n{text}"
        last_error: str | None = None
        for attempt in range(self._max_retries):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                generation_config = {
                    "temperature": self._temperature,
                    "top_p": self._top_p,
                    "max_output_tokens": self._max_output_tokens,
                }
                model = genai.GenerativeModel(
                    model_name=self._model_id.replace("gemini/", ""),
                    generation_config=generation_config,
                )
                response = model.generate_content(
                    full_prompt,
                    request_options={"timeout": 60},
                )
                if not response or not response.text:
                    last_error = "empty response from model"
                    continue
                out_text = response.text.strip()
                in_count = kwargs.get("input_token_count") or len(text.split())
                out_count = len(out_text.split())
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    in_count = getattr(response.usage_metadata, "prompt_token_count", None) or in_count
                    out_count = getattr(response.usage_metadata, "candidates_token_count", None) or out_count
                return {
                    "model_id": self._model_id,
                    "output_text": out_text,
                    "runtime_sec": round(time.perf_counter() - start, 6),
                    "input_token_count": in_count,
                    "output_token_count": out_count,
                }
            except Exception as e:
                err_str = str(e).strip()
                last_error = err_str
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    time.sleep(2 ** (attempt + 1))
                    continue
                if attempt == self._max_retries - 1:
                    break
                time.sleep(1 * (attempt + 1))
        return {
            "model_id": self._model_id,
            "error": last_error or "generation failed",
            "runtime_sec": round(time.perf_counter() - start, 6),
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
