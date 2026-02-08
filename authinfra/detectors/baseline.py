"""
Baseline detector: Hello-SimpleAI/chatgpt-detector-roberta (Hugging Face).

Black-box wrapper. No fine-tuning or training. Output schema is strict.
Dataset citation: Hello-SimpleAI/HC3.
"""

import time
from typing import Any

# Lazy import so authinfra CLI works without torch/transformers installed until detector is used
_PIPELINE: Any = None
_MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"
_DEFAULT_MAX_TOKENS = 512


def _get_pipeline(cache_dir: str | None = None):
    """Load pipeline once; downloads model on first use."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "transformers (and torch) are required for the baseline detector. "
            "Install with: pip install transformers torch"
        ) from e
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    _PIPELINE = pipeline(
        task="text-classification",
        model=_MODEL_ID,
        **kwargs,
    )
    return _PIPELINE


def get_model_id() -> str:
    """Return the Hugging Face model id (for schema and CLI)."""
    return _MODEL_ID


def ensure_downloaded(cache_dir: str | None = None) -> bool:
    """
    Download the baseline model from Hugging Face if not already cached.
    Returns True if the model is available (after optional download).
    """
    try:
        _get_pipeline(cache_dir=cache_dir)
        return True
    except Exception:
        return False


def run_inference(
    text: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run baseline detector on raw text. Returns a strict JSON-schema result.

    Schema:
      - model: str (Hugging Face model id)
      - runtime_sec: float
      - probability: float in [0, 1] (probability of AI-generated), or null on error
      - error: str | null (if set, probability may be null)
      - input_truncated: bool (true if input was truncated for length)
    """
    result: dict[str, Any] = {
        "model": _MODEL_ID,
        "runtime_sec": 0.0,
        "probability": None,
        "error": None,
        "input_truncated": False,
    }

    # Input length guard
    if not text or not text.strip():
        result["error"] = "empty input"
        return result

    try:
        pipe = _get_pipeline(cache_dir=cache_dir)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Input length check: tokenize to see if we will truncate
    try:
        enc = pipe.tokenizer.encode(text.strip(), add_special_tokens=True)
        result["input_truncated"] = len(enc) > max_tokens
    except Exception:
        pass

    start = time.perf_counter()
    try:
        out = pipe(
            text.strip(),
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_all_scores=True,
        )
    except Exception as e:
        result["runtime_sec"] = round(time.perf_counter() - start, 6)
        result["error"] = str(e)
        return result

    result["runtime_sec"] = round(time.perf_counter() - start, 6)

    # Pipeline returns list of {label, score}; map to P(AI) in [0,1]
    if not out or not out[0]:
        result["error"] = "empty pipeline output"
        return result
    scores = out[0]
    prob_ai = 0.5
    for item in scores:
        label = (item.get("label") or "").upper()
        score = float(item.get("score", 0))
        if "AI" in label or "CHATGPT" in label or "GPT" in label:
            prob_ai = max(0.0, min(1.0, score))
            break
        if "HUMAN" in label:
            prob_ai = max(0.0, min(1.0, 1.0 - score))
            break
    result["probability"] = round(prob_ai, 6)
    return result
