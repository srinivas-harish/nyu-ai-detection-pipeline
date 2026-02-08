"""
Deterministic chunking: min/max tokens, overlap.
Same input + params => same chunks.
"""

from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

# Default encoding for token count when tiktoken unavailable: whitespace split
_DEFAULT_ENCODING = "cl100k_base"


def _get_encoder(encoding_name: str | None = None):
    """Return tokenizer that yields token count. Deterministic."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(encoding_name or _DEFAULT_ENCODING)
    except Exception:
        return None


def token_count(text: str, encoding_name: str | None = None) -> int:
    """Token count for string. Deterministic."""
    enc = _get_encoder(encoding_name)
    if enc is not None:
        return len(enc.encode(text))
    return len(text.split())


def chunk_text(
    text: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks. Deterministic.
    Each chunk has token count in [min_tokens, max_tokens] (last chunk may be shorter).
    overlap_tokens is the number of tokens shared with the next chunk.

    Returns list of dicts: {start_idx, end_idx, text, token_count}.
    start_idx/end_idx are token indices in the full sequence (for provenance).
    """
    enc = _get_encoder(encoding_name)
    if enc is not None:
        tokens = enc.encode(text)
    else:
        tokens = text.split()  # type: ignore

    if not tokens:
        return []
    if min_tokens <= 0 or max_tokens < min_tokens:
        return []
    overlap = max(0, min(overlap_tokens, max_tokens - 1))
    step = max_tokens - overlap
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        if enc is not None:
            chunk_text_str = enc.decode(chunk_tokens)
        else:
            chunk_text_str = " ".join(chunk_tokens)  # type: ignore
        tc = len(chunk_tokens)
        if tc >= min_tokens or start + max_tokens >= len(tokens):
            chunks.append({
                "start_idx": start,
                "end_idx": end,
                "text": chunk_text_str,
                "token_count": tc,
            })
        start += step
        if start >= len(tokens):
            break
    return chunks
