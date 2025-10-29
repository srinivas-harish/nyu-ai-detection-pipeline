"""Create JSONL training examples for AI-text detection.

Each example contains one human chunk and all model rewrites generated
using a randomly selected prompt from a fixed set of 10 prompts.

CLI example:
  python data_helpers/make_training_examples.py \
    --input_csv data/1/clean_30.csv \
    --token_budget 100000 \
    --min_chunk 300 --max_chunk 1000 --overlap 32 \
    --models grok,chatgpt,deepseek,gemini-2.5-flash-lite \
    --temperature 0.4 --top_p 0.9 \
    --out ./train_examples.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


PROMPTS: List[str] = [
    # 1
    "Rewrite the following CRS report in the same register and structure. Preserve section headings and order. Keep total length within ±5 percent of the source. Do not add new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the revised report.",
    # 2
    "Rewrite the following CRS section in the same register and structure. Preserve the heading and numbering. Keep length within ±5 percent of the source. Do not add facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the revised section.",
    # 3
    "Paraphrase the text below while preserving tone and structure. Keep length within ±5 percent. Do not introduce new information. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text.",
    # 4
    "Re-express the report below in fresh wording, matching its structure and headings. Keep length within ±5 percent. No new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the report text.",
    # 5
    "Rewrite this section in the same style and order of ideas. Length within ±5 percent. No added facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the section.",
    # 6
    "Produce a reformulation of the following content, keeping headings, order, and tone. Length ±5 percent. No new content. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text.",
    # 7
    "Restate the following CRS material. Keep headings and structure, and stay within ±5 percent length. Do not add or remove facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text.",
    # 8
    "Rewrite the passage below with the same structure and voice. Keep headings. Length within ±5 percent. Do not add facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the passage.",
    # 9
    "Reframe the content in equivalent language while preserving the original sections and order. Stay within ±5 percent in length. No new details. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text.",
    # 10
    "Rewrite the text below. Keep the structure and headings identical. Keep overall length within ±5 percent. Do not introduce new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text.",
]


def _load_keys(path: str) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a text file into a dict.
    Expected keys: GROK_API_KEY, OPENAI_API_KEY/CHATGPT_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY.
    """
    if not path or not os.path.isfile(path):
        return {}
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Tokenizer:
    def __init__(self, model_hint: str = "gpt-4o-mini") -> None:
        self.model_hint = model_hint
        self.enc = None
        if tiktoken is not None:
            try:
                self.enc = tiktoken.encoding_for_model(model_hint)
            except Exception:
                try:
                    self.enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.enc = None

    def encode(self, s: str) -> List[int]:
        if self.enc is not None:
            try:
                return self.enc.encode(s or "")
            except Exception:
                pass
        # whitespace fallback
        # represent each "token" as a word; we won't use the ids semantically
        return (s or "").split()

    def decode(self, ids: List[int]) -> str:
        if self.enc is not None:
            try:
                return self.enc.decode(ids)
            except Exception:
                pass
        # whitespace fallback
        return " ".join(ids)  # type: ignore[arg-type]

    def count(self, s: str) -> int:
        if self.enc is not None:
            try:
                return len(self.enc.encode(s or ""))
            except Exception:
                pass
        return len((s or "").split())


def _truncate_to_tokens(tok: Tokenizer, s: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tok.encode(s)
    if len(ids) <= max_tokens:
        return s or ""
    return tok.decode(ids[:max_tokens])


def _collect_budget(rows: List[dict], tok: Tokenizer, token_budget: int) -> Tuple[str, int, List[dict]]:
    """Concatenate text fields in CSV order until reaching token_budget.
    Returns the concatenated text, total token count, and provenance list with per-report token spans.
    Each provenance item: {report_id, title, date, token_start, token_end, text}
    """
    parts: List[str] = []
    taken: List[dict] = []
    used = 0
    for row in rows:
        if used >= token_budget:
            break
        rid = (row.get("id") or "").replace(".json", "")
        title = row.get("title") or ""
        date = row.get("date") or ""
        text = row.get("text") or ""
        remain = token_budget - used
        piece = _truncate_to_tokens(tok, text, remain)
        got = tok.count(piece)
        if got <= 0:
            continue
        parts.append(piece)
        taken.append({
            "report_id": rid,
            "title": title,
            "date": date,
            "token_start": used,
            "token_end": used + got,
            "text": text,
        })
        used += got
    concat = ("\n\n".join(parts)).strip()
    total = tok.count(concat)
    return concat, total, taken


def _choose_break(tok: Tokenizer, ids: List[int], start: int, max_len: int, min_len: int) -> int:
    """Select an end index for the chunk between [min_len, max_len], snapping near paragraph breaks.
    Scans backwards from start+max_len up to 64 tokens to find a break whose decoded tail ends with \n\n.
    Returns absolute token end index.
    """
    n = len(ids)
    end = min(n, start + max_len)
    base_len = end - start
    if base_len < min_len and end < n:
        end = min(n, start + min_len)
    # Try to snap to paragraph boundary by backing off up to 64 tokens
    best_end = end
    back_limit = min(64, end - start)
    for b in range(back_limit + 1):
        cand_end = end - b
        if cand_end - start < min_len:
            break
        segment = tok.decode(ids[start:cand_end])
        if "\n\n" in segment[-400:]:
            best_end = cand_end
            break
    return best_end


def _build_chunks(tok: Tokenizer, text: str, min_chunk: int, max_chunk: int, overlap: int) -> List[Tuple[int, int]]:
    ids = tok.encode(text)
    n = len(ids)
    if n == 0:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = _choose_break(tok, ids, start, max_chunk, min_chunk)
        if end <= start:
            end = min(n, start + max_chunk)
            if end <= start:
                break
        out.append((start, end))
        if end >= n:
            break
        step = max(1, (end - start) - max(0, overlap))
        start += step
    return out


def _intersect_report_spans(tok: Tokenizer, chunks: List[Tuple[int, int]], provenance: List[dict]) -> List[List[dict]]:
    """For each chunk token span, compute a list of report spans with approximate char offsets.
    Returns a list of lists; inner list entries: {report_id, char_start, char_end}.
    Char offsets are approximated using decoded token prefixes (may not match original text exactly).
    """
    out: List[List[dict]] = []
    for (c_start, c_end) in chunks:
        spans: List[dict] = []
        for prov in provenance:
            r_start = prov["token_start"]
            r_end = prov["token_end"]
            if c_end <= r_start or c_start >= r_end:
                continue
            # overlap in tokens
            local_start = max(0, c_start - r_start)
            local_end = min(r_end - r_start, c_end - r_start)
            # approximate char offsets by decoding token prefixes
            r_text = prov.get("text") or ""
            r_ids = tok.encode(r_text)
            # guard
            local_start = max(0, min(local_start, len(r_ids)))
            local_end = max(local_start, min(local_end, len(r_ids)))
            pre = tok.decode(r_ids[:local_start])
            seg = tok.decode(r_ids[:local_end])
            spans.append({
                "report_id": prov["report_id"],
                "char_start": len(pre),
                "char_end": len(seg),
            })
        out.append(spans)
    return out


def _sha_key(chunk_text: str, prompt_id: int, model_list: str, temperature: float, top_p: float) -> str:
    h = hashlib.sha256()
    parts = [chunk_text, str(prompt_id), model_list, f"{temperature:.3f}", f"{top_p:.3f}"]
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _load_existing_keys(path: str) -> Tuple[set, int]:
    keys = set()
    count = 0
    if not os.path.isfile(path):
        return keys, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("dedupe_key")
            if k:
                keys.add(k)
    return keys, count


# ----------------------------- API calling -----------------------------

DEFAULT_MODELS = {
    "chatgpt": "gpt-4o-mini",
    "grok": "grok-2-latest",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash-lite",
}


def _provider_of(model_name: str) -> str:
    m = (model_name or "").lower().strip()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("openai") or m.startswith("chatgpt") or m.startswith("gpt-"):
        return "chatgpt"
    if m.startswith("grok") or m.startswith("xai"):
        return "grok"
    return ""


def _normalize_models(models: List[str]) -> List[Tuple[str, str]]:
    """Map user-provided model strings to (provider_key, actual_model_id).
    - provider_key in {grok, chatgpt, deepseek, gemini}
    - actual_model_id is a concrete model string acceptable by the API
    Uses DEFAULT_MODELS for bare aliases like "grok" or "chatgpt".
    """
    out: List[Tuple[str, str]] = []
    for m in models:
        m = (m or "").strip()
        if not m:
            continue
        prov = _provider_of(m)
        if not prov:
            continue
        if prov == "gemini":
            actual = m if m != "gemini" else DEFAULT_MODELS["gemini"]
        elif prov == "chatgpt":
            # Allow aliases: chatgpt, openai, gpt-*
            actual = m if (m.startswith("gpt-") or m.startswith("openai")) else DEFAULT_MODELS["chatgpt"] if m in ("chatgpt", "openai") else m
        elif prov == "deepseek":
            actual = m if m != "deepseek" else DEFAULT_MODELS["deepseek"]
        elif prov == "grok":
            actual = m if m != "grok" else DEFAULT_MODELS["grok"]
        else:
            actual = m
        out.append((prov, actual))
    return out


def _estimate_usage(tok: Tokenizer, prompt: str, input_text: str, output_text: str) -> Dict[str, int]:
    inp = tok.count(prompt + "\n\n" + input_text)
    out = tok.count(output_text or "")
    return {"input_tokens": int(inp), "output_tokens": int(out)}


def _call_gemini(model: str, prompt: str, input_text: str, max_tokens_hint: int, api_key: str, timeout: Tuple[int,int]) -> Tuple[str, Dict[str, int], str]:
    if requests is None:
        raise RuntimeError("requests not available")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model or DEFAULT_MODELS['gemini']}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": f"{prompt}\n\n{input_text}"}]}],
        "generationConfig": {"maxOutputTokens": int(max_tokens_hint)},
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        candidates = data.get("candidates") or []
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = parts[0].get("text") if parts else ""
    except Exception:
        text = ""
    usage = {}
    return text or "", usage, (model or DEFAULT_MODELS["gemini"])  # version


def _call_chatgpt(model: str, prompt: str, input_text: str, max_tokens_hint: int, api_key: str, timeout: Tuple[int,int]) -> Tuple[str, Dict[str, int], str]:
    if requests is None:
        raise RuntimeError("requests not available")
    url = "https://api.openai.com/v1/chat/completions"
    actual = model or DEFAULT_MODELS["chatgpt"]
    payload = {
        "model": actual,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{input_text}"}],
        "max_tokens": int(max_tokens_hint),
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        choices = data.get("choices") or []
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
    except Exception:
        text = ""
    usage = data.get("usage") or {}
    return text or "", usage, actual


def _call_deepseek(model: str, prompt: str, input_text: str, max_tokens_hint: int, api_key: str, timeout: Tuple[int,int]) -> Tuple[str, Dict[str, int], str]:
    if requests is None:
        raise RuntimeError("requests not available")
    url = "https://api.deepseek.com/chat/completions"
    actual = model or DEFAULT_MODELS["deepseek"]
    payload = {
        "model": actual,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{input_text}"}],
        "max_tokens": int(max_tokens_hint),
    }
    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        choices = data.get("choices") or []
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
    except Exception:
        text = ""
    usage = data.get("usage") or {}
    return text or "", usage, actual


def _call_grok(model: str, prompt: str, input_text: str, max_tokens_hint: int, api_key: str, timeout: Tuple[int,int]) -> Tuple[str, Dict[str, int], str]:
    if requests is None:
        raise RuntimeError("requests not available")
    url = "https://api.x.ai/v1/chat/completions"
    actual = model or DEFAULT_MODELS["grok"]
    payload = {
        "model": actual,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{input_text}"}],
        "max_tokens": int(max_tokens_hint),
    }
    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {api_key}"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        choices = data.get("choices") or []
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
    except Exception:
        text = ""
    usage = data.get("usage") or {}
    return text or "", usage, actual


def call_model(model_name: str, text: str, prompt: str, temperature: float, top_p: float, max_tokens_hint: int, api_keys: Dict[str, str], tok: Tokenizer, dry_run: bool = False, timeout_sec: int = 20, max_retries: int = 3, verbose: bool = False, connect_timeout: int | None = None, read_timeout: int | None = None) -> Dict[str, object]:
    """Unified entry point for all providers. Returns dict with output_text, usage, model_version.
    Retries on transient failures up to 5 attempts with exponential backoff.
    """
    provider = _provider_of(model_name)
    if not provider:
        raise ValueError(f"unsupported model name: {model_name}")

    if dry_run:
        out_text = f"[DRYRUN] model={model_name} prompt_id=? len={len(text.split())}"
        return {
            "output_text": out_text,
            "usage": _estimate_usage(tok, prompt, text, out_text),
            "model_version": model_name,
        }

    # pick key
    key_map = {
        "gemini": ["GEMINI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "chatgpt": ["OPENAI_API_KEY", "CHATGPT_API_KEY"],
        "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    }
    api_key = None
    for k in key_map[provider]:
        if k in api_keys and api_keys[k]:
            api_key = api_keys[k]
            break
    if not api_key:
        raise ValueError(f"missing API key for provider '{provider}'")

    attempts = 0
    last_err: Optional[Exception] = None
    while attempts < max_retries:
        attempts += 1
        try:
            t0 = time.time()
            if verbose:
                print(f"[api] start provider={provider} model={model_name} attempt={attempts}/{max_retries} max_out={max_tokens_hint}")
            timeout_pair = (int(connect_timeout or timeout_sec), int(read_timeout or timeout_sec))
            if provider == "gemini":
                text_out, usage, version = _call_gemini(model_name, prompt, text, max_tokens_hint, api_key, timeout_pair)
            elif provider == "chatgpt":
                text_out, usage, version = _call_chatgpt(model_name, prompt, text, max_tokens_hint, api_key, timeout_pair)
            elif provider == "deepseek":
                text_out, usage, version = _call_deepseek(model_name, prompt, text, max_tokens_hint, api_key, timeout_pair)
            elif provider == "grok":
                text_out, usage, version = _call_grok(model_name, prompt, text, max_tokens_hint, api_key, timeout_pair)
            else:
                raise ValueError("invalid provider")
            latency = time.time() - t0
            if not usage:
                usage = _estimate_usage(tok, prompt, text, text_out)
            if verbose:
                in_tok = usage.get("input_tokens") if isinstance(usage, dict) else None
                out_tok = usage.get("output_tokens") if isinstance(usage, dict) else None
                print(f"[api] done provider={provider} model={model_name} latency={latency:.2f}s usage_in={in_tok} usage_out={out_tok}")
            return {
                "output_text": text_out,
                "usage": usage,
                "model_version": version,
                "latency_sec": round(latency, 3),
            }
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[api] error provider={provider} model={model_name} attempt={attempts}: {e}")
            time.sleep(min(5, 2 ** (attempts - 1)))
            continue
    # failed
    return {
        "error": str(last_err) if last_err else "unknown error",
        "output_text": "",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "model_version": model_name,
        "latency_sec": None,
    }


def _read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create JSONL training examples from CRS CSV")
    p.add_argument("--input_csv", required=True, help="CSV with id,title,date,text")
    p.add_argument("--token_budget", type=int, required=True, help="First-N tokens from concatenated texts")
    p.add_argument("--min_chunk", type=int, default=300, help="Minimum tokens per chunk")
    p.add_argument("--max_chunk", type=int, default=1000, help="Maximum tokens per chunk")
    p.add_argument("--overlap", type=int, default=0, help="Overlap tokens between chunks (0-64)")
    p.add_argument("--out", default="./train_examples.jsonl", help="Output JSONL file path")
    p.add_argument("--models", required=True, help="Comma-separated model names")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--tokenizer_hint", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Do not call APIs; write placeholders")
    p.add_argument("--keys", default=os.path.join(os.path.dirname(__file__), "api_keys.txt"), help="Path to api_keys.txt")
    p.add_argument("--timeout_sec", type=int, default=20, help="Per-request timeout seconds")
    p.add_argument("--max_retries", type=int, default=3, help="Max retries per model call")
    p.add_argument("--verbose", action="store_true", help="Print detailed API call progress")
    p.add_argument("--connect_timeout", type=int, default=None, help="Connect timeout (overrides --timeout_sec)")
    p.add_argument("--read_timeout", type=int, default=None, help="Read timeout (overrides --timeout_sec)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    random.seed(args.seed)

    tok = Tokenizer(args.tokenizer_hint)
    tok_mode = (
        f"tiktoken({args.tokenizer_hint})" if (tiktoken is not None and getattr(tok, "enc", None) is not None) else
        ("tiktoken(cl100k_base)" if tiktoken is not None else "whitespace")
    )
    rows = _read_csv_rows(args.input_csv)
    concat_text, total_tokens, provenance = _collect_budget(rows, tok, int(args.token_budget))

    # chunk token spans
    overlap = max(0, min(64, int(args.overlap)))
    chunks = _build_chunks(tok, concat_text, int(args.min_chunk), int(args.max_chunk), overlap)
    spans_meta = _intersect_report_spans(tok, chunks, provenance)

    # setup output + idempotency
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dedupe_keys, existing = _load_existing_keys(args.out)
    raw_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    norm_models = _normalize_models(raw_models)
    # For hashing, include resolved model ids
    model_list_str = ",".join([actual for (_, actual) in norm_models])
    keys = _load_keys(args.keys)

    script_version = "v1.0.0"
    prompts_version = "v1"
    t0 = time.time()
    total_chunks = 0
    # Precompute chunks to know N for progress/ETA
    ids_all = tok.encode(concat_text)
    total_chunks = len(_build_chunks(tok, concat_text, int(args.min_chunk), int(args.max_chunk), overlap))

    if args.verbose:
        print(f"[start] seed={args.seed} tokenizer={tok_mode} prompts_version={prompts_version} script_version={script_version}")
        print(f"[start] models_resolved={[m for (_, m) in norm_models]}")
        print(f"[start] token_budget={args.token_budget} chunks={total_chunks} overlap={overlap}")

    # simple metrics
    metrics = {prov: {"ok": 0, "err": 0, "lat": []} for prov, _ in norm_models}

    with open(args.out, "a", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(chunks, start=1):
            ids = tok.encode(concat_text)
            chunk_text = tok.decode(ids[start:end])
            # randomly choose a prompt per chunk
            p_idx = random.randint(1, len(PROMPTS))
            prompt_text = PROMPTS[p_idx - 1]
            if args.verbose:
                print(f"[chunk] {idx}/{total_chunks} span={start}-{end} tokens={end-start} words={len(chunk_text.split())} prompt_id={p_idx}")

            dedupe_key = _sha_key(chunk_text, p_idx, model_list_str, float(args.temperature), float(args.top_p))
            if dedupe_key in dedupe_keys:
                if args.verbose:
                    print(f"[dedupe] skip chunk={idx} key={dedupe_key[:8]}...")
                continue

            # call models
            ai_outputs: Dict[str, dict] = {}
            for prov_key, actual_model in norm_models:
                if args.verbose:
                    print(f"[call] chunk={idx} provider={prov_key} model={actual_model} start")
                max_tokens_hint = max(128, int((end - start) * 1.1))
                result = call_model(
                    model_name=actual_model,
                    text=chunk_text,
                    prompt=prompt_text,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens_hint=max_tokens_hint,
                    api_keys=keys,
                    tok=tok,
                    dry_run=bool(args.dry_run),
                    timeout_sec=int(args.timeout_sec),
                    max_retries=int(args.max_retries),
                    verbose=bool(args.verbose),
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                )
                if args.verbose:
                    if result.get("error"):
                        print(f"[call] chunk={idx} provider={prov_key} model={actual_model} error={result.get('error')}")
                    else:
                        print(f"[call] chunk={idx} provider={prov_key} model={actual_model} ok latency={result.get('latency_sec')}s")
                # normalize word count
                out_text = result.get("output_text") or ""
                # key by provider family for stable schema: grok/chatgpt/deepseek/gemini
                ai_outputs[prov_key] = {
                    "output_text": out_text,
                    "word_count": len((out_text or "").split()),
                    "usage": result.get("usage") or {"input_tokens": 0, "output_tokens": 0},
                    "latency_sec": result.get("latency_sec"),
                    "model_version": result.get("model_version") or actual_model,
                    **({"error": result.get("error")} if result.get("error") else {}),
                }
                # metrics update
                if result.get("error"):
                    metrics[prov_key]["err"] += 1
                else:
                    metrics[prov_key]["ok"] += 1
                    lat = result.get("latency_sec")
                    if isinstance(lat, (int, float)):
                        metrics[prov_key]["lat"].append(float(lat))

            example = {
                "example_id": f"ex_{(existing + idx):06d}",
                "chunk_index": idx,
                "start_token": int(start),
                "end_token": int(end),
                "token_length": int(end - start),
                "overlap": int(overlap),
                "report_span": spans_meta[idx - 1],
                "prompt_id": int(p_idx),
                "prompt_text": prompt_text,
                "human": {
                    "text": chunk_text,
                    "word_count": len((chunk_text or "").split()),
                },
                "ai": ai_outputs,
                "gen_params": {
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "tokenizer_hint": args.tokenizer_hint,
                    "prompts_version": prompts_version,
                },
                "run_info": {
                    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "script_version": script_version,
                },
                "dedupe_key": dedupe_key,
            }
            line = json.dumps(example, ensure_ascii=False) + "\n"
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            if args.verbose:
                try:
                    size = os.path.getsize(args.out)
                except Exception:
                    size = 0
                print(f"[write] {args.out} +{len(line.encode('utf-8'))}B total={size}B")
                # progress/ETA
                done = idx
                rem = max(0, total_chunks - done)
                rate = done / max(1e-6, (time.time() - t0))
                eta = rem / max(1e-6, rate)
                print(f"[progress] written={done}/{total_chunks} rate={rate:.2f} ex/s eta={eta:.1f}s")
            dedupe_keys.add(dedupe_key)

    elapsed = time.time() - t0
    # Final summary
    if args.verbose:
        print("[summary] per-model results:")
        for prov, stat in metrics.items():
            lats = sorted(stat["lat"]) if stat["lat"] else []
            def pct(p: float) -> float:
                if not lats:
                    return 0.0
                k = int(max(0, min(len(lats)-1, round(p * (len(lats)-1)))))
                return lats[k]
            avg = sum(lats)/len(lats) if lats else 0.0
            print(f"  - {prov}: ok={stat['ok']} err={stat['err']} avg={avg:.2f}s p50={pct(0.5):.2f}s p95={pct(0.95):.2f}s")
    print(f"Wrote examples to {args.out} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
