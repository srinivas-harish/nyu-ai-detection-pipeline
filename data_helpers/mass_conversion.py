"""Mass conversion + sanity check for CRS data.

Features
- Sanity-only mode: count reports, tokens (tries real tokenizer), above-threshold items.
- Generation loop: iterate reports (CSV rows), split into sections, call models via api_runner.
- Saves JSONL per model with enough metadata to reconstruct.

Hardcoded prompts: PROMPTS[1..10]. Select with --prompt_id.

Tokenization
- Tries tiktoken (if installed) for OpenAI-like counts.
- Falls back to a simple whitespace token count.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import time
from typing import List, Tuple

try:
    # Optional; not required to run
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    # Local import
    from .api_runner import run_api  # type: ignore
except Exception:  # pragma: no cover
    from api_runner import run_api  # type: ignore


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


def _load_csv_rows(input_dir: str) -> List[dict]:
    paths = sorted(glob.glob(os.path.join(input_dir, "clean_*.csv")))
    rows: List[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _get_tokenizer(model_hint: str):
    # Try tiktoken for openai-like models
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_hint)
            return lambda s: len(enc.encode(s))
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return lambda s: len(enc.encode(s))
            except Exception:
                pass
    # Fallback: whitespace tokens as approximation
    return lambda s: len((s or "").split())


def _truncate_to_tokens(text: str, max_tokens: int, model_hint: str = "gpt-4o-mini") -> str:
    """Return a prefix of text with at most max_tokens tokens (best-effort).
    Uses tiktoken if available; else falls back to word-slice.
    """
    s = text or ""
    if max_tokens <= 0:
        return ""
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
        if enc is not None:
            ids = enc.encode(s)
            if len(ids) <= max_tokens:
                return s
            return enc.decode(ids[:max_tokens])
    # fallback: words
    words = s.split()
    if len(words) <= max_tokens:
        return s
    return " ".join(words[:max_tokens])


def _encode(text: str, model_hint: str):
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
        if enc is not None:
            return enc, enc.encode(text or "")
    # fallback: word list as tokens
    words = (text or "").split()
    return None, words


def _decode(enc, ids):
    if enc is not None:
        return enc.decode(ids)
    return " ".join(ids)

def _last_tokens(text: str, max_tokens: int, model_hint: str) -> str:
    """Return the last max_tokens worth of tokens from text."""
    if max_tokens <= 0:
        return ""
    enc, ids = _encode(text or "", model_hint)
    if not ids:
        return ""
    if len(ids) <= max_tokens:
        return text or ""
    tail = ids[-max_tokens:]
    return _decode(enc, tail)


def _chunk_by_tokens(text: str, chunk_tokens: int, overlap: int, model_hint: str) -> List[str]:
    enc, ids = _encode(text, model_hint)
    if not ids:
        return []
    out = []
    start = 0
    step = max(1, chunk_tokens - max(0, overlap))
    L = len(ids)
    while start < L:
        end = min(L, start + chunk_tokens)
        out.append(_decode(enc, ids[start:end]))
        if end == L:
            break
        start = end - max(0, overlap)
    return out


_HEADING_RE = re.compile(
    r"^(?:"  # any of:
    r"[A-Z][A-Z\s\-/&\d]{3,}"  # ALL CAPS or mixed caps words
    r"|(?:\d+(?:\.\d+)*)\s+[A-Z][^\n]{2,}"  # 1., 1.1., etc. followed by Title Case
    r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,6}"  # Title Case with 2-7 words
    r")$"
)


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split by candidate headings and blank lines; robust fallback without HTML."""
    blocks = (text or "").replace("\r", "\n").split("\n\n")
    out: List[Tuple[str, str]] = []
    buf_title: str | None = None
    buf_body: list[str] = []
    for blk in blocks:
        lines = [ln.strip() for ln in blk.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        is_heading = (
            len(lines) == 1
            and _HEADING_RE.match(lines[0]) is not None
            and lines[0][-1] not in ".;:!?"
        )
        if is_heading:
            if buf_title is not None or buf_body:
                out.append((buf_title or "", "\n".join(buf_body).strip()))
                buf_body = []
            buf_title = lines[0]
        else:
            buf_body.append("\n".join(lines))
    if buf_title is not None or buf_body:
        out.append((buf_title or "", "\n".join(buf_body).strip()))
    # Merge tiny bodies
    merged: List[Tuple[str, str]] = []
    for t, b in out:
        if merged and len((b or "").split()) < 50:
            pt, pb = merged[-1]
            merged[-1] = (pt, (pb + "\n\n" + b).strip())
        else:
            merged.append((t, b))
    return merged


def _subsplit_long_sections(sections: List[Tuple[str, str]], max_words: int = 1200, overlap: int = 64) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for title, body in sections:
        words = (body or "").split()
        if len(words) <= max_words:
            out.append((title, body))
            continue
        step = max_words - overlap if max_words > overlap else max_words
        start = 0
        idx = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            chunk = " ".join(words[start:end])
            ctitle = title if idx == 0 else f"{title} (part {idx+1})" if title else f"Section (part {idx+1})"
            out.append((ctitle, chunk))
            if end == len(words):
                break
            start = end - overlap
            idx += 1
    return out


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _already_done(outfile: str, key: str) -> bool:
    if not os.path.exists(outfile):
        return False
    try:
        with open(outfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("key") == key:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


def _reconstruct_report(model: str, out_dir: str, report_id: str) -> None:
    path = os.path.join(out_dir, f"{model.replace('/', '_')}.jsonl")
    if not os.path.exists(path):
        return
    secs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("report_id") == report_id:
                secs.append(rec)
    if not secs:
        return
    secs.sort(key=lambda r: int(r.get("section_index", 0)))
    full: list[str] = []
    last_title: str | None = None
    for r in secs:
        title = (r.get("section_title") or "").split(" (part")[0].strip()
        if title and title != last_title:
            full.append(title + "\n")
            last_title = title
        full.append((r.get("output_text") or "").rstrip() + "\n\n")
    full_text = "".join(full).strip()
    out = {
        "report_id": report_id,
        "model": model,
        "sections": len(secs),
        "text": full_text,
    }
    with open(os.path.join(out_dir, f"{report_id}__{model.replace('/', '_')}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def sanity_check(input_csv_dir: str, model_hint: str = "gpt-4o-mini", min_words: int = 0) -> dict:
    rows = _load_csv_rows(input_csv_dir)
    tok = _get_tokenizer(model_hint)
    n_reports = len(rows)
    total_tokens = 0
    above = 0
    for r in rows:
        text = r.get("text") or ""
        wc = len(text.split())
        if wc >= min_words:
            above += 1
        total_tokens += tok(text)
    info = {
        "reports": n_reports,
        "min_words": int(min_words),
        "above_min": above,
        "total_tokens": int(total_tokens),
    }
    print(json.dumps(info, indent=2))
    return info


def generate(
    input_csv_dir: str,
    out_dir: str,
    models: List[str],
    keys: str,
    prompt_id: int,
    min_words: int,
    target_tokens: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_csv_rows(input_csv_dir)
    tok = _get_tokenizer(models[0] if models else "gpt-4o-mini")
    prompt = PROMPTS[max(1, min(10, prompt_id)) - 1]
    kept_tokens = 0
    temperature = 0.4
    top_p = 0.9
    max_retries = 5
    for row in rows:
        text = row.get("text") or ""
        if len(text.split()) < int(min_words):
            continue
        report_id = (row.get("id") or "").replace(".json", "")
        title = row.get("title") or ""
        date = row.get("date") or ""
        sections = _split_sections(text)
        sections = _subsplit_long_sections(sections, max_words=1200, overlap=64)
        for idx, (sec_title, sec_body) in enumerate(sections):
            if not sec_body:
                continue
            src_wc = len(sec_body.split())
            # token-ish output cap based on source tokens (+5%)
            src_tok = max(1, tok(sec_body))
            max_out = max(128, int(src_tok * 1.05))
            for model in models:
                payload = (
                    f"{prompt}\n\nTitle: {sec_title}\n\n" if sec_title else f"{prompt}\n\n"
                ) + sec_body
                key = _hash_key(report_id, str(idx), model, str(prompt_id))
                outfile = os.path.join(out_dir, f"{model.replace('/', '_')}.jsonl")
                if _already_done(outfile, key):
                    continue
                # call
                start = time.time()
                out_text = ""
                for attempt in range(max_retries):
                    try:
                        out_text = run_api(model, payload, max_output=max_out, keys_path=keys)
                        break
                    except Exception as e:
                        out_text = f"[error] {e}"
                        time.sleep(min(30, 2 ** attempt))
                elapsed = time.time() - start
                rec = {
                    "report_id": report_id,
                    "title": title,
                    "date": date,
                    "section_index": idx,
                    "section_title": sec_title,
                    "model": model,
                    "prompt_id": prompt_id,
                    "source_word_count": src_wc,
                    "source_token_est": src_tok,
                    "output_word_count": len((out_text or "").split()),
                    "output_token_est": tok(out_text or ""),
                    "output_text": out_text,
                    "elapsed_sec": round(elapsed, 3),
                    "temperature": temperature,
                    "top_p": top_p,
                    "key": key,
                }
                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept_tokens += tok(sec_body)
            if target_tokens and kept_tokens >= target_tokens:
                print(f"[mass] Reached target_tokens={target_tokens}.")
                # reconstruct per model for this last report
                for model in models:
                    _reconstruct_report(model, out_dir, report_id)
                return
        # after finishing a report for all sections, reconstruct per model
        for model in models:
            _reconstruct_report(model, out_dir, report_id)


def first_tokens_csv(
    input_csv_dir: str,
    out_dir: str,
    first_tokens: int,
    min_words: int = 3000,
    models: List[str] | None = None,
    keys: str | None = None,
    sanity_model_hint: str = "gpt-4o-mini",
    prompt_id: int = 2,
) -> None:
    """Build two CSVs: original first N tokens, and AI-modified ones per model.

    - Filters source rows by min_words.
    - Concatenates eligible rows until reaching first_tokens; truncates the last chunk if needed.
    - Writes original CSV: original_first_{N}.csv
    - If models provided, writes ai_first_{N}__{model}.csv for each model.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_csv_rows(input_csv_dir)
    tok = _get_tokenizer(sanity_model_hint)
    prompt = PROMPTS[max(1, min(10, prompt_id)) - 1]
    budget = int(first_tokens)

    # 1) Build a single concatenated original text up to N tokens
    orig_parts: List[str] = []
    taken_meta: List[dict] = []
    for row in rows:
        if budget <= 0:
            break
        text = row.get("text") or ""
        if len(text.split()) < int(min_words):
            continue
        # take what we can from this report
        remain = budget
        piece = _truncate_to_tokens(text, remain, sanity_model_hint)
        got = tok(piece)
        if got <= 0:
            continue
        orig_parts.append(piece)
        taken_meta.append({
            "report_id": (row.get("id") or "").replace(".json", ""),
            "title": row.get("title") or "",
            "date": row.get("date") or "",
            "token_est": got,
        })
        budget -= got

    orig_concat = ("\n\n".join(orig_parts)).strip()
    total_tokens = tok(orig_concat)

    # write original as a single row
    orig_path = os.path.join(out_dir, f"original_first_{first_tokens}.csv")
    with open(orig_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "title", "date", "text", "token_est"],
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        # summarize provenance from first and last items
        rep = taken_meta[0]["report_id"] + (".." + taken_meta[-1]["report_id"] if len(taken_meta) > 1 else "") if taken_meta else ""
        tit = taken_meta[0]["title"] if taken_meta else ""
        dat = taken_meta[0]["date"] if taken_meta else ""
        writer.writerow({
            "report_id": rep,
            "title": tit,
            "date": dat,
            "text": orig_concat,
            "token_est": total_tokens,
        })
    # Intentionally quiet: avoid verbose terminal output

    # if no models, stop here
    models = models or []
    if not models:
        return

    # 2) Chunk the original text and generate per-chunk, then stitch
    # Use larger chunks to reduce under-generation and API overhead.
    # Keep a modest overlap to preserve continuity.
    # If first_tokens is small, keep chunk at most that size.
    chunk_tokens = max(1024, min(int(first_tokens) if first_tokens else 2048, 3072))
    overlap_tokens = 128
    chunks = _chunk_by_tokens(orig_concat, chunk_tokens, overlap_tokens, sanity_model_hint)

    for model in models:
        out_parts: List[str] = []
        ai_so_far = 0
        for ch in chunks:
            src_tok = tok(ch)
            # Aim to at least match the source chunk length.
            # Some providers impose caps; they will truncate if needed.
            max_out = max(512, int(src_tok))
            payload = f"{prompt}\n\n" + ch
            try:
                out_text = run_api(model, payload, max_output=max_out, keys_path=keys, quiet=True)
            except Exception as e:
                out_text = f"[error] {e}"
            out_parts.append((out_text or "").strip())
            ai_so_far += tok(out_text or "")
            print(f"[progress] tokens≈{ai_so_far}/{total_tokens}", end="\r", flush=True)
        ai_concat = ("\n\n".join(out_parts)).strip()
        # Force-match the token budget of the original by truncation if necessary.
        ai_tokens = tok(ai_concat)
        if ai_tokens > total_tokens:
            ai_concat = _truncate_to_tokens(ai_concat, total_tokens, sanity_model_hint)
            ai_tokens = tok(ai_concat)

        # If still short, top up by asking the model to continue until we reach the budget.
        max_continuations = 50
        while ai_tokens < total_tokens and max_continuations > 0:
            need = total_tokens - ai_tokens
            tail_ctx = _last_tokens(ai_concat, 400, sanity_model_hint) if ai_concat else ""
            cont_prompt = (
                "Continue the rewritten text in the same style and structure. "
                "Do not repeat any content already written. Continue from here:\n\n" + tail_ctx
            )
            max_out = min(1024, need + 256)
            try:
                extra = run_api(model, cont_prompt, max_output=max_out, keys_path=keys, quiet=True)
            except Exception as e:
                extra = f"[error] {e}"
            if not extra:
                break
            ai_concat = (ai_concat + "\n\n" + extra.strip()).strip()
            ai_tokens = tok(ai_concat)
            ai_so_far = ai_tokens
            print(f"[progress] tokens≈{ai_so_far}/{total_tokens}", end="\r", flush=True)
            max_continuations -= 1

        # Final truncate to exact budget, if needed
        if ai_tokens > total_tokens:
            ai_concat = _truncate_to_tokens(ai_concat, total_tokens, sanity_model_hint)
            ai_tokens = tok(ai_concat)
        # Finish progress line with a newline
        print(f"[progress] tokens≈{ai_tokens}/{total_tokens}")
        ai_path = os.path.join(out_dir, f"ai_first_{first_tokens}__{model.replace('/', '_')}.csv")
        with open(ai_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["report_id", "title", "date", "text", "token_est"],
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow({
                "report_id": rep,
                "title": tit,
                "date": dat,
                "text": ai_concat,
                "token_est": ai_tokens,
            })
        # Quiet: avoid additional terminal prints


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CRS mass conversion + sanity check")
    p.add_argument("--input_csv_dir", required=True, help="Folder with clean_*.csv files")
    p.add_argument("--out", required=False, default="./gen_out", help="Output folder for JSONL/CSV")
    p.add_argument("--models", required=False, default="", help="Comma-separated model ids (empty = originals only)")
    p.add_argument("--keys", required=False, default=os.path.join(os.path.dirname(__file__), "api_keys.txt"), help="Path to api_keys.txt")
    p.add_argument("--prompt_id", type=int, default=2, help="Pick 1..10 prompt variant")
    p.add_argument("--min_words", type=int, default=3000, help="Minimum words per report/section")
    p.add_argument("--target_tokens", type=int, default=0, help="Stop after this many source tokens (approx)")
    p.add_argument("--sanity_only", action="store_true", help="Run only sanity check and exit")
    p.add_argument("--sanity_model", default="gpt-4o-mini", help="Tokenizer hint for sanity counts")
    p.add_argument("--first_tokens", type=int, default=0, help="Build first-N tokens CSVs (original and AI)")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.sanity_only:
        sanity_check(args.input_csv_dir, model_hint=args.sanity_model, min_words=args.min_words)
    else:
        models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
        # first-tokens mode
        if args.first_tokens and args.first_tokens > 0:
            first_tokens_csv(
                input_csv_dir=args.input_csv_dir,
                out_dir=args.out,
                first_tokens=args.first_tokens,
                min_words=args.min_words,
                models=models,
                keys=args.keys,
                sanity_model_hint=args.sanity_model,
                prompt_id=args.prompt_id,
            )
        else:
            if not models:
                raise SystemExit("please supply at least one model via --models, or use --first_tokens for CSV export")
            # sanity first
            sanity_check(args.input_csv_dir, model_hint=args.sanity_model, min_words=args.min_words)
            generate(
                input_csv_dir=args.input_csv_dir,
                out_dir=args.out,
                models=models,
                keys=args.keys,
                prompt_id=args.prompt_id,
                min_words=args.min_words,
                target_tokens=args.target_tokens,
            )
