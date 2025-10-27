"""Clean CRS JSONs into CSVs. Minimal CLI."""

import argparse
import csv
import json
import os
import time

from bs4 import BeautifulSoup

try:
    # Package import path
    from . import crs_scraper  # type: ignore
except Exception:  # pragma: no cover
    # Script fallback
    import crs_scraper  # type: ignore


DEFAULT_JSON_DIR = os.path.join(os.path.dirname(__file__), "jsons")
DEFAULT_BASE_URL = "https://www.everycrsreport.com/reports.csv"
CHUNK_SIZE = 50


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _list_json_files(json_dir: str) -> list:
    files = []
    if not os.path.isdir(json_dir):
        return files
    for name in os.listdir(json_dir):
        if name.startswith("report_") and name.endswith(".json"):
            files.append(os.path.join(json_dir, name))
    # Sort by report_<idx>.json
    def _index_key(p: str) -> int:
        base = os.path.basename(p)
        # Parse number between prefixes
        try:
            num_part = base[len("report_") : -len(".json")]
            return int(num_part)
        except Exception:
            return 10**9

    files.sort(key=_index_key)
    return files


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                return {"data": data}
    except Exception as e:
        print(f"[filter] Failed to read {path}: {e}")
        return {}


def _find_first_string(d, keys) -> str:
    """First string under any of the keys."""
    if isinstance(d, dict):
        for k, v in d.items():
            lk = (k or "").lower()
            if lk in keys:
                if isinstance(v, str):
                    return v
            # Recurse
            res = _find_first_string(v, keys)
            if isinstance(res, str) and res:
                return res
    elif isinstance(d, list):
        for it in d:
            res = _find_first_string(it, keys)
            if isinstance(res, str) and res:
                return res
    return ""


def _find_all_strings(d, keys) -> list:
    """All strings under any of the keys."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            lk = (k or "").lower()
            if lk in keys and isinstance(v, str):
                out.append(v)
            out.extend(_find_all_strings(v, keys))
    elif isinstance(d, list):
        for it in d:
            out.extend(_find_all_strings(it, keys))
    return out


def _clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    # Drop tables
    for tag in soup.find_all(["table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption"]):
        try:
            tag.decompose()
        except Exception:
            continue
    # keep paragraph breaks
    cleaned = soup.get_text(separator="\n", strip=True)
    # Drop boilerplate
    for phrase in (
        "Source: Congressional Research Service,",
        "Source: Congressional Research Service",
        "Congressional Research Service",
        "CRS Report",
    ):
        while phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
    # normalize but retain blank lines
    cleaned = cleaned.replace("\r", "\n")
    lines = [ln.strip() for ln in cleaned.split("\n")]
    out_lines = []
    blank = 0
    for ln in lines:
        if ln:
            out_lines.append(ln)
            blank = 0
        else:
            if blank == 0:
                out_lines.append("")
            blank = 1
    cleaned = "\n".join(out_lines)
    return cleaned.strip()


def _extract_fields(data: dict) -> tuple:
    """Get (title, date, text)."""
    # Title
    title = _find_first_string(data, {"title", "short_title"})
    if not title:
        # EveryCRSReport usually provides versions[0].title
        title = _find_first_string(data, {"name"})

    # Date
    date = _find_first_string(
        data,
        {
            "date",
            "published",
            "published_at",
            "latestpubdate",
            "publication_date",
            "update_date",
        },
    )

    # Text
    text_candidates = _find_all_strings(
        data,
        {"text", "body", "content", "html", "summary", "document", "description"},
    )
    # Keep unique order
    seen = set()
    unique_texts = []
    for t in text_candidates:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    # Clean + join (preserve sections)
    cleaned_parts = [_clean_html(t) for t in unique_texts if t]
    text = "\n\n".join([p for p in cleaned_parts if p])
    text = text.strip()

    return title or "", date or "", text or ""


def _save_csv_chunk(records: list, out_dir: str, chunk_index: int) -> str:
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"clean_{chunk_index}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "date", "text", "word_count"],
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    return out_path


def process_jsons(
    n: int,
    json_dir: str,
    out_dir: str,
    min_words: int = 0,
    target_tokens: int = 0,
) -> tuple[int, int, int]:
    """Process JSONs, keep rows >= min_words.

    Stops when either kept == n (if n>0) or total_words >= target_tokens (if target_tokens>0).
    Returns (processed, kept, total_words).
    """
    files = _list_json_files(json_dir)
    if not files:
        print("[filter] No JSON files found to process.")
        return (0, 0, 0)

    to_process = files
    records = []
    total_words = 0
    processed = 0
    kept = 0

    for idx, path in enumerate(to_process, start=1):
        data = _read_json(path)
        title, date, text = _extract_fields(data)
        word_count = len(text.split()) if text else 0
        rec = {
            "id": os.path.basename(path),
            "title": title,
            "date": date,
            "text": text,
            "word_count": word_count,
        }
        processed += 1
        if word_count >= max(0, int(min_words)):
            records.append(rec)
            total_words += word_count
            kept += 1

        # Save chunk when reaching CHUNK_SIZE
        stop_by_n = n and kept == n
        stop_by_tokens = target_tokens and total_words >= target_tokens
        if len(records) == CHUNK_SIZE or stop_by_n or stop_by_tokens:
            chunk_idx = (processed - 1) // CHUNK_SIZE
            out_path = _save_csv_chunk(records, out_dir, chunk_idx)
            print(f"[filter] Saved {out_path} ({len(records)} rows)")
            records = []
            if stop_by_n or stop_by_tokens:
                break

    # Save remaining records
    if records and (not n or kept < n) and (not target_tokens or total_words < target_tokens):
        chunk_idx = (processed - 1) // CHUNK_SIZE
        out_path = _save_csv_chunk(records, out_dir, chunk_idx)
        print(f"[filter] Saved {out_path} ({len(records)} rows)")

    print(f"[filter] Summary: processed={processed}, kept={kept}, total_words={total_words}")
    return (processed, kept, total_words)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean CRS JSONs into ML-ready CSVs.")
    p.add_argument("--n", type=int, default=0, help="Target rows to keep (after filtering)")
    p.add_argument("--target_tokens", type=int, default=0, help="Stop when total kept tokens (words) reach this")
    p.add_argument("--out", default="./clean_data", help="Output directory for CSV files")
    p.add_argument(
        "--json_dir",
        default=DEFAULT_JSON_DIR,
        help="Directory containing report_*.json (defaults to detector/jsons)",
    )
    p.add_argument(
        "--base",
        default=DEFAULT_BASE_URL,
        help="Base listing URL used to fetch JSONs if missing",
    )
    p.add_argument("--min_words", type=int, default=0, help="Minimum word count to keep a report")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Ensure JSONs exist; if none, fetch automatically
    existing = _list_json_files(args.json_dir)
    if not existing:
        first_batch = max(args.n, 50)
        print(f"[filter] No JSONs found in {args.json_dir}. Fetching {first_batch} using base={args.base}...")
        crs_scraper.fetch_multiple(base_url=args.base, n=first_batch, dest_dir=args.json_dir)
    else:
        print(f"[filter] Found {len(existing)} JSONs in {args.json_dir}; proceeding.")

    start = time.time()
    if not args.n and not args.target_tokens:
        raise SystemExit("please provide --n or --target_tokens")

    processed, kept, total_words = process_jsons(
        n=args.n,
        json_dir=args.json_dir,
        out_dir=args.out,
        min_words=args.min_words,
        target_tokens=args.target_tokens,
    )

    attempts = 0
    def _need_more() -> bool:
        need_n = args.n and kept < args.n
        need_tok = args.target_tokens and total_words < args.target_tokens
        return (need_n or need_tok)

    while _need_more() and attempts < 5 and args.base:
        # heuristic: fetch more based on how far we are
        missing_rows = max(0, (args.n - kept)) if args.n else 0
        missing_tokens = max(0, (args.target_tokens - total_words)) if args.target_tokens else 0
        batch = 100
        if missing_rows:
            batch = max(batch, missing_rows * 3)
        if missing_tokens:
            # assume ~1200 words/report avg when chasing targets
            batch = max(batch, missing_tokens // 1200)
        print(f"[filter] Need more data (kept={kept}, tokens={total_words}). Fetching +{batch}...")
        crs_scraper.fetch_multiple(base_url=args.base, n=batch, dest_dir=args.json_dir)
        processed, kept, total_words = process_jsons(
            n=args.n,
            json_dir=args.json_dir,
            out_dir=args.out,
            min_words=args.min_words,
            target_tokens=args.target_tokens,
        )
        attempts += 1

    elapsed = time.time() - start
    goal = (
        f"rows {kept}/{args.n}" if args.n else f"tokens {total_words}/{args.target_tokens}"
    )
    print(f"[filter] Done in {elapsed:.2f}s. Met goal: {goal}")
