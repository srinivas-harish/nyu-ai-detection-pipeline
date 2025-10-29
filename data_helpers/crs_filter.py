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
    print(f"Listing JSON files in {json_dir}...")
    if not os.path.isdir(json_dir):
        return files
    for name in os.listdir(json_dir):
        if name.startswith("report_") and name.endswith(".json"):
            print(f"files is {name}")
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
    cleaned = soup.get_text(separator=" ", strip=True)
    # Drop boilerplate
    for phrase in (
        "Source: Congressional Research Service,",
        "Source: Congressional Research Service",
        "Congressional Research Service",
        "CRS Report",
    ):
        while phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
    # Normalize whitespace (crush newlines/tabs)
    cleaned = " ".join(cleaned.split())
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
        {"text", "body", "content", "html", "summary", "document", "description", "extracted_text"},
    )
    # Keep unique order
    seen = set()
    unique_texts = []
    for t in text_candidates:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    # Clean + join
    cleaned_parts = [_clean_html(t) for t in unique_texts if t]
    text = " ".join([p for p in cleaned_parts if p])
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


def process_jsons( json_dir: str, out_dir: str,n) -> None:
    """Process up to n JSON files into CSV chunks."""
    files = _list_json_files(json_dir)
    
    if not files:
        print("[filter] No JSON files found to process.")
        return

    to_process = files[:n]
    records = []
    total_words = 0
    processed = 0

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
        records.append(rec)
        total_words += word_count
        processed += 1

        # Save chunk when reaching CHUNK_SIZE
        if len(records) == CHUNK_SIZE:
            chunk_idx = (processed - 1) // CHUNK_SIZE
            out_path = _save_csv_chunk(records, out_dir, chunk_idx)
            print(f"[filter] Saved {out_path} ({len(records)} rows)")
            records = []

    # Save remaining records
    if records:
        chunk_idx = (processed - 1) // CHUNK_SIZE
        out_path = _save_csv_chunk(records, out_dir, chunk_idx)
        print(f"[filter] Saved {out_path} ({len(records)} rows)")

    print(f"[filter] Summary: processed={processed}, total_words={total_words}")
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean CRS JSONs into ML-ready CSVs.")
    p.add_argument("--n", type=int, required=True, help="Number of JSON files to process")
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
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Ensure JSONs exist; if none, fetch automatically
    existing = _list_json_files(args.json_dir)
    if not existing:
        print(
            f"[filter] No JSONs found in {args.json_dir}. Fetching {args.n} using base={args.base}..."
        )
        crs_scraper.fetch_multiple(base_url=args.base, n=args.n, dest_dir=args.json_dir)
    else:
        print(f"[filter] Found {len(existing)} JSONs in {args.json_dir}; proceeding to clean.")

    start = time.time()
    process_jsons(n=args.n, json_dir=args.json_dir, out_dir=args.out)
    elapsed = time.time() - start
    print(f"[filter] Done in {elapsed:.2f}s.")
