"""Fetch CRS report JSONs. Simple CLI + import."""

import argparse
import csv
import json
import os
import time

import requests


def fetch_json(url: str) -> dict:
    """GET JSON or {}."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        parsed = resp.json()
        if isinstance(parsed, dict):
            return parsed
        else:
            return {"data": parsed}
    except requests.exceptions.RequestException as e:
        print(f"[fetch_json] network-error: {e}")
        return {}
    except ValueError:
        print("[fetch_json] parse-error: Non-JSON response")
        return {}


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _next_start_index(dest_dir: str) -> int:
    """Find next report index to write (1-based)."""
    max_idx = 0
    if os.path.isdir(dest_dir):
        for name in os.listdir(dest_dir):
            if name.startswith("report_") and name.endswith(".json"):
                try:
                    idx = int(name[len("report_"):-len(".json")])
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
    return max_idx + 1


def _download_from_everycrsreport(n: int, dest_dir: str, csv_url: str = "https://www.everycrsreport.com/reports.csv") -> int:
    """Use EveryCRSReport CSV to save n JSONs."""
    print(f"[scraper] Using EveryCRSReport listing: {csv_url}")
    try:
        resp = requests.get(csv_url, timeout=60)
        resp.raise_for_status()
        content = resp.text
    except requests.exceptions.RequestException as e:
        print(f"[scraper] Failed to fetch CSV listing: {e}")
        return 0

    lines = content.splitlines()
    if not lines:
        print("[scraper] Empty CSV listing")
        return 0

    reader = csv.DictReader(lines)
    base = "https://www.everycrsreport.com/"
    saved = 0
    start_idx = _next_start_index(dest_dir)

    for i, row in enumerate(reader, start=1):
        if saved >= n:
            break
        json_path = (row.get("url") or "").strip()
        if not json_path or not json_path.endswith(".json"):
            continue
        full_url = base + json_path
        data = fetch_json(full_url)
        if not data:
            continue
        out_path = os.path.join(dest_dir, f"report_{start_idx + saved}.json")
        _save_json(data, out_path)
        saved += 1
        title = row.get("title") or data.get("title") or ""
        print(f"[scraper] Saved {out_path} | title='{title}'")
    return saved


def _extract_list_from_listing(payload: dict) -> list:
    """Pull a list out of common keys."""
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "items"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    if isinstance(payload.get("reports"), list):  # a few APIs use this key
        return payload.get("reports")
    # If the payload itself looks like an item, treat it as single-item list
    return [payload]


def _download_from_crs_api(base_url: str, n: int, dest_dir: str) -> int:
    """Paged fetch + save n items."""
    saved = 0
    start_idx = _next_start_index(dest_dir)
    page = 1
    print(f"[scraper] Using CRS API: {base_url}")
    while saved < n:
        try:
            resp = requests.get(base_url, params={"page": page}, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[scraper] network-error on page {page}: {e}")
            break
        except ValueError:
            print(f"[scraper] parse-error: Non-JSON response on page {page}")
            break

        items = _extract_list_from_listing(payload)
        if not items:
            print(f"[scraper] No items on page {page}; stopping.")
            break

        for item in items:
            if saved >= n:
                break
            # if item is not a dict, wrap to be JSON-serializable dict
            if not isinstance(item, dict):
                item = {"data": item}
            out_path = os.path.join(dest_dir, f"report_{start_idx + saved}.json")
            _save_json(item, out_path)
            saved += 1
            title = item.get("title") or ""
            print(f"[scraper] Saved {out_path} | title='{title}'")

        page += 1
        time.sleep(0.2)  # be polite to the API

    return saved


def fetch_multiple(base_url: str, n: int, dest_dir: str) -> None:
    """Save n reports to dest_dir."""
    _ensure_dir(dest_dir)
    start = time.time()

    use_everycrs = ("everycrsreport.com" in (base_url or "")) or (base_url or "").endswith(".csv")
    if use_everycrs:
        saved = _download_from_everycrsreport(n=n, dest_dir=dest_dir, csv_url=base_url if base_url else "https://www.everycrsreport.com/reports.csv")
    else:
        saved = _download_from_crs_api(base_url=base_url, n=n, dest_dir=dest_dir)

    elapsed = time.time() - start
    print(f"[scraper] Done. Saved {saved} report JSON files in {elapsed:.2f}s.")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download CRS report JSONs from public APIs.")
    p.add_argument("--base", required=True, help="Base listing URL (CRS API or EveryCRSReport CSV)")
    p.add_argument("--n", type=int, required=True, help="Number of report JSONs to fetch")
    p.add_argument("--out", default="./jsons", help="Destination directory to save JSON files")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    fetch_multiple(args.base, args.n, args.out)
