"""
Dataset compiler: generation JSONL(s) -> folder with manifest, train.jsonl, valid.jsonl.
Explicit model/prompt selection; all filters logged; reproducible split.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from authinfra.datasets.schema import MANIFEST_SCHEMA_VERSION, FilterLogEntry, Manifest


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file; return list of dicts. Skips empty lines."""
    lines: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def compile_dataset(
    source_paths: list[str] | list[Path],
    output_dir: str | Path,
    *,
    dataset_name: str = "dataset",
    model_ids: list[str] | None = None,
    prompt_ids: list[str] | None = None,
    split_ratio: float = 0.9,
    split_seed: int = 0,
) -> Manifest:
    """
    Consume one or more generation JSONL files; select by model_ids and prompt_ids;
    write output_dir/manifest.json, output_dir/train.jsonl, output_dir/valid.jsonl.
    No examples silently dropped; filter_log records every exclusion reason and count.
    Split is deterministic given split_seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_set = set(model_ids) if model_ids else None
    prompt_set = set(str(p) for p in (prompt_ids or []))

    filter_counts: dict[str, int] = {
        "total_read": 0,
        "excluded_error": 0,
        "excluded_no_output": 0,
        "excluded_model_not_selected": 0,
        "excluded_prompt_not_selected": 0,
        "included": 0,
    }

    included: list[dict[str, Any]] = []

    for src in source_paths:
        path = Path(src)
        if not path.is_file():
            continue
        for row in _read_jsonl_lines(path):
            filter_counts["total_read"] += 1
            if row.get("error") is not None and row.get("error") != "":
                filter_counts["excluded_error"] += 1
                continue
            if not row.get("output_text"):
                filter_counts["excluded_no_output"] += 1
                continue
            if model_set is not None and (row.get("model_id") or "") not in model_set:
                filter_counts["excluded_model_not_selected"] += 1
                continue
            if prompt_set and (str(row.get("prompt_id") or "") not in prompt_set):
                filter_counts["excluded_prompt_not_selected"] += 1
                continue
            filter_counts["included"] += 1
            included.append(row)

    # Deterministic split
    rng = random.Random(split_seed)
    ordered = sorted(included, key=lambda r: (r.get("job_id", ""), r.get("chunk_index", 0)))
    rng.shuffle(ordered)
    n = len(ordered)
    train_n = max(0, int(round(n * split_ratio)))
    valid_n = n - train_n
    train_rows = ordered[:train_n]
    valid_rows = ordered[train_n:]

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(valid_path, "w", encoding="utf-8") as f:
        for row in valid_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    filter_log: list[FilterLogEntry] = [
        {"reason": k, "count": v} for k, v in sorted(filter_counts.items())
    ]

    manifest: Manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": [str(Path(p).resolve()) for p in source_paths],
        "model_ids": sorted(model_ids) if model_ids else [],
        "prompt_ids": sorted(prompt_set) if prompt_set else [],
        "filter_log": filter_log,
        "train_count": train_n,
        "valid_count": valid_n,
        "train_path": str(train_path.resolve()),
        "valid_path": str(valid_path.resolve()),
        "split_ratio": split_ratio,
        "split_seed": split_seed,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_manifest(dataset_dir: str | Path) -> Manifest | None:
    """Load manifest.json from a compiled dataset folder. Returns None if missing."""
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_summary_counts(dataset_dir: str | Path) -> dict[str, int] | None:
    """Return train_count, valid_count, total from manifest. None if no manifest."""
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        return None
    train = manifest.get("train_count", 0)
    valid = manifest.get("valid_count", 0)
    return {"train_count": train, "valid_count": valid, "total": train + valid}
