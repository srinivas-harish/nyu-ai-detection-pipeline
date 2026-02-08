"""
Generation job runner: human text -> prompt + model -> JSONL artifact.
Errors captured explicitly per line. Dry-run produces placeholders.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from authinfra.generation.adapters import BaseAdapter, DryRunAdapter, get_adapter
from authinfra.generation.chunking import chunk_text
from authinfra.generation.prompts import get_prompt, get_registry_version
from authinfra.generation.schema import JSONLLine


def run_generation(
    input_text: str,
    prompt_id: str,
    model_name: str,
    output_path: str | Path,
    *,
    min_tokens: int = 300,
    max_tokens: int = 1000,
    overlap_tokens: int = 32,
    dry_run: bool = True,
    job_id: str | None = None,
) -> tuple[int, int]:
    """
    Chunk input_text, run adapter per chunk, write one JSONL line per chunk.
    Returns (lines_written, error_count).
    Errors are written into each line's "error" field; no exception raised for partial failure.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    job_id = job_id or str(uuid.uuid4())[:8]
    ts = datetime.now(timezone.utc).isoformat()

    prompt_entry = get_prompt(prompt_id)
    if prompt_entry is None:
        # Single failure line
        line: JSONLLine = {
            "job_id": job_id,
            "timestamp_utc": ts,
            "chunk_index": 0,
            "prompt_id": prompt_id,
            "prompt_version": get_registry_version(),
            "prompt_text": "",
            "model_id": model_name,
            "input_token_count": 0,
            "output_text": None,
            "output_token_count": None,
            "runtime_sec": None,
            "error": f"unknown prompt_id: {prompt_id}",
            "dry_run": dry_run,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return 1, 1

    adapter: BaseAdapter = DryRunAdapter(model_id=model_name) if dry_run else get_adapter(model_name)

    chunks = chunk_text(input_text, min_tokens, max_tokens, overlap_tokens)
    if not chunks:
        line0: JSONLLine = {
            "job_id": job_id,
            "timestamp_utc": ts,
            "chunk_index": 0,
            "prompt_id": prompt_id,
            "prompt_version": prompt_entry["version"],
            "prompt_text": prompt_entry["text"],
            "model_id": adapter.model_id,
            "input_token_count": 0,
            "output_text": None,
            "output_token_count": None,
            "runtime_sec": None,
            "error": "no chunks produced (empty or too short input)",
            "dry_run": dry_run,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line0, ensure_ascii=False) + "\n")
        return 1, 1

    written = 0
    errors = 0
    prompt_text = prompt_entry["text"]

    with open(path, "a", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            full_prompt = f"{prompt_text}\n\n---\n\n{ch['text']}"
            result = adapter.generate(prompt=prompt_text, text=ch["text"])

            err = result.get("error")
            if err:
                errors += 1
            out_line: JSONLLine = {
                "job_id": job_id,
                "timestamp_utc": ts,
                "chunk_index": i,
                "prompt_id": prompt_id,
                "prompt_version": prompt_entry["version"],
                "prompt_text": prompt_text,
                "model_id": result.get("model_id") or adapter.model_id,
                "input_token_count": result.get("input_token_count") or ch.get("token_count", 0),
                "output_text": result.get("output_text") if not err else None,
                "output_token_count": result.get("output_token_count"),
                "runtime_sec": result.get("runtime_sec"),
                "error": err,
                "dry_run": dry_run,
            }
            f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            written += 1

    return written, errors
