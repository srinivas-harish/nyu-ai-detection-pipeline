"""
Minimal worker: picks up pending jobs from job store, runs authinfra generation or compile.
Writes artifacts to filesystem. Structured logs per job stage. No refactor of authinfra.
"""

import logging
import sys
import time
from pathlib import Path

# Repo root for authinfra and job_store
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.api.job_store import (
    ARTIFACTS_ROOT,
    get_job,
    list_pending_jobs,
    update_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("worker")


def run_generation_job(job_id: str, params: dict) -> None:
    """Execute generation; update job with result or error."""
    update_job(job_id, status="running")
    logger.info("job_start", extra={"job_id": job_id, "type": "generation", "stage": "running"})

    output_dir = ARTIFACTS_ROOT / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.jsonl"

    try:
        from authinfra.generation.runner import run_generation
        written, err_count = run_generation(
            params["input_text"],
            prompt_id=params.get("prompt_id", "1"),
            model_name=params.get("model", "dry-run"),
            output_path=str(output_path),
            min_tokens=int(params.get("min_tokens", 300)),
            max_tokens=int(params.get("max_tokens", 1000)),
            overlap_tokens=int(params.get("overlap", 32)),
            dry_run=params.get("model", "dry-run").lower() == "dry-run",
            job_id=job_id,
        )
        update_job(
            job_id,
            status="completed",
            result={
                "output_path": str(output_path.resolve()),
                "lines_written": written,
                "error_count": err_count,
            },
        )
        logger.info(
            "job_completed",
            extra={"job_id": job_id, "type": "generation", "lines_written": written, "error_count": err_count},
        )
    except Exception as e:
        logger.exception("job_failed", extra={"job_id": job_id, "type": "generation"})
        update_job(job_id, status="failed", error=str(e))


def run_compile_job(job_id: str, params: dict) -> None:
    """Execute dataset compile; update job with result or error."""
    update_job(job_id, status="running")
    logger.info("job_start", extra={"job_id": job_id, "type": "compile", "stage": "running"})

    output_dir = ARTIFACTS_ROOT / "datasets" / params["name"]
    input_paths = params["input_paths"]

    try:
        from authinfra.datasets.compiler import compile_dataset
        manifest = compile_dataset(
            input_paths,
            output_dir,
            dataset_name=params.get("name", "dataset"),
            model_ids=params.get("model_ids") or None,
            prompt_ids=params.get("prompt_ids") or None,
            split_ratio=float(params.get("split_ratio", 0.9)),
            split_seed=int(params.get("split_seed", 0)),
        )
        update_job(
            job_id,
            status="completed",
            result={
                "manifest_path": str((output_dir / "manifest.json").resolve()),
                "train_count": manifest["train_count"],
                "valid_count": manifest["valid_count"],
                "filter_log": manifest.get("filter_log", []),
            },
        )
        logger.info(
            "job_completed",
            extra={
                "job_id": job_id,
                "type": "compile",
                "train_count": manifest["train_count"],
                "valid_count": manifest["valid_count"],
            },
        )
    except Exception as e:
        logger.exception("job_failed", extra={"job_id": job_id, "type": "compile"})
        update_job(job_id, status="failed", error=str(e))


def run_mass_convert_job(job_id: str, params: dict) -> None:
    """Run mass conversion: each input file -> run_generation, per-file JSONL + aggregated."""
    update_job(job_id, status="running")
    logger.info("job_start", extra={"job_id": job_id, "type": "mass_convert", "stage": "running"})

    input_paths = params.get("input_paths") or []
    output_dir = Path(params.get("output_path", str(ARTIFACTS_ROOT / "generation" / job_id)))
    prompt_id = params.get("prompt_id", "1")
    model = params.get("model", "dry-run")
    dry_run = str(model).lower() == "dry-run"
    min_tokens = int(params.get("min_tokens", 300))
    max_tokens = int(params.get("max_tokens", 1000))
    overlap = int(params.get("overlap", 32))
    concurrency = params.get("concurrency")

    try:
        from authinfra.generation.runner import run_generation

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregated_path = output_dir / "aggregated.jsonl"
        total_written = 0
        total_errors = 0
        files_processed = 0

        with open(aggregated_path, "w", encoding="utf-8") as agg_f:
            for fp in input_paths:
                p = Path(fp)
                if not p.is_file():
                    logger.warning("mass_convert_skip", extra={"path": fp, "reason": "not a file"})
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning("mass_convert_read_failed", extra={"path": fp, "error": str(e)})
                    total_errors += 1
                    continue
                out_name = p.name.replace(p.suffix, "") + ".jsonl"
                per_file_path = output_dir / out_name
                written, err_count = run_generation(
                    text,
                    prompt_id=prompt_id,
                    model_name=model,
                    output_path=str(per_file_path),
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap,
                    dry_run=dry_run,
                    job_id=job_id,
                    concurrency=concurrency,
                )
                total_written += written
                total_errors += err_count
                files_processed += 1
                with open(per_file_path, encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            agg_f.write(line)

        update_job(
            job_id,
            status="completed",
            result={
                "output_path": str(output_dir.resolve()),
                "aggregated": str(aggregated_path.resolve()),
                "files_processed": files_processed,
                "lines_written": total_written,
                "error_count": total_errors,
            },
        )
        logger.info(
            "job_completed",
            extra={"job_id": job_id, "type": "mass_convert", "files": files_processed, "lines": total_written, "errors": total_errors},
        )
    except Exception as e:
        logger.exception("job_failed", extra={"job_id": job_id, "type": "mass_convert"})
        update_job(job_id, status="failed", error=str(e))


def poll_interval() -> float:
    """Seconds between polls (env AUTHINFRA_WORKER_POLL_SECONDS)."""
    import os
    try:
        return float(os.environ.get("AUTHINFRA_WORKER_POLL_SECONDS", "2"))
    except ValueError:
        return 2.0


def main() -> int:
    logger.info("worker_started", extra={"artifacts_root": str(ARTIFACTS_ROOT)})
    interval = poll_interval()

    while True:
        try:
            for job_type in ("generation", "compile", "mass_convert"):
                pending = list_pending_jobs(job_type=job_type)
                for job in pending:
                    jid = job["job_id"]
                    params = job.get("params") or {}
                    if job_type == "generation":
                        run_generation_job(jid, params)
                    elif job_type == "compile":
                        run_compile_job(jid, params)
                    else:
                        run_mass_convert_job(jid, params)
        except Exception as e:
            logger.exception("worker_loop_error", extra={"error": str(e)})
        time.sleep(interval)


if __name__ == "__main__":
    main()
