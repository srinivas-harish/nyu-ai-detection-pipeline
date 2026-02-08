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
            for job_type in ("generation", "compile"):
                pending = list_pending_jobs(job_type=job_type)
                for job in pending:
                    jid = job["job_id"]
                    params = job.get("params") or {}
                    if job_type == "generation":
                        run_generation_job(jid, params)
                    else:
                        run_compile_job(jid, params)
        except Exception as e:
            logger.exception("worker_loop_error", extra={"error": str(e)})
        time.sleep(interval)


if __name__ == "__main__":
    main()
