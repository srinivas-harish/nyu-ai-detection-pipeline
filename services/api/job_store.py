"""
Filesystem job store for API and worker. One JSON file per job.
No refactor of existing logic; integration only.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root: parent of services/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_DIR = Path(os.environ.get("AUTHINFRA_JOBS_DIR", str(_REPO_ROOT / "artifacts" / "jobs")))
ARTIFACTS_ROOT = Path(os.environ.get("AUTHINFRA_ARTIFACTS", str(_REPO_ROOT / "artifacts")))


def _ensure_jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def create_job(job_type: str, params: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
    """Create a pending job; return job record with job_id. Optional job_id for mass_convert."""
    _ensure_jobs_dir()
    job_id = job_id or str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    job: dict[str, Any] = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "params": params,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    path = JOBS_DIR / f"{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    """Load job by id. Returns None if not found."""
    path = JOBS_DIR / f"{job_id}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_job(job_id: str, status: str | None = None, result: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any] | None:
    """Update job fields; pass only what changes. Returns updated job or None."""
    job = get_job(job_id)
    if job is None:
        return None
    if status is not None:
        job["status"] = status
    if result is not None:
        job["result"] = result
    if error is not None:
        job["error"] = error
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = JOBS_DIR / f"{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
    return job


def list_pending_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
    """List jobs with status pending. Optionally filter by type."""
    _ensure_jobs_dir()
    out: list[dict[str, Any]] = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            if j.get("status") != "pending":
                continue
            if job_type is not None and j.get("type") != job_type:
                continue
            out.append(j)
        except Exception:
            continue
    return out
