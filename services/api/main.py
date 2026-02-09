"""
Minimal API service: generation (async via worker), dataset compile (async), inference (sync).
Fail loudly; structured logs per request. No refactor of authinfra.
"""

import logging
import sys
from pathlib import Path

# Ensure repo root on path for authinfra
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.api.job_store import (
    ARTIFACTS_ROOT,
    JOBS_DIR,
    create_job,
    get_job,
    list_pending_jobs,
    update_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("api")

app = FastAPI(title="AuthInfra API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response models ---


class GenerateRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Raw input text to chunk and generate from")
    prompt_id: str = Field("1", description="Registry prompt ID 1-10")
    model: str = Field("dry-run", description="Model adapter: dry-run, openai, anthropic, gemini")
    prompt_version: str | None = Field(None, description="Optional; registry uses v1. Custom prompt not applied by backend yet.")


class GenerateStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class CompileRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Dataset name (folder under artifacts/datasets)")
    input_paths: list[str] = Field(..., min_length=1, description="Paths to generation JSONL files")
    model_ids: list[str] | None = Field(None, description="Filter by model_ids; None = all")
    prompt_ids: list[str] | None = Field(None, description="Filter by prompt_ids; None = all")
    split_ratio: float = Field(0.9, ge=0.0, le=1.0)
    split_seed: int = Field(0)


class CompileStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class InferenceRequest(BaseModel):
    text: str = Field(..., description="Text to run baseline detector on")


# --- Endpoints ---


@app.post("/generate")
def start_generation(req: GenerateRequest):
    """Enqueue a generation job. Returns job_id; poll GET /jobs/{job_id} for status."""
    try:
        job = create_job(
            "generation",
            {
                "input_text": req.input_text,
                "prompt_id": req.prompt_id,
                "model": req.model,
                "prompt_version": req.prompt_version or "v1",
            },
        )
        logger.info("job_created", extra={"job_id": job["job_id"], "type": "generation"})
        return {"job_id": job["job_id"], "status": "pending"}
    except Exception as e:
        logger.exception("job_create_failed", extra={"type": "generation"})
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate/mass")
async def start_mass_convert(
    output_path: str = Form(..., description="Output directory path"),
    prompt_id: str = Form("1"),
    model: str = Form("dry-run"),
    files: list[UploadFile] = File(..., description="One or more .txt/.md files"),
):
    """Upload files and enqueue mass conversion. Returns job_id; poll GET /jobs/{job_id}."""
    if not files:
        raise HTTPException(status_code=400, detail="at least one file required")
    job_id = str(uuid.uuid4())[:12]
    mass_dir = ARTIFACTS_ROOT / "mass_input" / job_id
    mass_dir.mkdir(parents=True, exist_ok=True)
    input_paths: list[str] = []
    try:
        for f in files:
            if not f.filename or not (f.filename.endswith(".txt") or f.filename.endswith(".md")):
                continue
            safe_name = f.filename.replace("..", "_").replace("/", "_")
            out_path = mass_dir / safe_name
            content = await f.read()
            out_path.write_bytes(content)
            input_paths.append(str(out_path.resolve()))
        if not input_paths:
            raise HTTPException(status_code=400, detail="no .txt or .md files in upload")
        job = create_job(
            "mass_convert",
            {
                "input_paths": input_paths,
                "output_path": output_path,
                "prompt_id": prompt_id,
                "model": model,
                "min_tokens": 300,
                "max_tokens": 1000,
                "overlap": 32,
            },
            job_id=job_id,
        )
        logger.info("job_created", extra={"job_id": job["job_id"], "type": "mass_convert", "files": len(input_paths)})
        return {"job_id": job["job_id"], "status": "pending"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("job_create_failed", extra={"type": "mass_convert"})
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Return job status (generation or compile)."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job["job_id"],
        "type": job["type"],
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.post("/datasets/compile")
def start_compile(req: CompileRequest):
    """Enqueue a dataset compile job. Returns job_id; poll GET /jobs/{job_id} for status."""
    try:
        job = create_job(
            "compile",
            {
                "name": req.name,
                "input_paths": req.input_paths,
                "model_ids": req.model_ids,
                "prompt_ids": req.prompt_ids,
                "split_ratio": req.split_ratio,
                "split_seed": req.split_seed,
            },
        )
        logger.info("job_created", extra={"job_id": job["job_id"], "type": "compile"})
        return {"job_id": job["job_id"], "status": "pending"}
    except Exception as e:
        logger.exception("job_create_failed", extra={"type": "compile"})
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/datasets")
def list_datasets():
    """List compiled dataset names (subdirs of artifacts/datasets that contain manifest.json)."""
    datasets_dir = ARTIFACTS_ROOT / "datasets"
    if not datasets_dir.is_dir():
        return {"datasets": []}
    names: list[str] = []
    for d in datasets_dir.iterdir():
        if d.is_dir() and (d / "manifest.json").is_file():
            names.append(d.name)
    return {"datasets": sorted(names)}


@app.get("/datasets/{name}/manifest")
def get_dataset_manifest(name: str):
    """Return manifest.json for a compiled dataset."""
    import json
    manifest_path = ARTIFACTS_ROOT / "datasets" / name / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="dataset or manifest not found")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("manifest_read_failed", extra={"dataset": name})
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/inference")
def run_inference(req: InferenceRequest):
    """Run baseline detector on text. Synchronous; returns detector JSON."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        from authinfra.detectors.baseline import run_inference as run_detector
        result = run_detector(req.text.strip())
        logger.info("inference_done", extra={"error": result.get("error"), "probability": result.get("probability")})
        return result
    except Exception as e:
        logger.exception("inference_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health():
    """Liveness; fails if job dir not writable."""
    from services.api.job_store import _ensure_jobs_dir
    try:
        _ensure_jobs_dir()
        return {"status": "ok", "jobs_dir": str(JOBS_DIR)}
    except Exception as e:
        logger.error("health_check_failed", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
