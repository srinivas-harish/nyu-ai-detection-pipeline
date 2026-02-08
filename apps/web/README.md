# AuthInfra Web UI

Operator console for generation, datasets, and inference. Dark-mode default; no synthetic metrics.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. Backend is not wired; pages show stub state and TODOs where APIs are missing.

## What works

- **Generate**: Prompt ID/version and model selection; custom prompt text. Start run shows stub status (no backend).
- **Datasets**: Placeholder for dataset list and manifest view; list is empty until backend serves compiled datasets.
- **Inference**: Text area and run button; result shows probability, runtime, error. Currently returns stub error (no backend).

## What does not work yet

- No API server. Set `NEXT_PUBLIC_API_BASE` and implement `/api/generate`, `/api/datasets`, `/api/inference` (or equivalent) to wire to authinfra.
- No file upload for generation inputs; no job status polling; no sentence-level inference output.
