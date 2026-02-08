# AuthInfra scaffold

Minimal scaffold for AI text detection infrastructure. Empirical and reversible.

## What exists

- **Package `authinfra/`** with subpackages:
  - `datasets` — placeholder
  - `generation` — placeholder
  - `detectors` — **baseline detector** (Hello-SimpleAI/chatgpt-detector-roberta); download + inference wrapper, strict JSON output
  - `inference` — placeholder
  - `training` — stub only (no training logic)
  - `utils` — structured JSON logging and config loader (env + optional JSON config file)
- **CLI**: `python -m authinfra` prints available commands. Implemented: `config`, `version`, `detector-download`, `detector-infer`.
- **Config**: `load_config()` from env (`AUTHINFRA_*`) and optional file (`AUTHINFRA_CONFIG` or path argument).
- **Logging**: `configure_logging()`, `get_logger()`; JSON logs by default.
- **Layout**: `services/api` and `services/worker` placeholders; `docs/` and `artifacts/` for future use.

## Baseline detector

- **Model**: [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta).
- **Dataset (citation)**: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3).
- Wrapper: `authinfra.detectors.baseline` — `ensure_downloaded()`, `run_inference(text)` → JSON with `model`, `runtime_sec`, `probability` [0,1], `error`, `input_truncated`.
- Guardrails: empty input rejected; input length checked (truncation at 512 tokens); graceful failure if model not downloaded or load fails.

**What the baseline does NOT guarantee:** Not tuned for CRS or other domains; may be wrong; for comparison only; swappable later.

## What does NOT exist yet

- No training loops or training logic.
- No fine-tuning of the baseline.
- No other detector implementation or dataset loading in authinfra.
- No inference pipeline beyond the single-file baseline.

## Running the CLI

From repo root:

```bash
python -m authinfra
python -m authinfra version
python -m authinfra config
python -m authinfra detector-download
python -m authinfra detector-infer --input path/to/file.txt
```

Disable JSON logs: `AUTHINFRA_JSON_LOGS=false python -m authinfra version`
