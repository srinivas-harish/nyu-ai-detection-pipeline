# AuthInfra scaffold

Minimal scaffold for AI text detection infrastructure. Empirical and reversible.

## What exists

- **Package `authinfra/`** with subpackages:
  - `datasets` — compiler: generation JSONL(s) → folder with manifest.json, train.jsonl, valid.jsonl; explicit model/prompt selection; filter_log; reproducible split
  - `generation` — prompt registry (IDs, versioning), chunking (min/max tokens, overlap), provider-agnostic adapters (dry-run + stubs), job runner writing JSONL with errors per line
  - `detectors` — **baseline detector** (Hello-SimpleAI/chatgpt-detector-roberta); download + inference wrapper, strict JSON output
  - `inference` — placeholder
  - `training` — stub only (no training logic)
  - `utils` — structured JSON logging and config loader (env + optional JSON config file)
- **CLI**: `python -m authinfra` prints available commands. Implemented: `config`, `version`, `detector-download`, `detector-infer`, `generate`, `dataset-compile`, `dataset-summary`.
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
- No real API wiring for generation adapters (OpenAI, Anthropic, Gemini are stubs).

## Generation pipeline

- Prompt registry: `get_prompt(id)`, `list_prompt_ids()`, version `v1`.
- Chunking: `chunk_text(text, min_tokens, max_tokens, overlap_tokens)` — deterministic.
- Adapters: `BaseAdapter`, `DryRunAdapter` (no API), stubs for OpenAI, Anthropic, Gemini. `get_adapter(name)`.
- Runner: `run_generation(input_text, prompt_id, model_name, output_path, dry_run=True, ...)` → writes JSONL; returns (lines_written, error_count).
- JSONL schema: each line has `job_id`, `timestamp_utc`, `chunk_index`, `prompt_id`, `prompt_version`, `prompt_text`, `model_id`, `input_token_count`, `output_text`, `output_token_count`, `runtime_sec`, `error`, `dry_run`. Errors are explicit per line.

## Dataset compiler

- Consumes one or more generation JSONL files; selects by `model_ids` and `prompt_ids` (explicit lists).
- Excludes lines with `error` set or missing `output_text`; all exclusions recorded in `filter_log` (reason + count).
- Output folder: `manifest.json` (schema_version v1, source_paths, model_ids, prompt_ids, filter_log, train_count, valid_count, split_ratio, split_seed), `train.jsonl`, `valid.jsonl`.
- Split is deterministic (split_seed). No examples silently dropped. Raw generation schema preserved in train/valid lines.
- `compile_dataset(...)`, `load_manifest(...)`, `dataset_summary_counts(...)`.

## Running the CLI

From repo root:

```bash
python -m authinfra
python -m authinfra version
python -m authinfra config
python -m authinfra detector-download
python -m authinfra detector-infer --input path/to/file.txt
python -m authinfra generate --input path/to/text.txt --output path/to/out.jsonl
python -m authinfra dataset-compile --name my_dataset --output-dir artifacts/datasets/my_dataset --inputs "gen1.jsonl gen2.jsonl" --models dry-run --prompts 1,3
python -m authinfra dataset-summary --dataset-dir artifacts/datasets/my_dataset
```

Disable JSON logs: `AUTHINFRA_JSON_LOGS=false python -m authinfra version`
