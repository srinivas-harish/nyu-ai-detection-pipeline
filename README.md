# nyu-ai-detection-pipeline

Research pipeline for domain-specific AI-text detection (NYU AI in Education / VIP). We scrape, filter, convert; training is planned later. This repo has the baseline detector, a generation pipeline, dataset compilation, and a small operator UI. No training or fine-tuning lives here yet.

**Run the UI:**

```bash
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000. To use the detector or run generation jobs you need the API as well — see [Quick start: UI + inference](#quick-start-ui--inference).

---

## Repo layout

| Path | What it is |
|------|-------------|
| `authinfra/` | Core lib: datasets, generation, detectors, inference (CLI + Python). Training is stubbed. |
| `apps/web/` | Next.js UI: Generate, Datasets, Inference (dark theme). |
| `services/` | FastAPI API + one worker for generation/compile. Local only. |
| `data_helpers/` | Scripts: CRS scraper, filter, conversions, API runner. |
| `data/` | Raw and processed input. |
| `artifacts/` | Generation JSONL, compiled datasets, job state. |
| `docs/` | Extra docs (if present). |

Stack: Python (authinfra), Next.js 14, FastAPI, Hugging Face (detector), Google Generative AI (Gemini). Generation is dry-run by default. Gemini reads `GEMINI_API_KEY` from `api_keys.md` at repo root (or `AUTHINFRA_API_KEYS_PATH`); we don’t log or send keys. For real Gemini you need `pip install -r requirements.txt` (includes `google-generativeai`).

---

## Quick start: UI + inference

Two terminals.

**Terminal 1 — API**

```bash
pip install -r requirements.txt
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — UI**

```bash
cd apps/web && npm install && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Go to http://localhost:3000/inference, paste text, hit “Run baseline detector”. First run can be slow while the model downloads.

UI without backend: `cd apps/web && npm run dev` — it’ll ask for the API base URL.

---

## Baseline detector

RoBERTa-based detector from Hugging Face. We use it as a comparison baseline only; it’s not tuned for CRS or this domain.

- Model: [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
- Citation: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

**CLI**

```bash
pip install -r requirements.txt   # transformers, torch
python -m authinfra detector-download
python -m authinfra detector-infer --input path/to/file.txt
```

Output is JSON: `model`, `runtime_sec`, `probability` (0–1 or `null` on error), `error`, `input_truncated`. If `error` is set, `probability` can be `null`; don’t treat 0.5 as a reliable cutoff.

---

## Generation pipeline

Chunks text, runs a prompt + model adapter, writes one JSONL line per chunk. Default is dry-run (no API). Errors go on each line.

- Prompts: registry IDs 1–10, version `v1` in `authinfra.generation.prompts`.
- Chunking: min/max tokens, overlap; deterministic (tiktoken if present, else whitespace).
- Adapters: dry-run; Gemini 3 Pro (reads `api_keys.md`); stubs for OpenAI/Anthropic. Keys only from `api_keys.md` / `AUTHINFRA_API_KEYS_PATH`, never logged.
- Parallelism: `AUTHINFRA_GENERATION_CONCURRENCY` or `--concurrency N` (1–32). Order of lines is deterministic. Higher concurrency can hit rate limits.
- JSONL fields: `job_id`, `timestamp_utc`, `chunk_index`, `prompt_id`, `prompt_version`, `prompt_text`, `model_id`, `input_token_count`, `output_text`, `output_token_count`, `runtime_sec`, `error`, `dry_run`.

**CLI**

```bash
# Dry-run (default)
python -m authinfra generate --input_file path/to/text.txt --output_path path/to/out.jsonl

# Gemini mass convert (folder → per-file JSONL + aggregated.jsonl)
python -m authinfra generate --input_dir path/to/folder --output_path artifacts/generation/mass_out \
  --model gemini --no-dry-run --prompt-id 1 --concurrency 4
```

Use either `--input_file` (or `--input`) or `--input_dir`, not both. `--output_path`: file for single input, or directory for mass (writes one `.jsonl` per file plus `aggregated.jsonl`). `--concurrency` defaults from env or 1; rate limits and timeouts can still happen.

---

## Dataset compiler

Turns generation JSONL into a dataset folder: `manifest.json`, `train.jsonl`, `valid.jsonl`. Split is deterministic (seed); filter_log lists exclusions. Records are model output + metadata (same schema as generation); we don’t store the original human chunk text. You choose `model_ids` and `prompt_ids`; the rest are dropped and logged.

**CLI**

```bash
python -m authinfra dataset-compile --name my_dataset --output-dir artifacts/datasets/my_dataset \
  --inputs "path/to/gen1.jsonl path/to/gen2.jsonl" --models dry-run --prompts 1,3

python -m authinfra dataset-summary --dataset-dir artifacts/datasets/my_dataset
```

Omit `--models` / `--prompts` to include everything. Same inputs + options + seed → same manifest and split.

---

## Web UI

Next.js: **Generate** (paste, run, poll), **Datasets** (list + manifest), **Inference** (paste, run detector). Dark theme, paste-only (no uploads). Compile isn’t in the UI; use API or CLI.

With backend: [Quick start](#quick-start-ui--inference). For generation and compile you also need the worker (next section).

UI only:

```bash
cd apps/web && npm install && npm run dev
```

Serves http://localhost:3000. Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` to talk to the API.

---

## API and worker

FastAPI + one worker. Jobs and artifacts live on disk; no external queue. Single worker, run from repo root, same env for API and worker.

**Full stack**

```bash
# Terminal 1
pip install -r requirements.txt && uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
python -m services.worker

# Terminal 3
cd apps/web && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Inference runs in the API. Generation and compile go to the worker (polls `artifacts/jobs/`, runs authinfra, updates job files); UI polls `GET /jobs/<id>`. Job dir must be writable (`artifacts/jobs` or `AUTHINFRA_JOBS_DIR`). If the worker isn’t running, those jobs stay pending. Compile `input_paths` must exist. CORS is open for local use.

---

## Data helpers and keys

For scripts that call external APIs: put keys in `data_helpers/api_keys.txt` (e.g. `OPENAI_API_KEY=...`, `CLAUDE_API_KEY=...`, `GEMINI_API_KEY=...`, `DEEPSEEK_API_KEY=...`, `GROK_API_KEY=...`). AuthInfra generation does *not* use this file; only data_helpers scripts do.

**Examples**

- CRS scrape:  
  `python data_helpers/crs_scraper.py --base https://www.everycrsreport.com/reports.csv --n 200 --out data/crs_jsons`
- Filter to CSV:  
  `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --n 30`
- Filter by token budget:  
  `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --target_tokens 200000`
- Sanity check:  
  `python data_helpers/mass_conversion.py --input_csv_dir data/clean --sanity_only --min_words 3000`
- Training-style samples:  
  `python data_helpers/make_training_examples.py --input_csv data/ai_input.csv --token_budget 150000 --min_chunk 300 --max_chunk 1000 --overlap 32 --models "grok,chatgpt,deepseek,gemini-2.5-flash-lite" --out data/samples/train_examples.jsonl --keys data_helpers/api_keys.txt`  
  Use `--dry_run` to skip API calls.

Outputs often go to `data_helpers/jsons`, `./clean_data`, `./gen_out` unless you pass `--out`. Install `tiktoken` for exact tokenization; otherwise scripts use whitespace.

---

## Out of scope

No training or fine-tuning in this repo. No accuracy or production guarantees for the detector, generation, or datasets.
