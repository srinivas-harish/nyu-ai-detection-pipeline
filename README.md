# nyu-ai-detection-pipeline

Domain-specific AI-text detection research pipeline for **NYU AI in Education (VIP)**.  
Scrape → filter → convert → (future: train). AuthInfra provides the detection scaffold: baseline model, generation, dataset compilation, and an operator UI—no training or fine-tuning in this repo.

---

## Repo layout

| Path | Role |
|------|------|
| `authinfra/` | Core library: datasets, generation, detectors, inference (CLI + Python). Training is a stub. |
| `apps/web/` | Next.js operator console: Generate, Datasets, Inference (dark UI). |
| `services/` | FastAPI API + filesystem worker for generation and compile jobs (experimental, local only). |
| `data_helpers/` | Scripts: CRS scraper, filter/cleaner, conversions, API runner. |
| `data/` | Raw and processed inputs. |
| `artifacts/` | Outputs: generation JSONL, compiled datasets, job state. |
| `docs/` | [AUTHINFRA.md](docs/AUTHINFRA.md) — what exists; [LIMITATIONS_AND_STABILITY.md](docs/LIMITATIONS_AND_STABILITY.md) — behavior and limits. |

**Stack (relevant):** Python (authinfra), Next.js 14, FastAPI, Hugging Face (baseline detector), Google Generative AI (Gemini adapter). Generation is dry-run by default; Gemini 3 Pro adapter reads keys from `api_keys.md` locally only (never logged or transmitted).

---

## Quick start: UI + inference

Get the UI up and run the baseline detector on pasted text. Inference is synchronous (no worker).

**Terminal 1 — API**

```bash
pip install -r requirements.txt
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — UI**

```bash
cd apps/web && npm install && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open **http://localhost:3000/inference**, paste text, click **Run baseline detector**. First run may be slow while the detector model downloads.

*UI only (no backend):* `cd apps/web && npm run dev` — you’ll get a prompt to set the API base URL.

---

## Baseline detector

Pretrained RoBERTa-based detector used as a **comparison baseline** only—not tuned for CRS or any domain; treat as a black box.

- **Model:** [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
- **Citation:** [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

**CLI**

```bash
pip install -r requirements.txt   # includes transformers, torch
python -m authinfra detector-download
python -m authinfra detector-infer --input path/to/file.txt
```

Output (stdout): JSON with `model`, `runtime_sec`, `probability` (0–1 or `null` on error), `error`, `input_truncated`.  
**Caveat:** `probability` can be `null` when `error` is set; 0.5 may indicate failed label mapping—do not use for accuracy claims.

---

## Generation pipeline

Chunk human text, apply a registered prompt and model adapter, write one JSONL line per chunk. Errors are per-line; dry-run is default (no API calls).

- **Prompts:** Registry IDs 1–10, version `v1` (`authinfra.generation.prompts`).
- **Chunking:** Min/max tokens, overlap; deterministic (tiktoken if available, else whitespace).
- **Adapters:** Dry-run (no API); **Gemini 3 Pro** (wired; reads `GEMINI_API_KEY` from `api_keys.md`); stubs for OpenAI, Anthropic. **API keys:** Only `api_keys.md` at repo root (or `AUTHINFRA_API_KEYS_PATH`) is read for Gemini; keys are never logged, printed, or transmitted.
- **Parallelism:** Bounded worker pool via `AUTHINFRA_GENERATION_CONCURRENCY` or `--concurrency N` (1–32). Output JSONL order is deterministic. Higher concurrency can hit rate limits; failures are captured per line.
- **Output fields:** `job_id`, `timestamp_utc`, `chunk_index`, `prompt_id`, `prompt_version`, `prompt_text`, `model_id`, `input_token_count`, `output_text`, `output_token_count`, `runtime_sec`, `error`, `dry_run`.

**CLI (single file or mass)**

```bash
# Dry-run (default)
python -m authinfra generate --input_file path/to/text.txt --output_path path/to/out.jsonl

# Gemini 3 Pro mass convert (folder → per-file JSONL + aggregated.jsonl)
python -m authinfra generate --input_dir path/to/folder --output_path artifacts/generation/mass_out \
  --model gemini --no-dry-run --prompt-id 1 --concurrency 4
```

- **`--input_file`** (or **`--input`**): single input file. **`--input_dir`**: directory; all `.txt`/`.md` under it are converted (recursive). Use one of the two, not both.
- **`--output_path`** (or **`--output`**): output file path, or directory for mass (writes `<name>.jsonl` per file and `aggregated.jsonl`).
- **`--concurrency`**: chunk-level parallelism (default from `AUTHINFRA_GENERATION_CONCURRENCY`, else 1). No guarantee of speed; rate limits and timeouts can occur.

---

## Dataset compiler

Turns generation JSONL into a **folder dataset**: `manifest.json`, `train.jsonl`, `valid.jsonl`. Deterministic split (seed); filter_log records every exclusion. No quality claims—plumbing only.

- **Record content:** Model output + metadata (same schema as generation JSONL). The human (input) chunk text is *not* stored.
- **Selection:** You specify `model_ids` and `prompt_ids`; everything else is excluded and logged.

**CLI**

```bash
python -m authinfra dataset-compile --name my_dataset --output-dir artifacts/datasets/my_dataset \
  --inputs "path/to/gen1.jsonl path/to/gen2.jsonl" --models dry-run --prompts 1,3

python -m authinfra dataset-summary --dataset-dir artifacts/datasets/my_dataset
```

Omit `--models` / `--prompts` to include all. Same sources + same options + same seed ⇒ same manifest and split.

---

## Web UI

Next.js app: **Generate** (paste text, start run, poll status), **Datasets** (list + manifest), **Inference** (paste text, run detector). Dark theme; no synthetic progress. Paste-only input—no file upload. Compile is not in the UI (use API or CLI).

**Run with backend:** See [Quick start](#quick-start-ui--inference) for API + UI. For generation and compile jobs you also need the worker (next section).

**Run UI only**

```bash
cd apps/web && npm install && npm run dev
```

Serves http://localhost:3000. Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` to hit the API.

---

## API and worker (experimental)

Local FastAPI service + one worker process. Job state and artifacts on disk; no external queue. **Experimental:** single worker, run from repo root, same env for API and worker.

**Full stack (API + worker + UI)**

```bash
# Terminal 1
pip install -r requirements.txt && uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
python -m services.worker

# Terminal 3
cd apps/web && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

- **Sync:** Inference runs in the API process; no worker.
- **Async:** Generation and dataset compile are enqueued; worker polls `artifacts/jobs/`, runs authinfra, updates job files; UI polls `GET /jobs/<id>`.

**Failure points:** Job dir must be writable (`artifacts/jobs` or `AUTHINFRA_JOBS_DIR`). If the worker isn’t running, generation/compile stay pending. Compile `input_paths` must exist (e.g. under `artifacts/generation/`). CORS is open for local use.

---

## Data helpers and keys

**API keys** (for scripts that call providers): put in `data_helpers/api_keys.txt`, e.g.  
`OPENAI_API_KEY=...`, `CLAUDE_API_KEY=...`, `GEMINI_API_KEY=...`, `DEEPSEEK_API_KEY=...`, `GROK_API_KEY=...`.  
AuthInfra generation does *not* read this file; only data_helpers scripts that need keys do.

**Example commands**

- **CRS scrape:**  
  `python data_helpers/crs_scraper.py --base https://www.everycrsreport.com/reports.csv --n 200 --out data/crs_jsons`
- **Filter to CSV (min words, row cap):**  
  `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --n 30`
- **Filter by token budget:**  
  `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --target_tokens 200000`
- **Sanity check:**  
  `python data_helpers/mass_conversion.py --input_csv_dir data/clean --sanity_only --min_words 3000`
- **Training-style samples (chunk + optional multi-model rewrite):**  
  `python data_helpers/make_training_examples.py --input_csv data/ai_input.csv --token_budget 150000 --min_chunk 300 --max_chunk 1000 --overlap 32 --models "grok,chatgpt,deepseek,gemini-2.5-flash-lite" --out data/samples/train_examples.jsonl --keys data_helpers/api_keys.txt`  
  Use `--dry_run` to scaffold without API calls.

Defaults: outputs often go to `data_helpers/jsons`, `./clean_data`, `./gen_out` unless you set `--out`. For exact tokenization, install `tiktoken`; otherwise scripts use whitespace.

---

## Out of scope

Training and fine-tuning are not implemented. No accuracy or production guarantees for the detector, generation, or compiled datasets. See [docs/LIMITATIONS_AND_STABILITY.md](docs/LIMITATIONS_AND_STABILITY.md) for a concise summary of behavior and limits.
