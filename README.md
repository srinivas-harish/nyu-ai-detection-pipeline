# nyu-ai-detection-pipeline

Pipeline and tools for domain-specific AI-text detection research, developed for  
**NYU’s AI in Education Vertically Integrated Projects (VIP) team)**.

Structure:
- `data_helpers/` – helper scripts (API runner, CRS scraper, filter/cleaner, conversions)
- `data/` – raw and processed datasets
- `notebooks/` – experiments and analysis

Pipeline: scrape → filter → convert (loop + overlap) → train


**Quick Setup**

- API keys file: put keys in `data_helpers/api_keys.txt` :
  - `GEMINI_API_KEY=...`
  - `DEEPSEEK_API_KEY=...`
  - `OPENAI_API_KEY=...`
  - `CLAUDE_API_KEY=...`
  - `GROK_API_KEY=...`


**Example Commands**

- CRS scraping (raw JSON) – `data_helpers/crs_scraper.py`
  - Grab 200 reports from EveryCRSReport CSV into a folder:
    - `python data_helpers/crs_scraper.py --base https://www.everycrsreport.com/reports.csv --n 200 --out data/crs_jsons`
 

- Clean + filter to CSV – `data_helpers/crs_filter.py`
  - Keep CRS files with minimum 3,000 words; stop after 30 kept rows:
    - `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --n 30`
  - Or target a token budget (approx. words) instead of a row count:
    - `python data_helpers/crs_filter.py --json_dir data/crs_jsons --out data/clean --min_words 3000 --target_tokens 200000`
  - Writes chunked CSVs like `clean_0.csv`, `clean_1.csv` under `--out`.
 
- Mass conversion + sanity checks – `data_helpers/mass_conversion.py`
  - Quick sanity check of your cleaned CSV folder:
    - `python data_helpers/mass_conversion.py --input_csv_dir data/clean --sanity_only --min_words 3000`
  - Note: large “source → single large AI” conversions are deprecated in this pipeline. Convert of sources to monolithic AI equivalents causes hallucination issues. And, this isn't the way most people use LLMs anyway.

- Training examples JSONL – `data_helpers/make_training_examples.py`
  - Take a large AI text and turn it into many AI samples by chunking (and optionally rewriting via multiple models).
  - If your AI text lives in a `.txt`, wrap it into a minimal CSV first (headers: `id,title,date,text`):
    - ``printf 'id,title,date,text\nA1,AI-Doc,2025-01-01,"%s"\n' "$(cat path/to/ai_text.txt | tr '\r' '\n')" > data/ai_input.csv``
  - Produce many samples from that AI text by chunking and calling several models:
    - `python data_helpers/make_training_examples.py --input_csv data/ai_input.csv --token_budget 150000 --min_chunk 300 --max_chunk 1000 --overlap 32 --models "grok,chatgpt,deepseek,gemini-2.5-flash-lite" --temperature 0.4 --top_p 0.9 --out data/samples/train_examples.jsonl --keys data_helpers/api_keys.txt`
  - Only chunk (no API calls yet), useful to scaffold samples quickly:
    - `python data_helpers/make_training_examples.py --input_csv data/ai_input.csv --token_budget 80000 --min_chunk 300 --max_chunk 1000 --overlap 32 --models "grok" --dry_run --out data/samples/train_examples.jsonl`
    - Note: `--dry_run` uses placeholders for outputs; rerun without `--dry_run` to populate actual model rewrites.


**Notes**
- Defaults: many scripts create sensible default folders (e.g., JSONs under `data_helpers/jsons`, cleaned CSVs under `./clean_data`, generated outputs under `./gen_out`). Set `--out` to keep things tidy in `data/`.
- Word vs token counts: where exact tokenization is needed, install `tiktoken`; otherwise scripts fall back to whitespace word counts.
- Please help optimize parallelizing API calls. 
- Sampling design for training has to be improved. However, we'll do this after we have the first training run. 
