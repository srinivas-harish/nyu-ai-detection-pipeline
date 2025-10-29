# nyu-ai-detection-pipeline

Pipeline and tools for domain-specific AI-text detection research, developed for  
**NYU’s AI in Education Vertically Integrated Projects (VIP) team**.

Structure:
- `src/` – core scripts (API runner, CRS scraper, cleaner, model code)
- `data/` – raw and processed datasets
- `notebooks/` – experiments and analysis

update:
The main.py script serves as the entry point for the entire CRS processing pipeline.
When you run it, it automatically triggers all stages in order:

Scraping – Uses crs_scraper.py to fetch CRS reports and store them as JSON files.

Filtering & Cleaning – Calls crs_filter.py to process the raw JSON files and create a structured CSV file.

Summarization – Invokes the summarization module to generate short summaries and keywords for each report using the API model.

All three steps run one after another — no manual intervention needed — and the final combined output is saved as
clean_data_with_summaries.csv.
You can start the full pipeline with a single command:
`python main.py`

the script automatically creates all required folders if they don’t already exist.

/jsons/ – stores the raw JSON files downloaded by crs_scraper.py.
Each file represents an individual CRS report (e.g., report_1.json, report_2.json).

/clean_data/ – created by crs_filter.py to store cleaned and structured data.
It contains files like clean_data.csv, which holds all processed report details in a tabular format.

clean_data_with_summaries.csv – added in the same folder after the summarization step.
This file includes extra columns for the automatically generated summaries and keywords.

.env:
insert gemini key after generating 