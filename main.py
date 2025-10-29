# main.py
import os
import time
import data_helpers.crs_scraper as crs_scraper
import data_helpers.crs_filter as crs_filter
import data_helpers.generate_summary as generate_summaries

# -------- CONFIGURATION --------
BASE_URL = "https://www.everycrsreport.com/reports.csv"
N_REPORTS = 20

JSON_DIR = "./jsons"
CLEAN_DIR = "./clean_data"
FINAL_OUT = os.path.join(CLEAN_DIR, "clean_data_with_summaries.csv")

# -------- STEP 1: SCRAPER --------
print("\n=== STEP 1: Fetching CRS Reports ===")
os.makedirs(JSON_DIR, exist_ok=True)
n_saved = crs_scraper.fetch_multiple(BASE_URL, N_REPORTS, JSON_DIR)
print(f"[main] Scraper completed. Saved {n_saved} reports to {JSON_DIR}")

# -------- STEP 2: FILTER --------
print("\n=== STEP 2: Cleaning and Filtering Data ===")
os.makedirs(CLEAN_DIR, exist_ok=True)
cleaned_csv_path = crs_filter.process_jsons(JSON_DIR, CLEAN_DIR, N_REPORTS)
print(f"[main]  Filter completed. Clean CSV ready at {cleaned_csv_path}")

# -------- STEP 3: SUMMARIZATION --------
print("\n=== STEP 3: Generating Summaries ===")
start = time.time()
generate_summaries.generate_summaries(  # call the function from your summarizer file
    input_csv=cleaned_csv_path,
    output_csv=FINAL_OUT
)
end = time.time()
print(f"[main]  Summaries generated successfully in {end - start:.2f}s.")
print(f"[main] Final output: {FINAL_OUT}")

print("\n Pipeline complete: Scraped → Cleaned → Summarized.")
