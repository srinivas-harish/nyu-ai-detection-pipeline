# generate_summaries.py
import pandas as pd
import time
import os,sys,re 
from .api_runner import run_api
api_key=os.getenv("GEMINI_API")
def clean_keywords(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    txt = raw.strip().strip('*"').strip()
    txt = re.sub(r"[*•-]+\s*", "", txt)
    txt = re.sub(r"\*\*", "", txt)
    txt = re.sub(r"[\r\n;|]+", ", ", txt)
    txt = re.sub(r",\s*,+", ", ", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = txt.strip(" ,")
    return txt

def generate_summaries(input_csv: str, output_csv: str,
                       model="gemini-2.5-flash",
                       api_key=api_key,
                       max_output=512):
    df = pd.read_csv(input_csv)
    summaries, keywords = [], []
    print(f"[generate_summaries] Loaded {len(df)} rows from {input_csv}")

    for i, row in df.iterrows():
        title = row.get("title", "")
        date = row.get("date", "")
        text = str(row.get("text", ""))[:2200]
        if not text.strip():
            summaries.append("")
            keywords.append("")
            continue

        prompt = f"""
        You are an expert research summarizer.
        Summarize this CRS report in 3–5 sentences.
        Then write 'Keywords:' followed by 5–8 comma-separated keywords.

        Title: {title}
        Date: {date}
        Report text:
        {text}
        """

        try:
            print(f"[{i+1}/{len(df)}] Summarizing: {title[:60]}...")
            response = run_api(model=model, prompt=prompt,
                               max_output=max_output, api_key=api_key)
            result = response.strip() if isinstance(response, str) else ""
            if "Keywords:" in result:
                parts = result.split("Keywords:")
                summaries.append(parts[0].strip())
                keywords.append(clean_keywords(parts[1]))
            else:
                summaries.append(result)
                keywords.append("")
            time.sleep(1.5)
        except Exception as e:
            print(f"[Error @ row {i}]: {e}")
            summaries.append("")
            keywords.append("")
            continue

    df["summary"] = summaries
    df["keywords"] = keywords
    df.to_csv(output_csv, index=False)
    print(f"[generate_summaries]  Saved summarized CSV to {output_csv}")
