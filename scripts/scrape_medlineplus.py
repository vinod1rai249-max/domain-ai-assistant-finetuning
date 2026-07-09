"""
Step 2: Non-instruction data collection (diabetes domain).

Pulls patient-education prose from MedlinePlus diabetes pages, cleans HTML,
splits into paragraphs, and writes data/non_instruction_data.txt.

Run this in Colab (needs internet access). Install deps first:
    !pip install requests beautifulsoup4 lxml

MedlinePlus content is U.S. government public health information (NIH/NLM),
free to reuse for educational purposes. Still cite the source in your README.
"""

import re
import time
import requests
from bs4 import BeautifulSoup

# Curated MedlinePlus pages covering diabetes management, symptoms,
# complications, diet, and emergencies (hypo/hyperglycemia) -- this spread
# is what gives you both routine and "trap question" material later.
MEDLINEPLUS_URLS = [
    "https://medlineplus.gov/diabetes.html",
    "https://medlineplus.gov/diabetestype1.html",
    "https://medlineplus.gov/diabetestype2.html",
    "https://medlineplus.gov/diabeticdiet.html",
    "https://medlineplus.gov/diabetesmedicines.html",
    "https://medlineplus.gov/diabetesinchildrenandteens.html",
    "https://medlineplus.gov/diabetesandpregnancy.html",
    "https://medlineplus.gov/diabetescomplications.html",
    "https://medlineplus.gov/hypoglycemia.html",
    "https://medlineplus.gov/ency/article/000305.htm",   # diabetic ketoacidosis
    "https://medlineplus.gov/ency/article/000320.htm",   # hyperosmolar hyperglycemic state
    "https://medlineplus.gov/diabeticeyeproblems.html",
    "https://medlineplus.gov/diabeticfootproblems.html",
    "https://medlineplus.gov/diabetickidneydisease.html",
    "https://medlineplus.gov/diabetesinsulin.html",
]

MIN_PARAGRAPH_WORDS = 25   # filter out nav fragments / captions
MIN_PARAGRAPH_CHARS = 150


def fetch(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (educational dataset collection)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text


def extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    # MedlinePlus topic pages keep main content inside <div id="topic-summary">
    # or <div class="section-body">; ency articles use <div id="ency_summary">
    # and <div class="section"> blocks. Try the common containers, fall back
    # to all <p> tags if none match.
    containers = soup.select(
        "#topic-summary, .section-body, #ency_summary, .section, main"
    )
    if not containers:
        containers = [soup]

    paragraphs = []
    seen = set()
    for container in containers:
        for p in container.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if (
                len(text) >= MIN_PARAGRAPH_CHARS
                and len(text.split()) >= MIN_PARAGRAPH_WORDS
                and text not in seen
            ):
                seen.add(text)
                paragraphs.append(text)
    return paragraphs


def main():
    all_paragraphs = []
    for url in MEDLINEPLUS_URLS:
        try:
            html = fetch(url)
            paras = extract_paragraphs(html)
            print(f"{url} -> {len(paras)} paragraphs")
            all_paragraphs.extend(paras)
        except Exception as e:
            print(f"FAILED {url}: {e}")
        time.sleep(1)  # be polite to the server

    # Dedup across pages (some content repeats between topic/ency pages)
    deduped = list(dict.fromkeys(all_paragraphs))
    print(f"\nTotal unique paragraphs: {len(deduped)} (need >= 50)")

    if len(deduped) < 50:
        print("WARNING: under the 50-paragraph minimum. Add more URLs or "
              "lower MIN_PARAGRAPH_WORDS, then re-run.")

    out_path = "data/non_instruction_data.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for para in deduped:
            f.write(para + "\n\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
