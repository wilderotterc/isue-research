"""
Fetch the two missing 2024-25 handbook chapters that had different URL slugs.
Run this after scrape_handbook.py.
"""

import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def fetch_and_save(url, filepath):
    print(f"Fetching: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()
        text = clean_text(soup.get_text(separator="\n"))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved: {filepath} ({len(text):,} chars)")
    except Exception as e:
        print(f"  ERROR: {e}")

# avg_ch2 for 2024-25 has a different slug
fetch_and_save(
    "https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2024-2025/application-and-verification-guide/ch2-filling-out-fafsa",
    "corpus/federal_outdated/avg_ch2_202425.txt"
)

time.sleep(2)

# vol2_ch3 for 2024-25 — search for correct slug
fetch_and_save(
    "https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2024-2025/vol2/ch3-administrative-requirements",
    "corpus/federal_outdated/vol2_ch3_202425.txt"
)

print("\nDone.")