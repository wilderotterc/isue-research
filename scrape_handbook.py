"""
Week 2 — FSA Handbook Full Scraper
Fetches all chapters from both 2025-26 and 2024-25 FSA Handbooks
and saves as clean UTF-8 text files.
"""

import os
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

def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return clean_text(text)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def save(folder, filename, text):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    Saved: {filename} ({len(text):,} chars)")

# ── Chapter URLs ─────────────────────────────────────────────

# Key chapters relevant to the research study
# Focus on AVG, Vol 1, Vol 3, Vol 6, Vol 7 — most relevant to student questions
# Other volumes included for completeness

BASE_2526 = "https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2025-2026"
BASE_2425 = "https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2024-2025"

CHAPTERS = [
    # Application and Verification Guide
    ("application-and-verification-guide",                                        "avg_intro"),
    ("application-and-verification-guide/ch1-application-process-fafsa-isir",    "avg_ch1"),
    ("application-and-verification-guide/ch2-filling-out-fafsa-form",            "avg_ch2"),
    ("application-and-verification-guide/ch3-student-aid-index-sai-and-pell-grant-eligibility", "avg_ch3"),
    ("application-and-verification-guide/ch4-verification-updates-and-corrections", "avg_ch4"),
    ("application-and-verification-guide/ch5-special-cases",                     "avg_ch5"),

    # Volume 1 — Student Eligibility (SAP, enrollment status)
    ("vol1",                                                                      "vol1_intro"),
    ("vol1/ch1-school-determined-requirements",                                   "vol1_ch1"),
    ("vol1/ch2-us-citizenship-eligible-noncitizens",                             "vol1_ch2"),
    ("vol1/ch3-nslds-financial-aid-history",                                      "vol1_ch3"),
    ("vol1/ch4-social-security-number",                                           "vol1_ch4"),

    # Volume 2 — School Eligibility
    ("vol2",                                                                      "vol2_intro"),
    ("vol2/ch1-institutional-eligibility",                                        "vol2_ch1"),
    ("vol2/ch3-title-iv-administrative-and-related-requirements",                 "vol2_ch3"),

    # Volume 3 — COA and Packaging (key for M3 questions)
    ("vol3",                                                                      "vol3_intro"),
    ("vol3/ch1-academic-years-academic-calendars-payment-periods-and-disbursements", "vol3_ch1"),
    ("vol3/ch2-cost-attendance-budget",                                           "vol3_ch2"),
    ("vol3/ch3-packaging-aid",                                                    "vol3_ch3"),

    # Volume 5 — Withdrawals (dropping below half time)
    ("vol5",                                                                      "vol5_intro"),
    ("vol5/ch1-general-requirements-withdrawals-and-return-title-iv-funds",       "vol5_ch1"),

    # Volume 6 — Campus-Based Programs (Work-Study)
    ("vol6",                                                                      "vol6_intro"),
    ("vol6/ch1-campus-based-programs-common-elements",                            "vol6_ch1"),
    ("vol6/ch2-federal-work-study-program",                                       "vol6_ch2"),

    # Volume 7 — Pell Grant Program
    ("vol7",                                                                      "vol7_intro"),
    ("vol7/ch1-student-eligibility-pell-grants",                                  "vol7_ch1"),
    ("vol7/ch2-calculating-pell-grants",                                          "vol7_ch2"),
    ("vol7/ch3-pell-grant-enrollment-intensity-and-cost-attendance",              "vol7_ch3"),
    ("vol7/ch8-pell-grant-lifetime-eligibility-used-leu",                         "vol7_ch8"),
]

HANDBOOK_VERSIONS = [
    (BASE_2526, "corpus/federal_primary",  "2025-26"),
    (BASE_2425, "corpus/federal_outdated", "2024-25"),
]

for base_url, folder, year in HANDBOOK_VERSIONS:
    print(f"\n{'='*60}")
    print(f"Fetching FSA Handbook {year}")
    print(f"{'='*60}")

    # Fetch intro/index page first
    print(f"\n  Fetching index page...")
    text = fetch_page(base_url)
    if text:
        save(folder, f"fsa_handbook_{year.replace('-','')}_index.txt", text)
    time.sleep(1)

    # Fetch each chapter
    for chapter_path, chapter_name in CHAPTERS:
        url = f"{base_url}/{chapter_path}"
        print(f"\n  Fetching {chapter_name}...")
        text = fetch_page(url)
        if text:
            filename = f"{chapter_name}_{year.replace('-','')}.txt"
            save(folder, filename, text)
        else:
            print(f"    FAILED — may need manual download: {url}")
        time.sleep(1.5)  # Be polite to the server

print("\n\nDone! Check corpus/federal_primary/ and corpus/federal_outdated/ for all files.")
print("Run python build_vectorstore.py next to chunk and embed everything.")