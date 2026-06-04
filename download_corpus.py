"""
Week 2 — Corpus Download Script (Updated)
Downloads all corpus documents and saves as clean UTF-8 text
in authority-labeled folders.
"""

import os
import re
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO

CORPUS = {
    "federal_primary":   "corpus/federal_primary",
    "federal_outdated":  "corpus/federal_outdated",
    "federal_consumer":  "corpus/federal_consumer",
    "professional":      "corpus/professional",
    "institutional":     "corpus/institutional",
}

for path in CORPUS.values():
    os.makedirs(path, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def save_text(folder, filename, text):
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path} ({len(text):,} chars)")

def fetch_html(url, folder, filename):
    print(f"Fetching HTML: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = clean_text(text)
        if len(text) < 500:
            print(f"  WARNING: Very short content ({len(text)} chars) — may need manual download")
        save_text(folder, filename, text)
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  --> Manual download needed: visit {url} and paste text into {folder}/{filename}")

def fetch_pdf(url, folder, filename):
    print(f"Fetching PDF: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        text = ""
        with pdfplumber.open(BytesIO(r.content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = clean_text(text)
        save_text(folder, filename, text)
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  --> Manual download needed: download PDF from {url} and place in {folder}/")

# ── Federal Consumer (studentaid.gov) ───────────────────────
# studentaid.gov blocks scrapers — these need manual copy-paste
# Instructions printed below
print("\n=== Federal Consumer (studentaid.gov) ===")
print("  studentaid.gov blocks automated requests.")
print("  Please manually visit each URL, copy all page text,")
print("  and save as a .txt file in corpus/federal_consumer/")
manual_pages = [
    ("https://studentaid.gov/understand-aid/types/grants/pell",                      "pell_grant.txt"),
    ("https://studentaid.gov/understand-aid/types/work-study",                       "work_study.txt"),
    ("https://studentaid.gov/understand-aid/types/loans/subsidized-unsubsidized",    "subsidized_loans.txt"),
    ("https://studentaid.gov/complete-aid-process/how-calculated",                   "how_aid_calculated.txt"),
    ("https://studentaid.gov/apply-for-aid/fafsa/fafsa-deadlines",                   "fafsa_deadlines.txt"),
    ("https://studentaid.gov/understand-aid/eligibility/staying-eligible",            "sap_consumer.txt"),
]
for url, fname in manual_pages:
    path = os.path.join(CORPUS["federal_consumer"], fname)
    if os.path.exists(path):
        print(f"  Already exists: {path}")
    else:
        print(f"  NEEDED: {url}  -->  {path}")

# ── Professional (NASFAA) ────────────────────────────────────
print("\n=== Professional (NASFAA) ===")
fetch_html(
    "https://www.nasfaa.org/about_financial_aid",
    CORPUS["professional"],
    "nasfaa_about_financial_aid.txt"
)
fetch_pdf(
    "https://www.nasfaa.org/uploads/documents/2025_National_Profile.pdf",
    CORPUS["professional"],
    "nasfaa_2025_national_profile.pdf.txt"
)

# ── Institutional ────────────────────────────────────────────
print("\n=== Institutional ===")
fetch_html(
    "https://www.washington.edu/financialaid/receiving-aid/satisfactory-academic-progress/",
    CORPUS["institutional"],
    "uw_sap.txt"
)
fetch_html(
    "https://www.pace.edu/financial-aid/policies-and-procedures/satisfactory-academic-progress-policy/undergraduate-students",
    CORPUS["institutional"],
    "pace_sap.txt"
)
# U-M blocks scrapers — manual needed
print("Fetching HTML: https://finaid.umich.edu/managing-your-aid/satisfactory-academic-progress")
umich_path = os.path.join(CORPUS["institutional"], "umich_sap.txt")
if os.path.exists(umich_path):
    print(f"  Already exists: {umich_path}")
else:
    print("  ERROR: 403 Forbidden — manual download needed")
    print("  --> Visit https://finaid.umich.edu/managing-your-aid/satisfactory-academic-progress")
    print(f"  --> Copy all text and save to {umich_path}")

fetch_html(
    "https://cssprofile.collegeboard.org/",
    CORPUS["institutional"],
    "collegeboard_css.txt"
)

# ── FSA Handbook — PDF download instructions ─────────────────
print("\n=== FSA Handbook (Manual PDF Downloads Required) ===")
print("""
The FSA Handbook volumes are behind a JavaScript-rendered page and cannot
be scraped automatically. Please download the key volumes manually:

2025-26 Handbook --> save PDFs to: corpus/federal_primary/
  Volume 1 (Student Eligibility):
    https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2025-2026/vol1
  Volume 3 (Calculating Award Amounts):
    https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2025-2026/vol3
  Application & Verification Guide (AVG):
    https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2025-2026/application-and-verification-guide

2024-25 Handbook --> save PDFs to: corpus/federal_outdated/
  Same volumes, prior year:
    https://fsapartners.ed.gov/knowledge-center/fsa-handbook/2024-2025

After downloading, run: python process_pdfs.py
to extract text from the PDFs automatically.
""")

print("\n=== Done ===")
print("Check corpus/ folders and complete any manual downloads listed above.")