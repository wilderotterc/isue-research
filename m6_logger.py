"""
M6 — Logger
Appends one row to raw_log.csv per run with 15 fields.
"""

import csv
import os
from datetime import datetime, timezone

LOG_FILE = "raw_log.csv"

FIELDS = [
    "run_id",
    "question_id",
    "category",
    "condition",
    "query",
    "retrieved_chunk_ids",
    "retrieved_authority_tags",
    "retrieved_award_years",
    "retrieved_documents",
    "answer",
    "faithfulness",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model_name",
    "timestamp",
]

def init_log():
    """Create raw_log.csv with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
        print(f"Created {LOG_FILE}")
    else:
        print(f"{LOG_FILE} already exists — appending.")

def log_run(
    run_id: int,
    question_id: str,
    category: str,
    condition: str,
    query: str,
    chunks: list[dict],
    llm_result: dict,
    faithfulness_score: float,
):
    """
    Append one row to raw_log.csv.

    Args:
        run_id:             sequential run number (1-165)
        question_id:        e.g. "S-01"
        category:           S / M2 / M3 / C / T
        condition:          Baseline-LLM / RAG-Standard / RAG-AuthAware
        query:              the question text
        chunks:             list of retrieved chunk dicts (empty for Baseline-LLM)
        llm_result:         dict from m5_llm_caller.call_llm()
        faithfulness_score: float from compute_faithfulness(), 0.0 for Baseline-LLM
    """
    chunk_ids   = " | ".join(c.get("chunk_id", "")         for c in chunks)
    authorities = " | ".join(c.get("source_authority", "") for c in chunks)
    years       = " | ".join(c.get("award_year", "")       for c in chunks)
    doc_names   = " | ".join(c.get("document_name", "")    for c in chunks)

    row = {
        "run_id":                   run_id,
        "question_id":              question_id,
        "category":                 category,
        "condition":                condition,
        "query":                    query,
        "retrieved_chunk_ids":      chunk_ids,
        "retrieved_authority_tags": authorities,
        "retrieved_award_years":    years,
        "retrieved_documents":      doc_names,
        "answer":                   llm_result.get("answer", ""),
        "faithfulness":             faithfulness_score,
        "prompt_tokens":            llm_result.get("prompt_tokens", 0),
        "completion_tokens":        llm_result.get("completion_tokens", 0),
        "total_tokens":             llm_result.get("total_tokens", 0),
        "model_name":               llm_result.get("model_version", ""),
        "timestamp":                datetime.now(timezone.utc).isoformat(),
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)

def get_run_count() -> int:
    """Return the number of completed runs in the log."""
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE, encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f))

if __name__ == "__main__":
    init_log()
    # Test log entry
    test_chunks = [
        {"chunk_id": "chunk_000001", "source_authority": "federal-primary",
         "award_year": "2025-26", "document_name": "vol7_ch2_202526.txt",
         "text": "test", "distance": 0.28},
    ]
    test_llm = {
        "answer": "The maximum Pell Grant is $7,395.",
        "model_version": "gpt-4o-mini-2024-07-18",
        "prompt_tokens": 120,
        "completion_tokens": 20,
        "total_tokens": 140,
    }
    log_run(
        run_id=1,
        question_id="S-01",
        category="S",
        condition="RAG-Standard",
        query="What is the maximum Pell Grant for 2025-26?",
        chunks=test_chunks,
        llm_result=test_llm,
        faithfulness_score=0.95,
    )
    print(f"Test row logged. Total rows: {get_run_count()}")