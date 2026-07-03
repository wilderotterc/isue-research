"""
run_experiment.py
Iterates over all 55 questions and all 3 conditions.
Produces exactly 165 rows in raw_log.csv.

Usage:
    python run_experiment.py           # full 165-run experiment
    python run_experiment.py --pilot   # 15-run pilot (1 per category x 3 conditions)
"""

import csv
import argparse
from datetime import datetime, timezone

from m1_indexer    import get_collection
from m2_retriever  import retrieve
from m3_reranker   import rerank
from m4_prompt_builder import build_prompt
from m5_llm_caller import call_llm
from m6_logger     import init_log, log_run, get_run_count
from metrics       import compute_faithfulness, compute_source_authority_accuracy

QUESTIONS_FILE = "questions_55.csv"
TARGET_YEAR    = "2025-26"
CONDITIONS     = ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]

# Pilot: one question per category
PILOT_IDS = ["S-01", "M2-01", "M3-01", "C-01", "T-01"]

def load_questions(pilot: bool = False) -> list[dict]:
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if pilot and row["question_id"] not in PILOT_IDS:
                continue
            questions.append(row)
    return questions

def run(pilot: bool = False):
    print(f"\n{'='*60}")
    print(f"RAG Experiment — {'PILOT (15 runs)' if pilot else 'FULL (165 runs)'}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # Confirm index
    col = get_collection()
    print(f"ChromaDB: {col.count()} chunks indexed\n")

    init_log()
    questions = load_questions(pilot=pilot)

    expected_runs = len(questions) * len(CONDITIONS)
    print(f"Questions loaded: {len(questions)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Expected runs: {expected_runs}\n")

    run_id = get_run_count() + 1

    for q in questions:
        qid      = q["question_id"]
        category = q["category"]
        query    = q["question_text"]

        print(f"--- {qid} [{category}] ---")

        for condition in CONDITIONS:
            print(f"  {condition}...", end=" ", flush=True)

            # ── Retrieve ──────────────────────────────────────────────
            if condition == "Baseline-LLM":
                chunks = []
            elif condition == "RAG-Standard":
                chunks = retrieve(query)
            elif condition == "RAG-AuthAware":
                chunks = retrieve(query)
                chunks = rerank(chunks, target_year=TARGET_YEAR)

            # ── Build prompt ──────────────────────────────────────────
            system_prompt, user_message = build_prompt(query, chunks)

            # ── Call LLM ──────────────────────────────────────────────
            llm_result = call_llm(system_prompt, user_message)

            # ── Compute faithfulness inline before logging ────────────
            contexts = [c["text"] for c in chunks]
            if condition == "Baseline-LLM" or not contexts:
                faithfulness_score = 0.0
            else:
                faithfulness_score = compute_faithfulness(
                    query, llm_result["answer"], contexts
                )

            # ── Compute source authority accuracy ─────────────────────
            retrieved_authorities = [c["source_authority"] for c in chunks]
            source_accuracy = compute_source_authority_accuracy(
                retrieved_authorities=retrieved_authorities,
                expected_authority=q.get("source_priority_note", "").split("(")[0].strip().split()[-1] if chunks else "",
                condition=condition,
            )

            # ── Log — faithfulness written to CSV in same row ─────────
            log_run(
                run_id=run_id,
                question_id=qid,
                category=category,
                condition=condition,
                query=query,
                chunks=chunks,
                llm_result=llm_result,
                faithfulness_score=faithfulness_score,
            )

            print(f"done (F={faithfulness_score:.2f}, tokens={llm_result['total_tokens']})")
            run_id += 1

    total = get_run_count()
    print(f"\n{'='*60}")
    print(f"Experiment complete. Total rows in raw_log.csv: {total}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    if pilot:
        print("\nPILOT CHECKS:")
        pilot_check(total)

def pilot_check(total_rows: int):
    """Manually inspect the pilot run for correctness."""
    print(f"\n  Row count: {total_rows} (expected 15)")
    assert total_rows == 15, f"Expected 15 rows, got {total_rows}"

    issues = []
    with open("raw_log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        # Check no empty answers
        if not row["answer"].strip():
            issues.append(f"{row['question_id']} {row['condition']}: empty answer")

        # Check authority metadata present for RAG conditions
        if row["condition"] != "Baseline-LLM":
            if not row["retrieved_authority_tags"].strip():
                issues.append(f"{row['question_id']} {row['condition']}: missing authority tags")
            if not row["retrieved_chunk_ids"].strip():
                issues.append(f"{row['question_id']} {row['condition']}: missing chunk IDs")

        # Check award year tags present for RAG conditions
        if row["condition"] != "Baseline-LLM":
            if not row["retrieved_award_years"].strip():
                issues.append(f"{row['question_id']} {row['condition']}: missing award years")

    # Check T-01 AuthAware: 2025-26 chunk should be in positions 1-2
    t01_authaware = [r for r in rows if r["question_id"] == "T-01" and r["condition"] == "RAG-AuthAware"]
    if t01_authaware:
        years = t01_authaware[0]["retrieved_award_years"].split(" | ")
        if len(years) >= 2:
            if "2025-26" not in years[:2]:
                issues.append("T-01 RAG-AuthAware: 2025-26 chunk not in positions 1-2")
            else:
                print("  T-01 AuthAware year check: PASS (2025-26 in positions 1-2)")
        if len(years) >= 4:
            if "2024-25" not in years[3:]:
                issues.append("T-01 RAG-AuthAware: 2024-25 chunk not demoted to positions 4-5")
            else:
                print("  T-01 AuthAware demotion check: PASS (2024-25 in positions 4-5)")

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All pilot checks passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 15-question pilot only")
    args = parser.parse_args()
    run(pilot=args.pilot)