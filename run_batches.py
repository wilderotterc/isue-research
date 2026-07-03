"""
run_batches.py
Runs the full 165-run experiment in configurable batches.
Usage:
    python run_batches.py --batch 1   # questions 1-20
    python run_batches.py --batch 2   # questions 21-40
    python run_batches.py --batch 3   # questions 41-55
    python run_batches.py --validate  # validate merged log
"""

import csv
import argparse
import os
from datetime import datetime, timezone

from m1_indexer    import get_collection
from m2_retriever  import retrieve
from m3_reranker   import rerank
from m4_prompt_builder import build_prompt
from m5_llm_caller import call_llm
from m6_logger     import init_log, log_run, get_run_count, LOG_FILE, FIELDS
from metrics       import compute_faithfulness, compute_source_authority_accuracy

QUESTIONS_FILE = "questions_55.csv"
TARGET_YEAR    = "2025-26"
CONDITIONS     = ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]
RESULTS_DIR    = "results"
MERGED_LOG     = os.path.join(RESULTS_DIR, "raw_log.csv")

BATCH_RANGES = {
    1: (0,  20),   # questions index 0-19  (S-01 to M2-05)
    2: (20, 40),   # questions index 20-39 (M2-06 to M3-10)
    3: (40, 55),   # questions index 40-54 (C-01 to T-10)
}

def load_questions():
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(row)
    return questions

def run_batch(batch_num: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch_file = os.path.join(RESULTS_DIR, f"batch_{batch_num}.csv")

    start_idx, end_idx = BATCH_RANGES[batch_num]
    all_questions = load_questions()
    questions = all_questions[start_idx:end_idx]

    print(f"\n{'='*60}")
    print(f"BATCH {batch_num} — Questions {start_idx+1}–{end_idx} ({len(questions)} questions, {len(questions)*3} runs)")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    col = get_collection()
    print(f"ChromaDB: {col.count()} chunks\n")

    # Use batch file as log for this batch
    global LOG_FILE
    import m6_logger
    m6_logger.LOG_FILE = batch_file

    init_log()
    run_id = (start_idx * 3) + 1  # ensure unique run IDs across batches

    for q in questions:
        qid      = q["question_id"]
        category = q["category"]
        query    = q["question_text"]

        print(f"--- {qid} [{category}] ---")

        for condition in CONDITIONS:
            print(f"  {condition}...", end=" ", flush=True)

            if condition == "Baseline-LLM":
                chunks = []
            elif condition == "RAG-Standard":
                chunks = retrieve(query)
            elif condition == "RAG-AuthAware":
                chunks = rerank(retrieve(query), target_year=TARGET_YEAR)

            system_prompt, user_message = build_prompt(query, chunks)
            llm_result = call_llm(system_prompt, user_message)

            contexts = [c["text"] for c in chunks]
            if condition == "Baseline-LLM" or not contexts:
                faithfulness_score = 0.0
            else:
                faithfulness_score = compute_faithfulness(
                    query, llm_result["answer"], contexts
                )

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

    with open(batch_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    actual_count = len(rows)
    print(f"\nBatch {batch_num} complete. Rows in {batch_file}: {actual_count}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

    # Quick batch validation
    print("\nBatch validation:")
    expected = len(questions) * 3
    print(f"  Row count: {actual_count} (expected {expected}) {'✅' if actual_count == expected else '❌'}")
    empty = [r for r in rows if not r.get("answer","").strip()]
    print(f"  Empty answers: {len(empty)} {'✅' if len(empty)==0 else '❌'}")
    missing_f = [r for r in rows if r.get("condition") != "Baseline-LLM" and not r.get("faithfulness","").strip()]
    print(f"  Missing faithfulness: {len(missing_f)} {'✅' if len(missing_f)==0 else '❌'}")

def validate():
    """Validate the merged raw_log.csv."""
    if not os.path.exists(MERGED_LOG):
        print(f"❌ Merged log not found: {MERGED_LOG}")
        print("Run --batch 1, 2, 3 first, then fix_run_ids.py, then --validate")
        return

    with open(MERGED_LOG, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Show batch breakdown
    for batch_num in [1, 2, 3]:
        batch_file = os.path.join(RESULTS_DIR, f"batch_{batch_num}.csv")
        if os.path.exists(batch_file):
            with open(batch_file, encoding="utf-8") as f:
                count = sum(1 for _ in csv.DictReader(f))
            print(f"  Batch {batch_num}: {count} rows")

    print(f"\nTotal rows: {len(all_rows)} (expected 165) {'✅' if len(all_rows)==165 else '❌'}")
    print(f"Validating: {MERGED_LOG}")

    run_ids = [r["run_id"] for r in all_rows]
    print(f"Duplicate run IDs: {len(run_ids) - len(set(run_ids))} {'✅' if len(run_ids)==len(set(run_ids)) else '❌'}")

    conditions = set(r["condition"] for r in all_rows)
    print(f"Conditions present: {conditions} {'✅' if len(conditions)==3 else '❌'}")

    for cond in CONDITIONS:
        count = sum(1 for r in all_rows if r["condition"] == cond)
        print(f"  {cond}: {count} rows {'✅' if count==55 else '❌'}")

    categories = set(r["category"] for r in all_rows)
    print(f"Categories present: {categories} {'✅' if len(categories)==5 else '❌'}")

    empty = [r for r in all_rows if not r.get("answer","").strip()]
    print(f"Empty answers: {len(empty)} {'✅' if len(empty)==0 else '❌'}")

    invalid_f = [r for r in all_rows if r.get("condition") != "Baseline-LLM" and float(r.get("faithfulness",-1)) < 0]
    print(f"Invalid faithfulness scores: {len(invalid_f)} {'✅' if len(invalid_f)==0 else '❌'}")

    print("\n✅ Validation complete." if len(all_rows)==165 and len(empty)==0 else "\n❌ Issues found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1,2,3], help="Run batch 1, 2, or 3")
    parser.add_argument("--validate", action="store_true", help="Merge and validate all batches")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    elif args.validate:
        validate()
    else:
        print("Usage: python run_batches.py --batch 1|2|3  OR  --validate")