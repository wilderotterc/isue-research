"""
rebuild_merged_log.py
Reads all three batch files, assigns clean sequential run IDs 1-165,
and writes a clean results/raw_log.csv.
"""
import csv
import os

RESULTS_DIR = "results"
FIELDS = [
    "run_id", "question_id", "category", "condition", "query",
    "retrieved_chunk_ids", "retrieved_authority_tags", "retrieved_award_years",
    "retrieved_documents", "answer", "faithfulness", "prompt_tokens",
    "completion_tokens", "total_tokens", "model_name", "timestamp",
]

all_rows = []
for batch_num in [1, 2, 3]:
    path = os.path.join(RESULTS_DIR, f"batch_{batch_num}.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_rows.extend(rows)
    print(f"Batch {batch_num}: {len(rows)} rows")

print(f"Total before fix: {len(all_rows)}")

# Remove duplicate rows (keep last occurrence of each question/condition pair)
seen = {}
for row in all_rows:
    key = (row["question_id"], row["condition"])
    seen[key] = row  # last one wins

deduped = list(seen.values())
print(f"Total after dedup: {len(deduped)}")

# Sort by question_id then condition for clean ordering
condition_order = {"Baseline-LLM": 0, "RAG-Standard": 1, "RAG-AuthAware": 2}
deduped.sort(key=lambda r: (r["question_id"], condition_order.get(r["condition"], 9)))

# Assign clean sequential run IDs
for i, row in enumerate(deduped, 1):
    row["run_id"] = str(i)

# Write clean merged log
out = os.path.join(RESULTS_DIR, "raw_log.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(deduped)

# Final validation
with open(out, encoding="utf-8") as f:
    final = list(csv.DictReader(f))
ids = [int(r["run_id"]) for r in final]
print(f"\nFinal raw_log.csv:")
print(f"  Rows: {len(final)} {'✅' if len(final)==165 else '❌'}")
print(f"  Run IDs: {min(ids)} to {max(ids)} {'✅' if max(ids)==165 else '❌'}")
print(f"  Duplicates: {len(ids)-len(set(ids))} {'✅' if len(ids)==len(set(ids)) else '❌'}")
conditions = set(r['condition'] for r in final)
print(f"  Conditions: {conditions} {'✅' if len(conditions)==3 else '❌'}")
categories = set(r['category'] for r in final)
print(f"  Categories: {categories} {'✅' if len(categories)==5 else '❌'}")
empty = [r for r in final if not r.get('answer','').strip()]
print(f"  Empty answers: {len(empty)} {'✅' if len(empty)==0 else '❌'}")
print("\nDone.")