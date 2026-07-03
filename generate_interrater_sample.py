"""
generate_interrater_sample.py
Generates a stratified 25% sample (~42 answers) for supervisor
independent scoring. Stratified across all 5 categories and 3 conditions.
"""

import csv
import random
import os

random.seed(42)  # reproducible sample

INPUT  = "results/results_with_metrics.csv"
OUTPUT = "results/interrater_sample.csv"

rows = []
with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Stratify by category — sample ~8-9 per category (42 total)
from collections import defaultdict
by_cat = defaultdict(list)
for row in rows:
    by_cat[row["category"]].append(row)

sample = []
targets = {"S": 9, "M2": 9, "M3": 9, "C": 7, "T": 8}  # ~42 total

for cat, n in targets.items():
    cat_rows = by_cat[cat]
    # Ensure all 3 conditions represented
    by_cond = defaultdict(list)
    for r in cat_rows:
        by_cond[r["condition"]].append(r)
    
    per_cond = n // 3
    remainder = n % 3
    
    selected = []
    for cond in ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]:
        take = per_cond + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        pool = by_cond[cond]
        selected.extend(random.sample(pool, min(take, len(pool))))
    
    sample.extend(selected)

print(f"Sample size: {len(sample)}")

# Count by category
from collections import Counter
cat_counts = Counter(r["category"] for r in sample)
cond_counts = Counter(r["condition"] for r in sample)
print(f"By category: {dict(cat_counts)}")
print(f"By condition: {dict(cond_counts)}")

# Write sample with only what the supervisor needs
fields = [
    "run_id", "question_id", "category", "condition",
    "query", "answer",
    "ground_truth_answer",  # we'll add this from questions_55.csv
    "my_score",             # supervisor fills this in
    "supervisor_score",     # supervisor fills this in  
    "notes",
]

# Load ground truth
gt = {}
with open("questions_55.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gt[row["question_id"]] = row["ground_truth_answer"]

# Sort sample by category then question_id for easy reading
sample.sort(key=lambda r: (r["category"], r["question_id"], r["condition"]))

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in sample:
        writer.writerow({
            "run_id":               row["run_id"],
            "question_id":          row["question_id"],
            "category":             row["category"],
            "condition":            row["condition"],
            "query":                row["query"],
            "answer":               row["answer"],
            "ground_truth_answer":  gt.get(row["question_id"], ""),
            "my_score":             row["answer_correctness"],  # researcher's score pre-filled
            "supervisor_score":     "",  # supervisor fills this in
            "notes":                "",
        })

print(f"\nWritten to {OUTPUT}")
print("Supervisor should fill in 'supervisor_score' column (0, 0.5, or 1)")
print("Then upload back for Cohen's kappa calculation")