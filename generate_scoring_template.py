"""
generate_scoring_template.py
Generates manual_scores.csv from questions_55.csv —
a template for scoring all 165 answers for Answer Correctness.
Run this after the full experiment completes.
"""

import csv
import os

QUESTIONS_FILE = "questions_55.csv"
RAW_LOG        = os.path.join("results", "raw_log.csv")
OUTPUT         = os.path.join("results", "manual_scores.csv")
CONDITIONS     = ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]

os.makedirs("results", exist_ok=True)

# Load ground truth answers
ground_truth = {}
with open(QUESTIONS_FILE, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ground_truth[row["question_id"]] = {
            "category":    row["category"],
            "question":    row["question_text"],
            "ground_truth": row["ground_truth_answer"],
            "source_priority": row["source_priority_note"],
        }

# Load answers from raw log
answers = {}
with open(RAW_LOG, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["question_id"], row["condition"])
        answers[key] = row["answer"]

# Write scoring template
fields = [
    "question_id", "category", "condition",
    "question_text", "ground_truth_answer",
    "llm_answer", "source_priority_note",
    "answer_correctness",  # fill in: 0, 0.5, or 1
    "scorer_notes",        # fill in: brief justification
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for qid, gt in ground_truth.items():
        for condition in CONDITIONS:
            writer.writerow({
                "question_id":         qid,
                "category":            gt["category"],
                "condition":           condition,
                "question_text":       gt["question"],
                "ground_truth_answer": gt["ground_truth"],
                "llm_answer":          answers.get((qid, condition), ""),
                "source_priority_note": gt["source_priority"],
                "answer_correctness":  "",   # annotator fills this in
                "scorer_notes":        "",   # annotator fills this in
            })

print(f"Scoring template written to {OUTPUT}")
print(f"165 rows ready for annotation (0 = wrong, 0.5 = partially correct, 1 = correct)")