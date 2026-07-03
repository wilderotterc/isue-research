import csv
import os

scores = {}
with open("results/manual_scores_scored.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        scores[(row["question_id"], row["condition"])] = {
            "answer_correctness": float(row["answer_correctness"]),
            "scorer_notes": row["scorer_notes"],
        }

PRIORITY = {
    "federal-primary": 1, "federal-consumer": 2,
    "professional": 3, "institutional": 4, "federal-outdated": 5,
}

source_priority_map = {
    "S-01":"federal-primary","S-02":"federal-consumer","S-03":"federal-consumer",
    "S-04":"federal-consumer","S-05":"federal-primary","S-06":"federal-primary",
    "S-07":"federal-primary","S-08":"federal-primary","S-09":"federal-primary",
    "S-10":"federal-primary","S-11":"federal-primary","S-12":"federal-primary",
    "S-13":"federal-consumer","S-14":"federal-consumer","S-15":"federal-primary",
    "M2-01":"federal-consumer","M2-02":"federal-consumer","M2-03":"federal-primary",
    "M2-04":"federal-primary","M2-05":"federal-primary","M2-06":"federal-primary",
    "M2-07":"federal-consumer","M2-08":"federal-primary","M2-09":"federal-primary",
    "M2-10":"federal-primary","M2-11":"federal-consumer","M2-12":"federal-consumer",
    "M2-13":"federal-primary","M2-14":"federal-consumer","M2-15":"federal-primary",
    "M3-01":"federal-primary","M3-02":"federal-primary","M3-03":"federal-primary",
    "M3-04":"federal-primary","M3-05":"federal-primary","M3-06":"federal-primary",
    "M3-07":"federal-primary","M3-08":"federal-primary","M3-09":"federal-primary",
    "M3-10":"federal-primary",
    "C-01":"federal-primary","C-02":"federal-primary","C-03":"federal-primary",
    "C-04":"federal-primary","C-05":"federal-primary",
    "T-01":"federal-primary","T-02":"federal-primary","T-03":"federal-primary",
    "T-04":"federal-primary","T-05":"federal-primary","T-06":"federal-primary",
    "T-07":"federal-primary","T-08":"federal-primary","T-09":"federal-primary",
    "T-10":"federal-primary",
}

def saa(auth_str, expected, condition):
    if condition == "Baseline-LLM":
        return 1
    auths = [a.strip() for a in auth_str.split("|") if a.strip()]
    if not auths:
        return 0
    if expected in auths:
        return 1
    ep = PRIORITY.get(expected, 9)
    return 1 if any(PRIORITY.get(a,9) < ep for a in auths) else 0

raw_rows = []
with open("results/raw_log.csv", encoding="utf-8") as f:
    raw_rows = list(csv.DictReader(f))

out_fields = list(raw_rows[0].keys()) + ["answer_correctness","source_authority_accuracy","scorer_notes"]

os.makedirs("results", exist_ok=True)
with open("results/results_with_metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=out_fields, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in raw_rows:
        key = (row["question_id"], row["condition"])
        s = scores.get(key, {"answer_correctness": -1, "scorer_notes": ""})
        expected = source_priority_map.get(row["question_id"], "federal-primary")
        row["answer_correctness"] = s["answer_correctness"]
        row["source_authority_accuracy"] = saa(row["retrieved_authority_tags"], expected, row["condition"])
        row["scorer_notes"] = s["scorer_notes"]
        writer.writerow(row)

print(f"Written results/results_with_metrics.csv")

# Summary
with open("results/results_with_metrics.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total rows: {len(rows)}")
from collections import defaultdict
by_cat = defaultdict(lambda: {"F":[],"A":[],"S":[]})
by_cond = defaultdict(lambda: {"F":[],"A":[],"S":[]})
for r in rows:
    cat = r["category"]
    cond = r["condition"]
    by_cat[cat]["F"].append(float(r["faithfulness"]))
    by_cat[cat]["A"].append(float(r["answer_correctness"]))
    by_cat[cat]["S"].append(float(r["source_authority_accuracy"]))
    by_cond[cond]["F"].append(float(r["faithfulness"]))
    by_cond[cond]["A"].append(float(r["answer_correctness"]))
    by_cond[cond]["S"].append(float(r["source_authority_accuracy"]))

print("\nBy category (F=Faithfulness, A=Correctness, S=SourceAuthority):")
for cat in ["S","M2","M3","C","T"]:
    d = by_cat[cat]
    print(f"  {cat}: F={sum(d['F'])/len(d['F']):.3f} A={sum(d['A'])/len(d['A']):.3f} S={sum(d['S'])/len(d['S']):.3f}")

print("\nBy condition:")
for cond in ["Baseline-LLM","RAG-Standard","RAG-AuthAware"]:
    d = by_cond[cond]
    print(f"  {cond}: F={sum(d['F'])/len(d['F']):.3f} A={sum(d['A'])/len(d['A']):.3f} S={sum(d['S'])/len(d['S']):.3f}")