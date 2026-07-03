"""
fix_run_ids.py
Reassigns clean sequential run IDs (1-165) to results/raw_log.csv
"""
import csv
import os

LOG = os.path.join("results", "raw_log.csv")

with open(LOG, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

for i, row in enumerate(rows, 1):
    row["run_id"] = str(i)

with open(LOG, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)

# Verify
with open(LOG, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ids = [int(r["run_id"]) for r in rows]
print(f"Rows: {len(rows)}")
print(f"Run IDs: {min(ids)} to {max(ids)}")
print(f"Duplicates: {len(ids) - len(set(ids))} {'✅' if len(ids)==len(set(ids)) else '❌'}")
print("Done.")