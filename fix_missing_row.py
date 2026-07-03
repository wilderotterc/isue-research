"""
Run the single missing row: S-06 Baseline-LLM
and append it to results/batch_1.csv
"""

import csv
import os
from m4_prompt_builder import build_prompt
from m5_llm_caller import call_llm
from m6_logger import log_run, FIELDS
from metrics import compute_faithfulness
from datetime import datetime, timezone

import m6_logger
m6_logger.LOG_FILE = os.path.join("results", "batch_1.csv")

query = "How much of my attempted credits do I need to complete to keep my federal financial aid?"
qid = "S-06"
category = "S"
condition = "Baseline-LLM"
chunks = []

system_prompt, user_message = build_prompt(query, chunks)
llm_result = call_llm(system_prompt, user_message)
faithfulness_score = 0.0  # Baseline-LLM always 0

# Get current max run_id from file and add 1
with open(m6_logger.LOG_FILE, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
max_run_id = max(int(r["run_id"]) for r in rows)
run_id = max_run_id + 1

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

print(f"Added missing row: {qid} {condition} (run_id={run_id})")
print(f"Answer: {llm_result['answer'][:100]}...")

# Verify
with open(m6_logger.LOG_FILE, encoding="utf-8") as f:
    total = sum(1 for _ in csv.DictReader(f))
print(f"batch_1.csv now has {total} rows {'✅' if total == 60 else '❌'}")