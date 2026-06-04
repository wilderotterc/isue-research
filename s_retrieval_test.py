"""
Week 3 — S-Question Retrieval Test
Queries ChromaDB with all 15 S-category questions and checks
whether the expected chunk appears in top-5 results.
Flags any baseline retrieval failures.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH    = "vectorstore"
COLLECTION = "financial_aid_corpus"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=DB_PATH)
col      = client.get_collection(COLLECTION)

S_QUESTIONS = [
    {
        "id": "S-01",
        "question": "What is the maximum Pell Grant I can get for the 2025-26 school year?",
        "expected_source": "vol7_ch2_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-02",
        "question": "When is the last day I can submit my FAFSA for the 2025-26 school year?",
        "expected_source": "fafsa_deadlines.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "S-03",
        "question": "How many years can I receive a Pell Grant in total?",
        "expected_source": "pell_grant.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "S-04",
        "question": "Do I have to pay back a Pell Grant?",
        "expected_source": "pell_grant.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "S-05",
        "question": "What GPA do I need to keep my federal financial aid?",
        "expected_source": "vol1_ch1_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-06",
        "question": "How much of my attempted credits do I need to complete to keep my federal financial aid?",
        "expected_source": "vol1_ch1_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-07",
        "question": "Can I get financial aid if I'm enrolled less than half-time or only part-time?",
        "expected_source": "vol7_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-08",
        "question": "What is the most I can borrow in federal loans as a first-year undergraduate?",
        "expected_source": "vol3_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-09",
        "question": "What is the most I can ever borrow in federal loans as an undergraduate student?",
        "expected_source": "vol3_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-10",
        "question": "Does my parents' income affect my financial aid even if I pay for school myself?",
        "expected_source": "avg_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-11",
        "question": "What happens to my financial aid if I repeat a class I already passed?",
        "expected_source": "vol1_ch1_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-12",
        "question": "Can I get federal financial aid if I owe money on a previous federal loan that I haven't paid back?",
        "expected_source": "vol1_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-13",
        "question": "How long do I have after I graduate before I have to start paying back my federal loans?",
        "expected_source": "subsidized_loans.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "S-14",
        "question": "What is Federal Work-Study?",
        "expected_source": "work_study.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "S-15",
        "question": "Does getting a scholarship affect how much federal aid I can receive?",
        "expected_source": "vol3_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
]

print("=" * 70)
print("S-QUESTION RETRIEVAL TEST — Week 3")
print("=" * 70)

results_log = []
passes = 0
fails  = 0
failures = []

for q in S_QUESTIONS:
    embedding = embedder.encode(q["question"]).tolist()

    results = col.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    top_meta  = results["metadatas"][0]
    top_dists = results["distances"][0]
    top_sources = [m["document_name"] for m in top_meta]

    found = q["expected_source"] in top_sources
    rank  = top_sources.index(q["expected_source"]) + 1 if found else None

    status = f"PASS (rank {rank})" if found else "FAIL — not in top-5"
    if found:
        passes += 1
    else:
        fails += 1
        failures.append(q["id"])

    print(f"\n{q['id']} — {status}")
    print(f"  Q: {q['question'][:70]}...")
    print(f"  Expected: {q['expected_source']} ({q['expected_authority']})")
    print(f"  Top-5:")
    for i, (meta, dist) in enumerate(zip(top_meta, top_dists), 1):
        marker = " <-- EXPECTED" if meta["document_name"] == q["expected_source"] else ""
        print(f"    {i}. {meta['document_name']} | {meta['source_authority']} | dist={dist:.4f}{marker}")

    results_log.append({
        "question_id":     q["id"],
        "question":        q["question"],
        "expected_source": q["expected_source"],
        "found_in_top5":   found,
        "rank":            rank,
        "top5_sources":    top_sources,
        "top5_distances":  [round(d, 4) for d in top_dists],
    })

print(f"\n{'=' * 70}")
print(f"SUMMARY: {passes}/15 passed | {fails}/15 failed")
if failures:
    print(f"Baseline retrieval failures: {', '.join(failures)}")
    print("These questions will be flagged in the results as retrieval failures.")
else:
    print("No baseline retrieval failures.")
print(f"{'=' * 70}")

with open("s_retrieval_test.json", "w") as f:
    json.dump(results_log, f, indent=2)
print("Results saved to s_retrieval_test.json")