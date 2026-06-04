"""
Week 2 — Retrieval Quality Audit
Queries ChromaDB with the 10 example questions and checks
whether the expected ground-truth chunk appears in top-5 results.
Run this after re-indexing with the corrected tokenizer.
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH   = "vectorstore"
COLLECTION = "financial_aid_corpus"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=DB_PATH)
col      = client.get_collection(COLLECTION)

# Each question paired with the expected source document and authority
QUESTIONS = [
    {
        "id": "S-01",
        "category": "S",
        "question": "What is the maximum Pell Grant award for the 2025-26 award year?",
        "expected_source": "vol7_ch2_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "S-02",
        "category": "S",
        "question": "What is the FAFSA deadline for the 2025-26 award year?",
        "expected_source": "fafsa_deadlines.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "M2-01",
        "category": "M2",
        "question": "If I take out a subsidized loan, do I accrue interest while enrolled in school?",
        "expected_source": "subsidized_loans.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "M2-02",
        "category": "M2",
        "question": "Can I receive both a Pell Grant and a Federal Work-Study award at the same time?",
        "expected_source": "pell_grant.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "M3-01",
        "category": "M3",
        "question": "Can I receive a Pell Grant, work-study, and a subsidized loan simultaneously, and is there a total aid cap?",
        "expected_source": "vol3_ch3_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "M3-02",
        "category": "M3",
        "question": "What happens to my financial aid if I drop below half-time enrollment and I have both federal loans and a university scholarship?",
        "expected_source": "vol5_ch1_202526.txt",
        "expected_authority": "federal-primary",
    },
    {
        "id": "C-01",
        "category": "C",
        "question": "Does the federal government require undergraduate students to complete 80% of their attempted credits to keep aid?",
        "expected_source": "uw_sap.txt",
        "expected_authority": "institutional",
    },
    {
        "id": "C-02",
        "category": "C",
        "question": "Is a 2.5 GPA required by the federal government for education students to maintain aid eligibility?",
        "expected_source": "pace_sap.txt",
        "expected_authority": "institutional",
    },
    {
        "id": "T-01",
        "category": "T",
        "question": "How is my financial need calculated — what formula does the government use?",
        "expected_source": "how_aid_calculated.txt",
        "expected_authority": "federal-consumer",
    },
    {
        "id": "T-02",
        "category": "T",
        "question": "What information from my tax returns is used to determine my financial aid eligibility?",
        "expected_source": "avg_ch2_202526.txt",
        "expected_authority": "federal-primary",
    },
]

print("=" * 70)
print("RETRIEVAL QUALITY AUDIT")
print("=" * 70)

results_log = []
passes = 0
fails  = 0

for q in QUESTIONS:
    embedding = embedder.encode(q["question"]).tolist()

    results = col.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    top_docs  = results["documents"][0]
    top_meta  = results["metadatas"][0]
    top_dists = results["distances"][0]

    # Check if expected source appears in top-5
    top_sources = [m["document_name"] for m in top_meta]
    found = q["expected_source"] in top_sources
    rank  = top_sources.index(q["expected_source"]) + 1 if found else None

    status = f"PASS (rank {rank})" if found else "FAIL — not in top-5"
    if found:
        passes += 1
    else:
        fails += 1

    print(f"\n{q['id']} [{q['category']}] — {status}")
    print(f"  Question: {q['question'][:70]}...")
    print(f"  Expected: {q['expected_source']} ({q['expected_authority']})")
    print(f"  Top-5 results:")
    for i, (meta, dist) in enumerate(zip(top_meta, top_dists), 1):
        marker = " <-- EXPECTED" if meta["document_name"] == q["expected_source"] else ""
        print(f"    {i}. {meta['document_name']} | {meta['source_authority']} | dist={dist:.4f}{marker}")

    results_log.append({
        "question_id":       q["id"],
        "category":          q["category"],
        "question":          q["question"],
        "expected_source":   q["expected_source"],
        "found_in_top5":     found,
        "rank":              rank,
        "top5_sources":      top_sources,
        "top5_distances":    [round(d, 4) for d in top_dists],
    })

print(f"\n{'=' * 70}")
print(f"SUMMARY: {passes}/10 passed | {fails}/10 failed")
print(f"{'=' * 70}")

# Save log
import json
with open("retrieval_audit.json", "w") as f:
    json.dump(results_log, f, indent=2)
print("Full results saved to retrieval_audit.json")