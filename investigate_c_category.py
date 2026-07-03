"""
investigate_c_category.py
For each C-category question under RAG-AuthAware, prints the full
retrieval order and identifies where the expected source authority lands.
"""

from m2_retriever import retrieve
from m3_reranker import rerank
import csv

TARGET_YEAR = "2025-26"

# Expected authority for each C-category question
expected_authority = {
    "C-01": "institutional",   # UW 80% vs federal 67%
    "C-02": "institutional",   # Pace 2.5 GPA vs federal 2.0
    "C-03": "institutional",   # Pace tiered 70% vs federal 67%
    "C-04": "federal-primary", # Federal 150% timeframe standard
    "C-05": "institutional",   # UW vs federal GPA conflict
}

questions = {}
with open("questions_55.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["category"] == "C":
            questions[row["question_id"]] = row["question_text"]

print("=" * 80)
print("C-CATEGORY RAG-AUTHAWARE RETRIEVAL POSITION ANALYSIS")
print("=" * 80)

for qid, query in questions.items():
    expected = expected_authority[qid]

    print(f"\n{'-'*80}")
    print(f"{qid}: {query}")
    print(f"Expected authority: {expected}")
    print(f"{'-'*80}")

    standard_chunks = retrieve(query)
    auth_chunks = rerank(retrieve(query), target_year=TARGET_YEAR)

    print("\n  RAG-Standard order:")
    std_rank = None
    for i, c in enumerate(standard_chunks, 1):
        marker = ""
        if c["source_authority"] == expected and std_rank is None:
            std_rank = i
            marker = " <-- EXPECTED AUTHORITY"
        print(f"    {i}. {c['document_name']} | {c['source_authority']} | dist={c['distance']:.4f}{marker}")

    print("\n  RAG-AuthAware order:")
    auth_rank = None
    for i, c in enumerate(auth_chunks, 1):
        marker = ""
        if c["source_authority"] == expected and auth_rank is None:
            auth_rank = i
            marker = " <-- EXPECTED AUTHORITY"
        print(f"    {i}. {c['document_name']} | {c['source_authority']} | priority={c.get('rerank_priority','?')} | dist={c['distance']:.4f}{marker}")

    print(f"\n  Position summary: RAG-Standard rank {std_rank} -> RAG-AuthAware rank {auth_rank}")
    if std_rank and auth_rank and auth_rank > std_rank:
        print(f"  >>> RE-RANKER PUSHED EXPECTED SOURCE DOWN by {auth_rank - std_rank} position(s)")
    elif std_rank and auth_rank and auth_rank < std_rank:
        print(f"  >>> RE-RANKER PROMOTED EXPECTED SOURCE UP by {std_rank - auth_rank} position(s)")
    elif std_rank == auth_rank:
        print(f"  >>> NO CHANGE IN POSITION")

print("\n" + "=" * 80)
print("Done.")