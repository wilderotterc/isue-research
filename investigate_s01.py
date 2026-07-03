"""
Investigate S-01 RAG-AuthAware failure.
Retrieves and re-ranks chunks for S-01, then prints full text
of all retrieved chunks to check whether $7,395 appears.
"""

from m2_retriever import retrieve
from m3_reranker import rerank

query = "What is the maximum Pell Grant I can get for the 2025-26 school year?"

print(f"Query: {query}\n")
print("=" * 70)

# RAG-Standard (no re-ranking)
print("\nRAG-Standard (cosine similarity order):")
std_chunks = retrieve(query)
for i, c in enumerate(std_chunks, 1):
    print(f"\n  Rank {i}: {c['document_name']} | {c['source_authority']} | year={c['award_year']} | dist={c['distance']:.4f}")
    print(f"  Chunk ID: {c['chunk_id']}")
    print(f"  Full text:\n{c['text']}")
    print(f"  Contains '$7,395': {'$7,395' in c['text'] or '7,395' in c['text']}")

print("\n" + "=" * 70)

# RAG-AuthAware (re-ranked)
print("\nRAG-AuthAware (re-ranked):")
auth_chunks = rerank(retrieve(query), target_year="2025-26")
for i, c in enumerate(auth_chunks, 1):
    print(f"\n  Rank {i}: {c['document_name']} | {c['source_authority']} | year={c['award_year']} | priority={c.get('rerank_priority','?')} | dist={c['distance']:.4f}")
    print(f"  Chunk ID: {c['chunk_id']}")
    print(f"  Full text:\n{c['text']}")
    print(f"  Contains '$7,395': {'$7,395' in c['text'] or '7,395' in c['text']}")

print("\n" + "=" * 70)
print("\nSummary:")
print(f"  RAG-Standard rank 1 contains $7,395: {'$7,395' in std_chunks[0]['text'] or '7,395' in std_chunks[0]['text']}")
print(f"  RAG-AuthAware rank 1 contains $7,395: {'$7,395' in auth_chunks[0]['text'] or '7,395' in auth_chunks[0]['text']}")