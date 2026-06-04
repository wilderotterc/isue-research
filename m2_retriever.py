"""
M2 — Retriever
Given a query string, returns the top-k=5 chunks by cosine similarity
with their IDs, texts, metadata, and distances.
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH    = "vectorstore"
COLLECTION = "financial_aid_corpus"
TOP_K      = 5

_embedder   = None
_collection = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_collection(COLLECTION)
    return _collection

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k chunks for a query.

    Returns a list of dicts, each containing:
        chunk_id, text, source_authority, document_name,
        award_year, distance
    """
    embedder = _get_embedder()
    col      = _get_collection()

    embedding = embedder.encode(query).tolist()

    results = col.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "chunk_id":         meta.get("chunk_id", ""),
            "text":             text,
            "source_authority": meta.get("source_authority", ""),
            "document_name":    meta.get("document_name", ""),
            "award_year":       meta.get("award_year", ""),
            "distance":         round(dist, 6),
        })

    return chunks

if __name__ == "__main__":
    query = "What is the maximum Pell Grant for 2025-26?"
    print(f"Query: {query}\n")
    chunks = retrieve(query)
    for i, c in enumerate(chunks, 1):
        print(f"{i}. [{c['source_authority']}] {c['document_name']} | dist={c['distance']:.4f}")
        print(f"   {c['text'][:120]}...")
        print()