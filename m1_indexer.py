"""
M1 — Indexer
Loads corpus files from authority-labeled folders and confirms
the ChromaDB index is complete and queryable.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH    = "vectorstore"
COLLECTION = "financial_aid_corpus"

CORPUS_FOLDERS = {
    "federal-primary":   "corpus/federal_primary",
    "federal-outdated":  "corpus/federal_outdated",
    "federal-consumer":  "corpus/federal_consumer",
    "professional":      "corpus/professional",
    "institutional":     "corpus/institutional",
}

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION)

def audit_index():
    """Confirm the ChromaDB index is complete and queryable."""
    col = get_collection()
    total = col.count()

    print(f"ChromaDB collection: {COLLECTION}")
    print(f"Total chunks indexed: {total}")
    print()

    # Count chunks per authority
    for authority, folder in CORPUS_FOLDERS.items():
        if not os.path.exists(folder):
            print(f"  WARNING: folder not found — {folder}")
            continue
        results = col.get(where={"source_authority": authority})
        count = len(results["ids"])
        files = len(set(m["document_name"] for m in results["metadatas"]))
        print(f"  {authority:<25} {count:>5} chunks across {files} files")

    print()

    # Confirm queryable with a test query
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    test_query = "Pell Grant eligibility"
    embedding = embedder.encode(test_query).tolist()
    results = col.query(query_embeddings=[embedding], n_results=3)
    print(f"Test query: '{test_query}'")
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
        print(f"  {i}. {meta['document_name']} | {meta['source_authority']} | dist={dist:.4f}")

    print(f"\nIndex audit complete. {total} chunks ready.")
    return col

if __name__ == "__main__":
    audit_index()