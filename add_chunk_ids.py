"""
Week 3 — Chunk ID Lookup
Queries ChromaDB for each of the 55 questions and records
the top matching chunk IDs in questions_55.csv.
"""

import csv
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH    = "vectorstore"
COLLECTION = "financial_aid_corpus"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=DB_PATH)
col      = client.get_collection(COLLECTION)

# Read questions
with open("questions_55.csv", encoding="utf-8") as f:
    rows = list(csv.reader(f))

headers = rows[0]
data    = rows[1:]

# Add chunk_ids column if not already present
if "chunk_ids_top5" not in headers:
    headers.append("chunk_ids_top5")

print(f"Looking up chunk IDs for {len(data)} questions...\n")

for row in data:
    # Pad row to match headers length
    while len(row) < len(headers):
        row.append("")

    question = row[2]
    qid      = row[0]

    embedding = embedder.encode(question).tolist()

    results = col.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["metadatas", "distances"]
    )

    top_meta  = results["metadatas"][0]
    top_ids   = [m["chunk_id"] for m in top_meta]
    top_docs  = [m["document_name"] for m in top_meta]

    chunk_id_str = " | ".join(
        f"{cid} ({doc})" for cid, doc in zip(top_ids, top_docs)
    )

    row[headers.index("chunk_ids_top5")] = chunk_id_str
    print(f"{qid}: {top_ids[0]} ... ({top_docs[0]})")

# Write updated CSV
with open("questions_55.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    writer.writerows(data)

print("\nDone — chunk IDs written to questions_55.csv")