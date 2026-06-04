"""
Week 2 — Chunk, Embed, and Build ChromaDB Vector Store
Reads all corpus text files, chunks them, tags with metadata,
embeds with all-MiniLM-L6-v2, and indexes into ChromaDB.
"""

import os
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import tiktoken

# ── Config ───────────────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens
CHUNK_OVERLAP = 50    # tokens
COLLECTION    = "financial_aid_corpus"
DB_PATH       = "vectorstore"

# ── Authority metadata map ───────────────────────────────────
CORPUS_META = {
    "corpus/federal_primary": {
        "source_authority": "federal-primary",
        "award_year": "2025-26",
    },
    "corpus/federal_outdated": {
        "source_authority": "federal-outdated",
        "award_year": "2024-25",
    },
    "corpus/federal_consumer": {
        "source_authority": "federal-consumer",
        "award_year": "2025-26",
    },
    "corpus/professional": {
        "source_authority": "professional",
        "award_year": "N/A",
    },
    "corpus/institutional": {
        "source_authority": "institutional",
        "award_year": "N/A",
    },
}

# ── Tokenizer for chunk sizing ───────────────────────────────
enc = tiktoken.encoding_for_model("gpt-4o-mini")  # o200k_base — correct tokenizer for gpt-4o-mini

def token_len(text):
    return len(enc.encode(text))

# ── Text splitter ────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Embedding model ──────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── ChromaDB ─────────────────────────────────────────────────
print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)

# Delete existing collection if rebuilding
try:
    client.delete_collection(COLLECTION)
    print("  Deleted existing collection.")
except:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# ── Process all files ────────────────────────────────────────
total_chunks = 0
stats = {}
chunk_id = 0

for folder, meta in CORPUS_META.items():
    if not os.path.exists(folder):
        print(f"Folder not found, skipping: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not files:
        print(f"No .txt files in {folder}, skipping.")
        continue

    folder_chunks = 0
    folder_tokens = 0

    print(f"\nProcessing: {folder} ({len(files)} files)")

    for filename in files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        if len(text.strip()) < 100:
            print(f"  Skipping (too short): {filename}")
            continue

        chunks = splitter.split_text(text)

        ids       = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            cid = f"chunk_{chunk_id:06d}"
            chunk_id += 1

            chunk_meta = {
                "source_authority": meta["source_authority"],
                "document_name":    filename,
                "award_year":       meta["award_year"],
                "chunk_id":         cid,
            }

            embedding = embedder.encode(chunk).tolist()

            ids.append(cid)
            documents.append(chunk)
            metadatas.append(chunk_meta)
            embeddings.append(embedding)

            folder_tokens += token_len(chunk)

        # Add to ChromaDB in batches
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        folder_chunks += len(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    total_chunks += folder_chunks
    stats[folder] = {
        "files": len(files),
        "chunks": folder_chunks,
        "total_tokens": folder_tokens,
        "avg_chunk_tokens": round(folder_tokens / folder_chunks) if folder_chunks else 0,
    }

# ── Corpus statistics ────────────────────────────────────────
print(f"\n{'='*60}")
print("CORPUS STATISTICS (Table I)")
print(f"{'='*60}")
print(f"{'Source':<25} {'Files':>6} {'Chunks':>8} {'Tokens':>10} {'Avg/Chunk':>10}")
print("-" * 65)

grand_chunks = 0
grand_tokens = 0

for folder, s in stats.items():
    label = folder.replace("corpus/", "")
    print(f"{label:<25} {s['files']:>6} {s['chunks']:>8} {s['total_tokens']:>10,} {s['avg_chunk_tokens']:>10}")
    grand_chunks += s["chunks"]
    grand_tokens += s["total_tokens"]

print("-" * 65)
print(f"{'TOTAL':<25} {'':>6} {grand_chunks:>8} {grand_tokens:>10,}")
print(f"\nTotal chunks indexed into ChromaDB: {grand_chunks}")

# Save stats to JSON for the report
with open("corpus_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Corpus stats saved to corpus_stats.json")
print("\nDone! Vector store is ready at:", DB_PATH)