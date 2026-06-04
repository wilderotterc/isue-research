"""
M4 — Prompt Builder
Formats retrieved chunks as numbered context blocks
with the grounding instruction.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about US student financial aid. "
    "Answer using ONLY the information in the context below. "
    "If the context does not contain enough information to answer the question, "
    "say so clearly."
)

def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build the system prompt and user message for the LLM.

    Args:
        query:  the student's question
        chunks: list of chunk dicts from retriever or re-ranker
                (empty list for Baseline-LLM condition)

    Returns:
        (system_prompt, user_message) tuple
    """
    if not chunks:
        # Baseline-LLM: no context
        return SYSTEM_PROMPT, query

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document_name", "unknown")
        authority = chunk.get("source_authority", "unknown")
        year = chunk.get("award_year", "N/A")
        text = chunk.get("text", "")
        context_blocks.append(
            f"[{i}] Source: {source} | Authority: {authority} | Year: {year}\n{text}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    user_message = f"""Context:

{context_str}

---

Question: {query}"""

    return SYSTEM_PROMPT, user_message

if __name__ == "__main__":
    test_chunks = [
        {
            "chunk_id": "c1",
            "document_name": "vol7_ch2_202526.txt",
            "source_authority": "federal-primary",
            "award_year": "2025-26",
            "text": "The maximum Federal Pell Grant award is $7,395 for the 2025-26 award year.",
            "distance": 0.28,
        },
        {
            "chunk_id": "c2",
            "document_name": "pell_grant.txt",
            "source_authority": "federal-consumer",
            "award_year": "2025-26",
            "text": "Pell Grants do not have to be repaid.",
            "distance": 0.31,
        },
    ]

    system, user = build_prompt("What is the maximum Pell Grant?", test_chunks)
    print("SYSTEM PROMPT:")
    print(system)
    print("\nUSER MESSAGE:")
    print(user)

    # Test baseline (no chunks)
    system_b, user_b = build_prompt("What is the maximum Pell Grant?", [])
    print("\nBASELINE (no context):")
    print(f"System: {system_b}")
    print(f"User: {user_b}")
    assert user_b == "What is the maximum Pell Grant?", "Baseline should pass query directly"
    print("\nAll assertions passed.")