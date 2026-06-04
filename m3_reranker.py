"""
M3 — Re-ranker (RAG-AuthAware only)
Given a list of retrieved chunks and a target award year,
re-ranks by source priority and demotes non-target-year chunks
to positions 4-5.

Priority order:
1. federal-primary (current year)
2. federal-consumer
3. professional
4. institutional
5. federal-primary (non-target year) / federal-outdated
"""

AUTHORITY_PRIORITY = {
    "federal-primary":  1,
    "federal-consumer": 2,
    "professional":     3,
    "institutional":    4,
    "federal-outdated": 5,
}

def rerank(chunks: list[dict], target_year: str = "2025-26") -> list[dict]:
    """
    Re-rank retrieved chunks by source authority and award year.

    Rules:
    - federal-primary chunks matching target_year get priority 1
    - federal-primary chunks NOT matching target_year get priority 5
      (demoted to positions 4-5)
    - All other authorities ranked by AUTHORITY_PRIORITY
    - Within the same priority, original cosine distance order preserved

    Args:
        chunks: list of chunk dicts from m2_retriever.retrieve()
        target_year: the award year to promote (default "2025-26")

    Returns:
        Re-ranked list of chunk dicts with added 'rerank_priority' field
    """
    def sort_key(chunk):
        authority = chunk["source_authority"]
        year      = chunk.get("award_year", "")

        if authority == "federal-primary":
            if year == target_year:
                priority = 1
            else:
                priority = 5  # demote outdated federal-primary
        else:
            priority = AUTHORITY_PRIORITY.get(authority, 9)

        # Secondary sort: original distance (lower = better)
        return (priority, chunk["distance"])

    ranked = sorted(chunks, key=sort_key)

    for i, chunk in enumerate(ranked):
        authority = chunk["source_authority"]
        year      = chunk.get("award_year", "")
        if authority == "federal-primary" and year == target_year:
            chunk["rerank_priority"] = 1
        elif authority == "federal-primary":
            chunk["rerank_priority"] = 5
        else:
            chunk["rerank_priority"] = AUTHORITY_PRIORITY.get(authority, 9)

    return ranked

if __name__ == "__main__":
    # Test with mock chunks
    test_chunks = [
        {"chunk_id": "c1", "source_authority": "federal-primary",  "award_year": "2024-25", "document_name": "avg_ch3_202425.txt", "distance": 0.31, "text": "old formula"},
        {"chunk_id": "c2", "source_authority": "federal-primary",  "award_year": "2025-26", "document_name": "avg_ch3_202526.txt", "distance": 0.33, "text": "new formula"},
        {"chunk_id": "c3", "source_authority": "institutional",    "award_year": "N/A",     "document_name": "uw_sap.txt",         "distance": 0.29, "text": "UW policy"},
        {"chunk_id": "c4", "source_authority": "federal-consumer", "award_year": "2025-26", "document_name": "how_aid_calculated.txt", "distance": 0.35, "text": "consumer page"},
        {"chunk_id": "c5", "source_authority": "professional",     "award_year": "N/A",     "document_name": "nasfaa_about.txt",   "distance": 0.38, "text": "NASFAA"},
    ]

    print("Before re-ranking:")
    for i, c in enumerate(test_chunks, 1):
        print(f"  {i}. [{c['source_authority']}] {c['document_name']} | year={c['award_year']} | dist={c['distance']}")

    ranked = rerank(test_chunks, target_year="2025-26")

    print("\nAfter re-ranking (target_year=2025-26):")
    for i, c in enumerate(ranked, 1):
        print(f"  {i}. [{c['source_authority']}] {c['document_name']} | year={c['award_year']} | priority={c['rerank_priority']} | dist={c['distance']}")

    # Verify: 2025-26 federal-primary should be rank 1, 2024-25 should be rank 4 or 5
    assert ranked[0]["award_year"] == "2025-26" and ranked[0]["source_authority"] == "federal-primary", "2025-26 federal-primary should be rank 1"
    assert ranked[-1]["award_year"] == "2024-25" or ranked[-1]["rerank_priority"] == 5, "2024-25 should be demoted"
    print("\nAll assertions passed.")