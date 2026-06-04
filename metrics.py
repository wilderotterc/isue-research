"""
metrics.py — Evaluation metrics
1. RAGAS Faithfulness (automated)
2. Answer Correctness (human annotation placeholder)
3. Source Authority Accuracy (rule-based)
"""

from dotenv import load_dotenv
import os
import openai
import json

load_dotenv()

# ── 1. Custom Faithfulness (replaces RAGAS) ──────────────────
# RAGAS library has breaking API changes across versions.
# This implementation replicates the RAGAS faithfulness logic
# directly using the OpenAI API: decompose answer into atomic
# claims, check each claim against context, return proportion supported.

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def compute_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Compute faithfulness score.
    Decomposes the answer into atomic claims and checks each
    against the retrieved contexts.

    Returns float in [0, 1]. Returns 0.0 if contexts are empty.
    Returns -1.0 on API error.
    """
    if not contexts or not answer.strip():
        return 0.0

    context_str = "\n\n".join(contexts)
    client = _get_client()

    try:
        # Step 1: Extract atomic claims from the answer
        claims_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "Extract all atomic factual claims from the answer. Return a JSON array of strings, one claim per item. Return only the JSON array, no other text."},
                {"role": "user", "content": f"Answer: {answer}"}
            ]
        )
        claims_text = claims_response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if claims_text.startswith("```"):
            claims_text = claims_text.split("```")[1]
            if claims_text.startswith("json"):
                claims_text = claims_text[4:]
        claims = json.loads(claims_text)
        if not claims:
            return 1.0

        # Step 2: Check each claim against context
        supported = 0
        for claim in claims:
            check_response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": "Answer only YES or NO. Is the following claim supported by the context?"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nClaim: {claim}"}
                ]
            )
            verdict = check_response.choices[0].message.content.strip().upper()
            if "YES" in verdict:
                supported += 1

        score = supported / len(claims)
        return round(score, 4)

    except Exception as e:
        print(f"  RAGAS error: {e}")
        return -1.0

# ── 2. Answer Correctness (human annotation placeholder) ─────

def compute_answer_correctness(question_id: str) -> float:
    """
    Placeholder for human annotation.
    Returns -1.0 to indicate annotation is pending.
    Human annotator replaces -1.0 with 0, 0.5, or 1.0.
    """
    return -1.0  # pending human annotation

# ── 3. Source Authority Accuracy (rule-based) ────────────────

SOURCE_PRIORITY = {
    "federal-primary":  1,
    "federal-consumer": 2,
    "professional":     3,
    "institutional":    4,
    "federal-outdated": 5,
}

def compute_source_authority_accuracy(
    retrieved_authorities: list[str],
    expected_authority: str,
    condition: str,
) -> int:
    """
    Rule-based check: did the answer reference a source consistent
    with the highest-authority annotation?

    Returns 1 if the expected authority appears in the retrieved set,
    0 otherwise. Always returns 1 for Baseline-LLM (no retrieval).

    Args:
        retrieved_authorities: list of source_authority strings from retrieved chunks
        expected_authority:    the annotated highest-priority source for this question
        condition:             Baseline-LLM / RAG-Standard / RAG-AuthAware
    """
    if condition == "Baseline-LLM":
        return 1  # no retrieval — metric not applicable, mark as N/A equivalent

    if expected_authority in retrieved_authorities:
        return 1

    # Also pass if a higher-priority source was retrieved
    expected_priority = SOURCE_PRIORITY.get(expected_authority, 9)
    for auth in retrieved_authorities:
        if SOURCE_PRIORITY.get(auth, 9) < expected_priority:
            return 1

    return 0

# ── Unit tests ───────────────────────────────────────────────

def run_unit_tests():
    print("Running metrics unit tests...\n")

    # Test 1: Source Authority Accuracy — exact match
    result = compute_source_authority_accuracy(
        retrieved_authorities=["federal-primary", "federal-consumer"],
        expected_authority="federal-primary",
        condition="RAG-Standard",
    )
    assert result == 1, f"Expected 1, got {result}"
    print("  Test 1 PASS: exact authority match returns 1")

    # Test 2: Source Authority Accuracy — miss
    result = compute_source_authority_accuracy(
        retrieved_authorities=["institutional", "professional"],
        expected_authority="federal-primary",
        condition="RAG-Standard",
    )
    assert result == 0, f"Expected 0, got {result}"
    print("  Test 2 PASS: authority miss returns 0")

    # Test 3: Source Authority Accuracy — Baseline always returns 1
    result = compute_source_authority_accuracy(
        retrieved_authorities=[],
        expected_authority="federal-primary",
        condition="Baseline-LLM",
    )
    assert result == 1, f"Expected 1, got {result}"
    print("  Test 3 PASS: Baseline-LLM always returns 1")

    # Test 4: Source Authority Accuracy — higher priority retrieved counts as pass
    result = compute_source_authority_accuracy(
        retrieved_authorities=["federal-primary"],
        expected_authority="federal-consumer",
        condition="RAG-Standard",
    )
    assert result == 1, f"Expected 1, got {result}"
    print("  Test 4 PASS: higher priority source retrieved counts as pass")

    # Test 5: Answer Correctness placeholder returns -1
    result = compute_answer_correctness("S-01")
    assert result == -1.0, f"Expected -1.0, got {result}"
    print("  Test 5 PASS: answer correctness placeholder returns -1.0")

    # Test 6: Faithfulness with empty context returns 0
    result = compute_faithfulness("test question", "test answer", [])
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("  Test 6 PASS: empty context faithfulness returns 0.0")

    print("\nAll unit tests passed.")

if __name__ == "__main__":
    run_unit_tests()