# Pre-Registration Document
## Evaluating Multi-Document Retrieval-Augmented Generation for Student Financial Aid Guidance

**Student Researcher:** Caleb Wilderotter
**Supervisor:** Dr. Malihe Aliasgari
**Date:** May 2026
**Status:** Awaiting supervisor signature before experiment begins

---

This document records all experimental parameters, metric definitions, and hypothesis directions before any experiment is run. Nothing in this document may be changed after supervisor sign-off without written notification and a new version number.

---

### 1. Pipeline Parameters

**LLM**
- Model: gpt-4o-mini (version string: gpt-4o-mini-2024-07-18)
- Provider: OpenAI
- Temperature: 0 (deterministic output)
- Max tokens: 400
- Context window: 128,000 tokens
- Cost per million input tokens: $0.15 (output: $0.60)

**Embedding model**
- Model: all-MiniLM-L6-v2
- Provider: HuggingFace / sentence-transformers
- Dimensions: 384
- Distance metric: cosine

**Chunking**
- Chunk size: 400 tokens
- Chunk overlap: 50 tokens
- Tokenizer: o200k_base via tiktoken.encoding_for_model("gpt-4o-mini")
- Splitter: LangChain RecursiveCharacterTextSplitter
- Separators (in order): paragraph break, line break, sentence boundary, word boundary

**Retrieval**
- Top-k: 5 passages per query
- Vector store: ChromaDB (persistent, cosine distance)
- Total chunks indexed: 1,641

---

### Corpus Modifications

One corpus modification was made after the initial vector store build and before the experiment began. It is documented here in full.

**NASFAA 2025 National Profile PDF removed from active corpus**

During the S-category retrieval quality audit (Week 3), the file `nasfaa_2025_national_profile.pdf.txt` was identified as causing systematic retrieval failures. The file (68 chunks, 81,445 characters) contains broad statistical summaries of federal aid programs — average award amounts, program participation rates, national totals — rather than specific eligibility rules. Its large size and broad content caused it to dominate top-5 results for general financial aid queries, displacing more specific authoritative chunks.

The file was moved from `corpus/professional/` to `corpus/excluded/` and the ChromaDB collection was rebuilt. The total chunk count dropped from 1,709 to 1,641. The remaining professional source (`nasfaa_about_financial_aid.txt`, 6 chunks) was retained as it contains consumer-facing eligibility guidance rather than statistical aggregates.

This modification was made before any experimental runs. The pre-registration chunk count above reflects the post-modification state. The `nasfaa_about_financial_aid.txt` file remains in the professional folder and is indexed.

**Authority re-ranking (RAG-AuthAware only)**
Priority order applied during re-ranking:
1. federal-primary (2025-26 FSA Handbook) — highest
2. federal-consumer (studentaid.gov)
3. professional (NASFAA)
4. institutional (UW, Pace, U-M, College Board)
5. federal-outdated (2024-25 FSA Handbook) — deprioritised but not removed

For T-category questions, 2025-26 documents are promoted over 2024-25 documents within the same authority tier.

**System prompt (all conditions)**
```
You are a helpful assistant answering questions about US student financial aid.
Answer based only on the context provided. If the context does not contain
enough information to answer the question, say so clearly.
```

**Baseline-LLM condition**
No context is passed. The system prompt is used without any retrieved passages.

---

### 2. Experimental Conditions

| Condition | Description |
|-----------|-------------|
| Baseline-LLM | Question submitted directly to LLM with empty context. No retrieval. |
| RAG-Standard | Top-5 passages retrieved by cosine similarity from full mixed corpus. No re-ranking. |
| RAG-AuthAware | Top-5 passages retrieved then re-ranked by source authority and award year. |

Total runs: 55 questions × 3 conditions = 165 runs.

---

### 3. Benchmark

**Question counts by category:**

| Category | Count | Definition |
|----------|------:|-----------|
| S — Single source | 15 | Complete answer in one passage from one document |
| M2 — Two source | 15 | Requires combining exactly two documents |
| M3 — Three or more source | 10 | Requires integrating three or more documents |
| C — Contradictory source | 5 | At least one corpus document conflicts with the federal answer |
| T — Temporal change | 10 | Correct answer changed between 2024-25 and 2025-26 award year |
| **Total** | **55** | |

All questions are written in lay language as a first-generation college student would phrase them. Questions are fixed before the experiment begins and will not be revised based on results.

**Baseline retrieval test results (S-category, pre-experiment):**
13/15 S-questions retrieve the expected chunk in top-5 under RAG-Standard.
- S-06 is a documented baseline retrieval failure: a query about the federal 67% completion standard retrieves the UW institutional page (80% requirement) due to vocabulary overlap with the institutional source. This is a pre-experiment demonstration of the authority confusion failure mode.
- S-07 is a documented baseline retrieval failure: vocabulary mismatch between lay phrasing ("less than half-time") and technical chunk content ("enrollment intensity percentage").

---

### 4. Metrics

**Metric 1 — RAGAS Faithfulness (F)**
- Implementation: RAGAS library, automated
- Definition: Decomposes the generated answer into atomic claims and checks each claim against the retrieved passages. F = 1.0 means every claim is supported by retrieved text. F = 0 means every claim is unsupported.
- Range: [0, 1]
- Note: F measures grounding, not correctness. A system can score F = 1.0 while scoring A = 0 if the retrieved text itself is outdated or from a lower-authority source.

**Metric 2 — Answer Correctness (A)**
- Implementation: Human annotation against ground-truth answers in questions_55.csv
- Scoring: 0 = wrong, 0.5 = partially correct, 1 = correct
- For C-category: correct means following the federal source (priority 1)
- For T-category: correct means reflecting the 2025-26 rule
- Inter-rater protocol: supervisor independently scores a stratified 25% sample (~42 answers). Cohen's κ ≥ 0.70 required for annotation to be reported as reliable. All disagreements resolved by discussion with reference to ground-truth answers.

**Metric 3 — Source Authority Accuracy (S)**
- Implementation: Rule-based check
- Scoring: 1 if the answer references a source consistent with the highest-authority annotation; 0 otherwise
- Purpose: Quantifies whether the system correctly resolves conflicting or competing sources. Primarily meaningful for C and T categories.

---

### 5. Research Questions and Hypothesis Directions

**RQ1.** Does RAG improve answer accuracy and grounding over a plain-LLM baseline, and does the improvement differ across question categories?

Hypothesis direction: RAG-Standard will outperform Baseline-LLM on Answer Correctness (A) for S and M2 questions where the correct passage is retrievable. The improvement will be smaller or absent for M3 questions requiring integration across multiple passages, and may be negative for T-category questions where the corpus contains both current and outdated documents.

**RQ2.** Does answer correctness degrade as the number of required source documents increases from one to two to three or more?

Hypothesis direction: A monotonic negative relationship between source count and Answer Correctness is expected under all three conditions. The slope will be steepest for RAG-Standard and shallower for RAG-AuthAware. Baseline-LLM degradation will reflect training data limitations rather than retrieval failures.

**RQ3.** Do temporally inconsistent documents and conflicting-authority documents cause answer errors, and does authority-aware re-ranking reduce those errors?

Hypothesis direction: RAG-Standard will produce lower Answer Correctness and Source Authority Accuracy on C and T categories than on S categories, due to authority confusion and temporal confusion failure modes. RAG-AuthAware will outperform RAG-Standard on C and T categories but not necessarily on S and M2 categories where re-ranking has no effect.

---

### 6. Statistical Analysis Plan

All analyses are performed in Python using scipy and statsmodels. All tests use α = 0.05 unless otherwise noted.

- **Descriptive statistics.** Mean (μ) and standard deviation (σ) for F, A, and S across all 15 conditions (5 categories × 3 experimental conditions). This produces Table II.
- **One-way ANOVA.** Tests whether question category significantly affects each metric. Reports F-statistic, degrees of freedom, and p-value.
- **Linear regression.** Regresses Answer Correctness A on source count (S → 1, M2 → 2, M3 → 3) for each experimental condition. Tests the monotonic degradation hypothesis. Reports slope, R², and p-value.
- **Bonferroni-corrected paired t-tests.** Compares RAG-Standard vs. RAG-AuthAware within each question category (α_corr = 0.05/3 = 0.0167).
- **Error taxonomy.** Every answer scored A = 0 classified into one of four types: retrieval failure, generation failure, authority confusion, or temporal confusion. Distribution reported as Table V.

---

### 7. Pre-Experiment Findings Already Documented

The following findings were observed during corpus construction and retrieval testing, before any experiment is run. They are recorded here to distinguish pre-experiment observations from experimental results.

- S-06 retrieval: A query about the federal SAP completion rate (67%) retrieves an institutional page (UW, 80%) at rank 1 under RAG-Standard. This confirms the authority confusion failure mode is present in the corpus before the experiment begins.
- C-02 retrieval (Week 2): A query about GPA requirements for education students retrieves Federal Vol. 1 Ch. 1 instead of the Pace institutional page, demonstrating that the institutional source is outvoted by federal content under RAG-Standard.
- T-01 retrieval (Week 2): A query about financial need calculation retrieves the consumer-facing studentaid.gov page rather than the technical FSA Handbook chapter, indicating that lay-phrased queries may preferentially retrieve consumer sources over primary sources.

---

### 8. Deviations Protocol

Any deviation from this pre-registration must be documented in the methods section of the final report with an explanation. Permitted deviations include: model version changes forced by API availability, minor prompt wording adjustments documented before re-running affected questions. Not permitted: changes to metric definitions, scoring rubrics, or hypothesis directions after results are partially observed.

---

**Supervisor sign-off**

I confirm that the pipeline parameters, metric definitions, and hypothesis directions recorded above are approved for the experiment to proceed.

Signature: _______________________________ Date: _______________

Dr. Malihe Aliasgari
