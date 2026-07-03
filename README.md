# IUSE RAG: Evaluating Multi-Document Retrieval-Augmented Generation for Student Financial Aid Guidance

Undergraduate research project · Kean University · Caleb Wilderotter · Supervised by Dr. Malihe Aliasgari · June 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project evaluates whether retrieval-augmented generation (RAG) can reliably answer student financial aid questions when the underlying corpus contains legitimately conflicting sources and version-dependent correct answers.

**Three experimental conditions** are compared across 165 runs on a 55-question benchmark:

| Condition | Description |
|-----------|-------------|
| Baseline-LLM | No retrieval — model uses training data only |
| RAG-Standard | Top-5 chunks by cosine similarity, no authority awareness |
| RAG-AuthAware | Top-5 chunks re-ranked by source authority priority |

**Key findings:**
- RAG improves mean Answer Correctness over baseline (0.545 vs. 0.255, p<0.001)
- Authority-aware re-ranking helps temporal conflicts but hurts contradictory-source questions
- Faithfulness and Answer Correctness diverge by ~0.40 across all RAG runs — a systematic gap

---

## Repository Structure

```
corpus/
  federal_primary/        FSA Handbook 2025-26 (30 files, 785 chunks)
  federal_outdated/       FSA Handbook 2024-25 (30 files, 786 chunks)
  federal_consumer/       studentaid.gov pages (6 files, 41 chunks)
  professional/           NASFAA (1 file, 6 chunks)
  institutional/          University SAP pages (4 files, 23 chunks)

questions/
  questions_55.csv        55 annotated questions with ground truth and source labels

pipeline/
  m1_indexer.py           Corpus chunking and ChromaDB indexing
  m2_retriever.py         Top-k cosine similarity retrieval
  m3_reranker.py          Authority-aware re-ranking (RAG-AuthAware only)
  m4_prompt_builder.py    Context formatting and prompt assembly
  m5_llm_caller.py        gpt-4o-mini API calls (temp=0)
  m6_logger.py            Run logging to raw_log.csv
  metrics.py              Custom Faithfulness metric implementation
  run_batches.py          Batch experiment runner

results/
  raw_log.csv             165 runs, 16 fields each
  results_with_metrics.csv  All three metrics per run
  manual_scores.csv       Human annotation with scorer notes
  interrater_sample_v2.xlsx  42-item inter-rater validation (κ=0.887)

analysis/
  week6_analysis.py       Tables II-VI and Figures 1-4
  table2_descriptive.csv
  table3_anova.csv        Two-way ANOVA (Condition × Category)
  table4_regression.csv
  table5_ttests.csv
  table6_error_taxonomy.csv
  figure1_faithfulness.pdf/png
  figure2_correctness.pdf/png
  figure3_degradation.pdf/png
  figure4_source_authority.pdf/png

demo/
  app.py                  Streamlit interactive demo

report/
  report.tex              Complete IEEE-format LaTeX paper

README.md
```

---

## Reproducing the Experiment

### 1. Clone and install

```bash
git clone https://github.com/wilderotterc/isue-research.git
cd isue-research
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install openai chromadb sentence-transformers streamlit scipy statsmodels openpyxl
```

### 2. Set API key

```bash
export OPENAI_API_KEY=your_key_here     # Mac/Linux
$env:OPENAI_API_KEY="your_key_here"    # Windows PowerShell
```

### 3. Build the vector store

```bash
python pipeline/m1_indexer.py
```

### 4. Run the full experiment

```bash
python pipeline/run_batches.py --batch 1
python pipeline/run_batches.py --batch 2
python pipeline/run_batches.py --batch 3
python pipeline/run_batches.py --validate
```

### 5. Score and compute metrics

Score answers manually using `results/manual_scores.csv`, then:

```bash
python pipeline/build_results_with_metrics.py
```

### 6. Run statistical analysis and generate figures

```bash
python analysis/week6_analysis.py
```

### 7. Launch the interactive demo

```bash
streamlit run demo/app.py
```

---

## Benchmark Categories

| Category | n | Description |
|----------|---|-------------|
| S | 15 | Single-source — answer in one document |
| M2 | 15 | Two-source — requires combining two documents |
| M3 | 10 | Three-or-more-source — multi-hop integration required |
| C | 5 | Contradictory-source — institutional vs. federal conflict |
| T | 10 | Temporal — correct answer changed between 2024-25 and 2025-26 |

---

## Corpus

1,641 chunks (400 tokens, 50-token overlap, `o200k_base` tokenizer) across five authority tiers:

- **Federal-primary**: FSA Handbook 2025-26 (authoritative, current-year)
- **Federal-outdated**: FSA Handbook 2024-25 (retained to create genuine temporal conflicts)
- **Federal-consumer**: studentaid.gov guidance pages
- **Professional**: NASFAA about page
- **Institutional**: SAP policy pages from 3 universities (UW, Pace, U-M / College Board)

---

## Re-Ranking Algorithm

For RAG-AuthAware, chunks are sorted by priority π(c):

```
π(c) = 1  if federal-primary AND award year matches target
π(c) = 5  if federal-primary AND award year does NOT match target
π(c) = P(authority)  otherwise

where P: federal-consumer=2, professional=3, institutional=4
```

Ties broken by original cosine distance.

---

## Results Summary

| Metric | Baseline-LLM | RAG-Standard | RAG-AuthAware |
|--------|-------------|--------------|---------------|
| Faithfulness (F) | 0.000 | 0.929 | 0.938 |
| Answer Correctness (A) | 0.255 | 0.545 | 0.509 |
| Source Authority Accuracy (S) | 1.000* | 0.927 | 0.927 |

*Baseline-LLM assigned S=1 by convention (no retrieval step).

Two-way ANOVA: Condition effect on Correctness F(2,150)=8.87, p<0.001.  
Inter-rater reliability: Cohen's κ=0.887 on 42-item stratified sample.

---

## Citation

```
Wilderotter, C. (2026). Evaluating multi-document retrieval-augmented generation
for student financial aid guidance: Authority confusion, temporal confusion, and
the limits of faithfulness. Kean University Undergraduate Research Report.
https://github.com/wilderotterc/isue-research
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.  
Corpus documents are reproduced from publicly available government and institutional sources for research purposes.
