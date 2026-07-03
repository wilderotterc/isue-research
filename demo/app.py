"""
demo/app.py
Interactive demo for the IUSE RAG Financial Aid project.
Demonstrates all three experimental conditions with color-coded
source authority badges and RAGAS-style Faithfulness scoring.

Usage:
    streamlit run demo/app.py

Requirements (install in project venv):
    pip install streamlit openai chromadb sentence-transformers
"""

import streamlit as st
import sys
import os

# Add parent directory to path so pipeline modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Aid RAG Demo",
    page_icon="🎓",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
.badge-federal-primary  { background:#22c55e; color:white; padding:2px 8px;
                           border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-federal-outdated { background:#ef4444; color:white; padding:2px 8px;
                           border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-federal-consumer { background:#3b82f6; color:white; padding:2px 8px;
                           border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-institutional    { background:#f59e0b; color:white; padding:2px 8px;
                           border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-professional     { background:#8b5cf6; color:white; padding:2px 8px;
                           border-radius:4px; font-size:0.75rem; font-weight:600; }
.chunk-box { border:1px solid #e2e8f0; border-radius:6px; padding:12px;
              margin-bottom:8px; background:#f8fafc; color:#1e293b !important; }
.answer-box { border:2px solid #2E4057; border-radius:8px; padding:16px;
               background:#f0f4f8; color:#1e293b !important; }
.metric-label { font-size:0.85rem; color:#64748b; font-weight:600; }
.finding-box { border-left:4px solid #f59e0b; padding:10px 14px;
                background:#fffbeb; border-radius:0 6px 6px 0;
                margin-top:12px; font-size:0.9rem; color:#1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Authority badge HTML ───────────────────────────────────────
BADGE_CLASS = {
    "federal-primary":  "badge-federal-primary",
    "federal-outdated": "badge-federal-outdated",
    "federal-consumer": "badge-federal-consumer",
    "institutional":    "badge-institutional",
    "professional":     "badge-professional",
}
BADGE_LABEL = {
    "federal-primary":  "✓ Federal Primary (2025-26)",
    "federal-outdated": "⚠ Federal Outdated (2024-25)",
    "federal-consumer": "ℹ Federal Consumer",
    "institutional":    "⚡ Institutional",
    "professional":     "★ Professional",
}

def authority_badge(auth):
    cls   = BADGE_CLASS.get(auth, "badge-professional")
    label = BADGE_LABEL.get(auth, auth)
    return f'<span class="{cls}">{label}</span>'

# ── Example questions ──────────────────────────────────────────
EXAMPLE_QUESTIONS = {
    "S — Pell Grant maximum (S-01)":
        "What is the maximum Pell Grant I can get for the 2025-26 school year?",
    "M2 — Subsidized loan interest (M2-01)":
        "If I take out a subsidized loan, do I have to pay interest while I'm still in school?",
    "M3 — Multiple aid types (M3-01)":
        "Can I get a Pell Grant, a Work-Study job, and a loan all at the same time, and is there a limit to how much I can get total?",
    "C — 80% completion rate (C-01) ← shows re-ranking effect":
        "Does the government require me to pass 80% of my classes to keep my financial aid?",
    "C — SAP maximum timeframe (C-04) ← re-ranker helps here":
        "Does the federal government set a maximum time limit for how long I can take to finish my degree and still get aid?",
    "T — How aid is calculated (T-01) ← temporal confusion":
        "How does the government figure out how much financial aid I need?",
    "T — FAFSA changes (T-03) ← 2025-26 simplification act":
        "What changed about the FAFSA recently?",
}

# ── Pipeline loader ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline and vector store…")
def load_pipeline():
    """Load retriever and re-ranker. Cached so it only runs once."""
    from m2_retriever import retrieve
    from m3_reranker import rerank
    from m4_prompt_builder import build_prompt
    from m5_llm_caller import call_llm
    from metrics import compute_faithfulness
    return retrieve, rerank, build_prompt, call_llm, compute_faithfulness

# ── Main UI ────────────────────────────────────────────────────
st.title("Financial Aid  Interactive Demo")
st.caption(
    "Evaluating Multi-Document Retrieval-Augmented Generation for Student "
    "Financial Aid Guidance · Kean University · Caleb Wilderotter · 2026"
)

st.markdown("""
This demo replicates the three experimental conditions from the research paper.
Ask any financial aid question and compare how each condition answers it.
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    condition = st.selectbox(
        "Condition",
        ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"],
        help=(
            "**Baseline-LLM**: no retrieval — model uses training data only.\n\n"
            "**RAG-Standard**: top-5 chunks by cosine similarity.\n\n"
            "**RAG-AuthAware**: top-5 chunks re-ranked by source authority "
            "(federal-primary 2025-26 > federal-consumer > professional > "
            "institutional > federal-outdated)."
        ),
    )
    st.divider()
    st.subheader("Legend")
    for auth, label in BADGE_LABEL.items():
        cls = BADGE_CLASS[auth]
        st.markdown(f'<span class="{cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown("")
    st.divider()
    st.subheader("About")
    st.markdown(
        "Corpus: **1,641 chunks** from FSA Handbook (2024-25 & 2025-26), "
        "studentaid.gov, NASFAA, and 3 university SAP pages.\n\n"
        "Model: **gpt-4o-mini** (temp=0)\n\n"
        "Embeddings: **all-MiniLM-L6-v2**"
    )

# Question input
col1, col2 = st.columns([3, 1])
with col1:
    example_key = st.selectbox(
        "Or choose an example question:",
        ["— type your own below —"] + list(EXAMPLE_QUESTIONS.keys()),
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)

if example_key != "— type your own below —":
    default_q = EXAMPLE_QUESTIONS[example_key]
else:
    default_q = ""

query = st.text_area(
    "Your question:",
    value=default_q,
    height=80,
    placeholder="e.g. How much of my attempted credits do I need to complete to keep my aid?",
)

run_btn = st.button("▶ Run", type="primary", disabled=not query.strip())

# ── Run pipeline ───────────────────────────────────────────────
if run_btn and query.strip():
    try:
        retrieve, rerank, build_prompt, call_llm, compute_faithfulness = load_pipeline()
    except ImportError as e:
        st.error(
            f"Pipeline modules not found: {e}\n\n"
            "Make sure you run this app from the project root directory:\n"
            "```\nstreamlit run demo/app.py\n```"
        )
        st.stop()

    with st.spinner("Running pipeline…"):

        # Step 1: Retrieve
        if condition == "Baseline-LLM":
            chunks = []
        else:
            chunks = retrieve(query)

        # Step 2: Re-rank (AuthAware only)
        if condition == "RAG-AuthAware" and chunks:
            chunks = rerank(chunks, target_year="2025-26")

        # Step 3: Build prompt and call LLM
        system_prompt, user_message = build_prompt(query, chunks)
        result = call_llm(system_prompt, user_message)
        answer = result.get("answer", "")

        # Step 4: Faithfulness
        if chunks and answer:
            context_list = [c.get("text", c.get("document", "")) for c in chunks]
            faithfulness = compute_faithfulness(query, answer, context_list)
        else:
            faithfulness = 0.0

    # ── Display results ────────────────────────────────────────
    st.divider()

    left, right = st.columns(2)

    # Left: retrieved chunks
    with left:
        st.subheader(f"Retrieved Chunks ({len(chunks)})")
        if not chunks:
            st.info("No retrieval in Baseline-LLM condition.")
        else:
            for i, chunk in enumerate(chunks, 1):
                auth = chunk.get("source_authority", "unknown")
                doc  = chunk.get("document_name", chunk.get("document", ""))
                year = chunk.get("award_year", "")
                text = chunk.get("text", chunk.get("document", ""))[:400]
                dist = chunk.get("distance", None)
                rank_label = f"Rank {i}"
                if condition == "RAG-AuthAware":
                    priority = chunk.get("rerank_priority", "?")
                    rank_label += f" (priority {priority})"
                dist_str = f" · dist={dist:.4f}" if dist is not None else ""
                st.markdown(
                    f'<div class="chunk-box">'
                    f'<b>{rank_label}</b> {authority_badge(auth)}'
                    f'<br><small><b>{doc}</b>{" · " + year if year else ""}{dist_str}</small>'
                    f'<br><br>{text}…'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Right: answer and metrics
    with right:
        st.subheader("Generated Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Faithfulness (F)", f"{faithfulness:.2f}")
        with m2:
            st.metric("Condition", condition.replace("RAG-", "RAG\u2011"))
        with m3:
            st.metric("Chunks retrieved", len(chunks))

        # Contextual insight box
        insight = ""
        if "C-01" in example_key or "80%" in query:
            if condition == "RAG-AuthAware":
                insight = (
                    "**C-category insight:** The re-ranker promotes federal-primary "
                    "chunks above institutional ones. But for this question the "
                    "institutional source (UW 80% rule) is what a student would "
                    "encounter — promoting federal chunks may hide the contradiction "
                    "the question is designed to surface."
                )
            elif condition == "RAG-Standard":
                insight = (
                    "**C-category insight:** Without re-ranking, cosine similarity "
                    "tends to surface the institutional SAP page (UW 80% rule) at "
                    "rank 1. This is the correct source for surfacing the conflict, "
                    "but a naïve system might report 80% as if it were federal policy."
                )
        elif "T-0" in example_key or "FAFSA" in query.upper() or "SAI" in query or "EFC" in query:
            if condition == "RAG-AuthAware":
                insight = (
                    "**T-category insight:** The re-ranker demotes 2024-25 federal "
                    "handbook chunks to rank 4-5, helping ensure the 2025-26 "
                    "answer (SAI, not EFC) is surfaced first. This is the re-ranker's "
                    "intended use case."
                )
            elif condition == "RAG-Standard":
                insight = (
                    "**T-category insight:** Without re-ranking, cosine similarity "
                    "may retrieve both the 2024-25 and 2025-26 handbook chunks. "
                    "The model must infer which year's rule applies — a common "
                    "source of temporal confusion."
                )
        if insight:
            st.markdown(
                f'<div class="finding-box">{insight}</div>',
                unsafe_allow_html=True,
            )

        # Token usage
        with st.expander("Token usage"):
            st.json({
                "prompt_tokens":     result.get("prompt_tokens", "—"),
                "completion_tokens": result.get("completion_tokens", "—"),
                "total_tokens":      result.get("total_tokens", "—"),
                "model":             result.get("model_name", "gpt-4o-mini"),
            })

# ── Side-by-side comparison mode ──────────────────────────────
st.divider()
with st.expander("Compare all three conditions side by side"):
    st.markdown(
        "Click **Run comparison** to send the same question to all three conditions "
        "simultaneously. Uses ~3× the API tokens."
    )
    compare_btn = st.button("Run comparison", disabled=not query.strip())
    if compare_btn and query.strip():
        try:
            retrieve, rerank, build_prompt, call_llm, compute_faithfulness = load_pipeline()
            results_all = {}
            with st.spinner("Running all three conditions…"):
                for cond in ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]:
                    if cond == "Baseline-LLM":
                        ch = []
                    else:
                        ch = retrieve(query)
                    if cond == "RAG-AuthAware" and ch:
                        ch = rerank(ch, target_year="2025-26")
                    sp, um = build_prompt(query, ch)
                    res = call_llm(sp, um)
                    ans = res.get("answer", "")
                    ctx_list = [c.get("text", c.get("document","")) for c in ch]
                    f   = compute_faithfulness(query, ans, ctx_list) if ch and ans else 0.0
                    results_all[cond] = {"answer": ans, "faithfulness": f, "chunks": ch}

            c1, c2, c3 = st.columns(3)
            for col, cond in zip([c1, c2, c3], ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]):
                with col:
                    r = results_all[cond]
                    st.markdown(f"**{cond}**")
                    st.metric("Faithfulness", f"{r['faithfulness']:.2f}")
                    st.markdown(
                        f'<div class="answer-box" style="font-size:0.85rem">{r["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
        except ImportError as e:
            st.error(f"Pipeline modules not found: {e}")
