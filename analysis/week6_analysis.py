"""
week6_analysis.py
Produces Tables II-VI and Figures 1-4 for Week 6.

Usage:
    python week6_analysis.py

Input:  results/results_with_metrics.csv
Output: results/table2_descriptive.csv
        results/table3_anova.csv
        results/table4_regression.csv
        results/table5_ttests.csv
        results/table6_error_taxonomy.csv
        figures/figure1_faithfulness.pdf / .png
        figures/figure2_correctness.pdf / .png
        figures/figure3_degradation.pdf / .png
        figures/figure4_source_authority.pdf / .png
"""

import csv
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, ttest_rel, linregress

INPUT = "results/results_with_metrics.csv"
RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CATEGORIES = ["S", "M2", "M3", "C", "T"]
CONDITIONS = ["Baseline-LLM", "RAG-Standard", "RAG-AuthAware"]
SOURCE_COUNT = {"S": 1, "M2": 2, "M3": 3}  # for regression — only S/M2/M3 have a clean count

# ── IEEE-style plotting defaults ──────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

COND_COLORS = {
    "Baseline-LLM":  "#999999",
    "RAG-Standard":  "#4A6FA5",
    "RAG-AuthAware": "#2E7D5B",
}
COND_LABELS = {
    "Baseline-LLM":  "Baseline-LLM",
    "RAG-Standard":  "RAG-Standard",
    "RAG-AuthAware": "RAG-AuthAware",
}

# ── Load data ──────────────────────────────────────────────────
rows = []
with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} rows")

for r in rows:
    r["faithfulness"] = float(r["faithfulness"])
    r["answer_correctness"] = float(r["answer_correctness"])
    r["source_authority_accuracy"] = float(r["source_authority_accuracy"])

def values(metric, cat=None, cond=None):
    out = []
    for r in rows:
        if cat is not None and r["category"] != cat:
            continue
        if cond is not None and r["condition"] != cond:
            continue
        out.append(r[metric])
    return out

# ═══════════════════════════════════════════════════════════════
# TABLE II — Descriptive statistics (mean, SD) for all 15 conditions
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE II — Descriptive Statistics")
print("="*70)

table2_rows = []
for cat in CATEGORIES:
    for cond in CONDITIONS:
        for metric, label in [("faithfulness","F"), ("answer_correctness","A"), ("source_authority_accuracy","S")]:
            vals = values(metric, cat, cond)
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            table2_rows.append({
                "category": cat, "condition": cond, "metric": label,
                "n": len(vals), "mean": round(mean,4), "sd": round(sd,4),
            })

with open(f"{RESULTS_DIR}/table2_descriptive.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["category","condition","metric","n","mean","sd"])
    w.writeheader()
    w.writerows(table2_rows)

# Print readable summary
for metric, label in [("faithfulness","Faithfulness"), ("answer_correctness","Answer Correctness"), ("source_authority_accuracy","Source Authority Accuracy")]:
    print(f"\n{label}:")
    print(f"{'Category':10}", end="")
    for cond in CONDITIONS:
        print(f"{cond:>18}", end="")
    print()
    for cat in CATEGORIES:
        print(f"{cat:10}", end="")
        for cond in CONDITIONS:
            vals = values(metric, cat, cond)
            print(f"{np.mean(vals):>10.3f} (SD={np.std(vals,ddof=1) if len(vals)>1 else 0:.3f})", end="")
        print()

print("\nTable II saved.")

# ═══════════════════════════════════════════════════════════════
# TABLE III — One-way ANOVA for each metric across 5 categories
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE III — One-Way ANOVA (across 5 categories)")
print("="*70)

table3_rows = []
for metric, label in [("faithfulness","Faithfulness"), ("answer_correctness","Answer Correctness"), ("source_authority_accuracy","Source Authority Accuracy")]:
    groups = [values(metric, cat=cat) for cat in CATEGORIES]
    f_stat, p_val = f_oneway(*groups)
    df_between = len(CATEGORIES) - 1
    df_within = sum(len(g) for g in groups) - len(CATEGORIES)
    table3_rows.append({
        "metric": label, "F_statistic": round(f_stat,4),
        "df_between": df_between, "df_within": df_within, "p_value": round(p_val,6),
        "significant_at_05": "Yes" if p_val < 0.05 else "No",
    })
    print(f"\n{label}: F({df_between},{df_within}) = {f_stat:.4f}, p = {p_val:.6f} {'***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns'))}")

with open(f"{RESULTS_DIR}/table3_anova.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric","F_statistic","df_between","df_within","p_value","significant_at_05"])
    w.writeheader()
    w.writerows(table3_rows)
print("\nTable III saved.")

# ═══════════════════════════════════════════════════════════════
# TABLE IV — Linear regression of A on source count, per condition
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE IV — Linear Regression: Answer Correctness ~ Source Count")
print("="*70)

table4_rows = []
for cond in CONDITIONS:
    x = []
    y = []
    for r in rows:
        if r["condition"] != cond:
            continue
        if r["category"] in SOURCE_COUNT:
            x.append(SOURCE_COUNT[r["category"]])
            y.append(r["answer_correctness"])
    x = np.array(x); y = np.array(y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    table4_rows.append({
        "condition": cond, "slope": round(slope,4), "intercept": round(intercept,4),
        "r_squared": round(r_squared,4), "p_value": round(p_value,6), "n": len(x),
    })
    print(f"\n{cond}: slope = {slope:.4f}, R² = {r_squared:.4f}, p = {p_value:.6f} {'(significant)' if p_value<0.05 else '(not significant)'}")

with open(f"{RESULTS_DIR}/table4_regression.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition","slope","intercept","r_squared","p_value","n"])
    w.writeheader()
    w.writerows(table4_rows)
print("\nTable IV saved.")

# ═══════════════════════════════════════════════════════════════
# TABLE V — Bonferroni-corrected paired t-tests: RAG-Standard vs RAG-AuthAware
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE V — Paired t-tests: RAG-Standard vs RAG-AuthAware (Bonferroni α=0.0167)")
print("="*70)

ALPHA_CORR = 0.05 / 3  # 0.0167

table5_rows = []
for cat in CATEGORIES:
    std_vals = []
    auth_vals = []
    # Pair by question_id to ensure correct pairing
    std_by_q = {r["question_id"]: r["answer_correctness"] for r in rows if r["category"]==cat and r["condition"]=="RAG-Standard"}
    auth_by_q = {r["question_id"]: r["answer_correctness"] for r in rows if r["category"]==cat and r["condition"]=="RAG-AuthAware"}
    common_qs = sorted(set(std_by_q) & set(auth_by_q))
    std_vals = [std_by_q[q] for q in common_qs]
    auth_vals = [auth_by_q[q] for q in common_qs]

    t_stat, p_val = ttest_rel(std_vals, auth_vals)
    if math.isnan(t_stat):
        # Identical paired values — no difference to test
        t_stat, p_val = 0.0, 1.0
    sig = "Yes" if p_val < ALPHA_CORR else "No"
    table5_rows.append({
        "category": cat, "n_pairs": len(common_qs),
        "mean_RAG_Standard": round(np.mean(std_vals),4),
        "mean_RAG_AuthAware": round(np.mean(auth_vals),4),
        "t_statistic": round(t_stat,4), "p_value": round(p_val,6),
        "significant_bonferroni": sig,
    })
    print(f"\n{cat}: t = {t_stat:.4f}, p = {p_val:.6f}, Std={np.mean(std_vals):.3f}, Auth={np.mean(auth_vals):.3f} -- {'SIGNIFICANT' if p_val<ALPHA_CORR else 'not significant'} (α_corr={ALPHA_CORR:.4f})")

with open(f"{RESULTS_DIR}/table5_ttests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["category","n_pairs","mean_RAG_Standard","mean_RAG_AuthAware","t_statistic","p_value","significant_bonferroni"])
    w.writeheader()
    w.writerows(table5_rows)
print("\nTable V saved.")

# ═══════════════════════════════════════════════════════════════
# TABLE VI — Error taxonomy for all A=0 answers
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TABLE VI — Error Taxonomy (A=0 answers)")
print("="*70)
print("\nNOTE: This requires manual classification based on scorer_notes.")
print("Auto-classifying based on note keywords, please review manually.\n")

def classify_error(note, category, condition):
    note_lower = note.lower()

    # Baseline-LLM has no retrieval step. A "no information available" answer
    # here reflects a training-data knowledge gap, not a retrieval failure.
    if condition == "Baseline-LLM":
        if any(phrase in note_lower for phrase in ["efc", "future tense", "prior year"]) or category == "T":
            return "temporal"
        return "generation"

    # For RAG conditions, check explicit failure type labels FIRST, since
    # these are the researcher's own determination and should take priority
    # over generic phrasing matches below
    if "generation failure" in note_lower:
        return "generation"
    if "retrieval failure" in note_lower:
        return "retrieval"
    if "authority confusion" in note_lower:
        return "authority"
    if "temporal confusion" in note_lower:
        return "temporal"

    # Fallback: infer from descriptive language when no explicit label given
    if any(phrase in note_lower for phrase in [
        "no information was found", "no information about",
        "no info was found", "not found in the context", "no relevant information"
    ]):
        return "retrieval"

    if any(phrase in note_lower for phrase in [
        "uw 80%", "comes from the university", "institutional policy"
    ]):
        return "authority"

    if any(phrase in note_lower for phrase in [
        "efc", "irs data retrieval tool", "irs-drt", "future tense", "prior year"
    ]) or category == "T":
        return "temporal"

    if "unhelpful non-answer" in note_lower:
        return "generation"

    return "generation"  # default fallback for unclassified A=0 cases

zero_rows = [r for r in rows if r["answer_correctness"] == 0.0]
print(f"Total A=0 answers: {len(zero_rows)}")

taxonomy = defaultdict(lambda: defaultdict(int))
table6_detail = []
for r in zero_rows:
    err_type = classify_error(r.get("scorer_notes",""), r["category"], r["condition"])
    taxonomy[r["category"]][err_type] += 1
    table6_detail.append({
        "question_id": r["question_id"], "category": r["category"], "condition": r["condition"],
        "error_type": err_type, "scorer_notes": r.get("scorer_notes","")[:100],
    })

error_types = ["retrieval", "generation", "authority", "temporal"]
table6_rows = []
for cat in CATEGORIES:
    row = {"category": cat}
    total = sum(taxonomy[cat].values())
    for et in error_types:
        row[et] = taxonomy[cat][et]
    row["total"] = total
    table6_rows.append(row)
    print(f"{cat}: " + ", ".join(f"{et}={taxonomy[cat][et]}" for et in error_types) + f", total={total}")

with open(f"{RESULTS_DIR}/table6_error_taxonomy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["category"]+error_types+["total"])
    w.writeheader()
    w.writerows(table6_rows)

with open(f"{RESULTS_DIR}/table6_error_detail.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["question_id","category","condition","error_type","scorer_notes"])
    w.writeheader()
    w.writerows(table6_detail)

print(f"\nTable VI saved (summary + detail for manual review).")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Faithfulness by category and condition
# ═══════════════════════════════════════════════════════════════
def grouped_bar(metric, title, ylabel, filename, ylim=(0,1.05)):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(CATEGORIES))
    width = 0.25

    for i, cond in enumerate(CONDITIONS):
        means = []
        sds = []
        for cat in CATEGORIES:
            vals = values(metric, cat, cond)
            means.append(np.mean(vals))
            sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=sds, capsize=2,
               label=COND_LABELS[cond], color=COND_COLORS[cond],
               edgecolor="black", linewidth=0.5, error_kw={"linewidth":0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_xlabel("Question Category")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}.pdf / .png")

print("\n" + "="*70)
print("FIGURES")
print("="*70)

grouped_bar("faithfulness", "Figure 1: Faithfulness by Category and Condition", "Faithfulness (F)", "figure1_faithfulness")
grouped_bar("answer_correctness", "Figure 2: Answer Correctness by Category and Condition", "Answer Correctness (A)", "figure2_correctness")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Multi-source degradation curve
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.5))
source_counts = [1, 2, 3]
cat_for_count = {1:"S", 2:"M2", 3:"M3"}

for cond in CONDITIONS:
    means = []
    for sc in source_counts:
        cat = cat_for_count[sc]
        vals = values("answer_correctness", cat, cond)
        means.append(np.mean(vals))
    ax.plot(source_counts, means, marker="o", label=COND_LABELS[cond],
            color=COND_COLORS[cond], linewidth=1.5, markersize=5)

    # Overlay regression line
    x_all = []
    y_all = []
    for r in rows:
        if r["condition"] != cond or r["category"] not in SOURCE_COUNT:
            continue
        x_all.append(SOURCE_COUNT[r["category"]])
        y_all.append(r["answer_correctness"])
    slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
    x_line = np.array([1, 3])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, linestyle="--", color=COND_COLORS[cond], linewidth=0.8, alpha=0.6)

ax.set_xticks(source_counts)
ax.set_xlabel("Number of Required Source Documents")
ax.set_ylabel("Mean Answer Correctness (A)")
ax.set_title("Figure 3: Multi-Source Degradation Curve", fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure3_degradation.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/figure3_degradation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figure3_degradation.pdf / .png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4 — Source Authority Accuracy for C and T only, Std vs Auth
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 3.2))
cats_ct = ["C", "T"]
x = np.arange(len(cats_ct))
width = 0.3

for i, cond in enumerate(["RAG-Standard", "RAG-AuthAware"]):
    means = []
    sds = []
    for cat in cats_ct:
        vals = values("source_authority_accuracy", cat, cond)
        means.append(np.mean(vals))
        sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    offset = (i - 0.5) * width
    ax.bar(x + offset, means, width, yerr=sds, capsize=2,
           label=COND_LABELS[cond], color=COND_COLORS[cond],
           edgecolor="black", linewidth=0.5, error_kw={"linewidth":0.8})

ax.set_xticks(x)
ax.set_xticklabels(cats_ct)
ax.set_xlabel("Question Category")
ax.set_ylabel("Source Authority Accuracy (S)")
ax.set_ylim(0, 1.15)
ax.set_title("Figure 4: Source Authority Accuracy (C, T categories)", fontsize=10)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure4_source_authority.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/figure4_source_authority.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figure4_source_authority.pdf / .png")

print("\n" + "="*70)
print("WEEK 6 ANALYSIS COMPLETE")
print("="*70)
print(f"\nTables saved to {RESULTS_DIR}/")
print(f"Figures saved to {FIG_DIR}/")
print("\nIMPORTANT: Table VI error classifications were auto-generated from")
print("scorer_notes keywords. Review results/table6_error_detail.csv manually")
print("before finalizing, especially the 'generation' fallback category.")
