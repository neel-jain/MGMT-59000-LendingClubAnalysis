"""
executive_dashboard.py
========================
Phase 5, Page 1: Executive Dashboard.

KPIs, grade distribution, top risk factors, and an executive summary
for the full Indiana borrower portfolio. Reuses `RiskScoringEngine`'s
threshold configuration and `ExplainabilityEngine`'s global importance
for the "Top Risk Factors" panel -- no modeling logic lives in this
file.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from common import (
    apply_borrower_filters, apply_global_style, get_explainability_engine, get_global_explanation,
    load_cleaned_dataset, render_executive_summary_box, render_missing_artifact_notice,
    render_page_header, render_section_header,
)

apply_global_style()
render_page_header(
    "Executive Dashboard",
    "Portfolio-level KPIs and key risk drivers for Indiana LendingClub borrowers.",
)

cleaned_df = load_cleaned_dataset()
if cleaned_df.empty:
    render_missing_artifact_notice("The borrower dataset", "python -m src.train_models")
    st.stop()

filters = st.session_state.get("borrower_filters", {})
df = apply_borrower_filters(cleaned_df, filters)

if df.empty:
    st.info("No borrowers match the current sidebar filters. Adjust the filters to see results.", icon="ℹ️")
    st.stop()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
render_section_header("Portfolio KPIs")

row1 = st.columns(4)
row1[0].metric("Total Loans", f"{len(df):,}")
row1[1].metric("Default Rate", f"{df['default_flag'].mean():.1%}")
row1[2].metric("Average Income", f"${df['annual_inc'].mean():,.0f}")
row1[3].metric("Average Loan Amount", f"${df['loan_amnt'].mean():,.0f}")

row2 = st.columns(3)
row2[0].metric("Average Interest Rate", f"{df['int_rate'].mean():.2f}%")
row2[1].metric("Average DTI", f"{df['dti'].mean():.1f}%")
row2[2].metric("Average Employment Length", f"{df['emp_length_years'].mean():.1f} yrs")

# ---------------------------------------------------------------------------
# Grade distribution + default rate by grade
# ---------------------------------------------------------------------------
render_section_header("Loan Grade Distribution")

col1, col2 = st.columns(2)
with col1:
    grade_counts = df["grade"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(grade_counts.index, grade_counts.values, color="#2E86AB")
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Number of Loans")
    ax.set_title("Grade Distribution", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    default_by_grade = df.groupby("grade")["default_flag"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(default_by_grade.index, default_by_grade.values, color="#C0392B")
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by Grade", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Top risk factors (from ExplainabilityEngine)
# ---------------------------------------------------------------------------
render_section_header("Top Risk Factors (Production Model)")

model_key = st.session_state.get("selected_model_key", "xgboost")
explain_engine = get_explainability_engine(model_key)

if explain_engine is None:
    render_missing_artifact_notice("Model explainability artifacts", "python -m src.train_models")
else:
    global_explanation = get_global_explanation(explain_engine, model_key, 300)

    col1, col2 = st.columns([3, 2])
    with col1:
        importance_top = global_explanation.importance_table.head(8)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        plot_df = importance_top.sort_values("mean_abs_shap")
        ax.barh(plot_df["feature_label"], plot_df["mean_abs_shap"], color="#2E86AB")
        ax.set_xlabel("Mean |SHAP value| (log-odds)")
        ax.set_title("Top Risk-Driving Variables", fontsize=12, fontweight="bold", loc="left")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.markdown("**Most influential variables:**")
        for f in global_explanation.top_features[:5]:
            st.markdown(f"- {f}")

# ---------------------------------------------------------------------------
# Executive takeaways
# ---------------------------------------------------------------------------
render_section_header("Key Executive Takeaways")

riskiest_grade = df.groupby("grade")["default_flag"].mean().idxmax()
riskiest_rate = df.groupby("grade")["default_flag"].mean().max()
summary = (
    f"This portfolio of {len(df):,} Indiana loans shows an overall default rate of "
    f"{df['default_flag'].mean():.1%}, with Grade {riskiest_grade} loans carrying the highest "
    f"observed default rate ({riskiest_rate:.1%}). The average borrower earns "
    f"${df['annual_inc'].mean():,.0f} annually with a {df['dti'].mean():.1f}% debt-to-income ratio, "
    f"borrowing ${df['loan_amnt'].mean():,.0f} at a {df['int_rate'].mean():.2f}% interest rate. "
)
if explain_engine is not None:
    summary += global_explanation.business_summary
render_executive_summary_box(summary)

with st.expander("View filtered borrower data"):
    st.dataframe(df.head(200))
