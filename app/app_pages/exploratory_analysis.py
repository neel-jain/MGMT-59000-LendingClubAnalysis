"""
exploratory_analysis.py
=========================
Phase 5, Page 2: Exploratory Analysis.

Interactive re-rendering of Phase 2's key EDA visualizations
(`src/eda_utils.py`) against the sidebar-filtered borrower dataset. No
new analytical logic is implemented here -- every chart is produced by
calling the same reusable plotting functions Phase 2's notebook used.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from common import (
    advanced_statistics_enabled, apply_borrower_filters, apply_global_style, load_cleaned_dataset,
    render_missing_artifact_notice, render_page_header, render_section_header,
)
from src import config, eda_utils

apply_global_style()
render_page_header(
    "Exploratory Analysis",
    "Interactive borrower-distribution and default-rate charts, filtered live via the sidebar.",
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

st.caption(f"Showing {len(df):,} of {len(cleaned_df):,} loans based on current sidebar filters.")

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
render_section_header("Borrower & Loan Distributions")

tab_labels = ["Grade", "Purpose", "Home Ownership", "Income", "Loan Amount", "Interest Rate", "DTI"]
tabs = st.tabs(tab_labels)

with tabs[0]:
    fig = eda_utils.plot_categorical_distribution(
        df, "grade", title="Loan Grade Distribution", order_by_count=False,
    )
    st.pyplot(fig)
with tabs[1]:
    fig = eda_utils.plot_categorical_distribution(df, "purpose", title="Loan Purpose Distribution")
    st.pyplot(fig)
with tabs[2]:
    fig = eda_utils.plot_categorical_distribution(df, "home_ownership", title="Home Ownership Distribution")
    st.pyplot(fig)
with tabs[3]:
    fig = eda_utils.plot_numeric_distribution(df, "annual_inc", title="Annual Income Distribution")
    st.pyplot(fig)
with tabs[4]:
    fig = eda_utils.plot_numeric_distribution(df, "loan_amnt", title="Loan Amount Distribution")
    st.pyplot(fig)
with tabs[5]:
    fig = eda_utils.plot_numeric_distribution(df, "int_rate", title="Interest Rate Distribution")
    st.pyplot(fig)
with tabs[6]:
    fig = eda_utils.plot_numeric_distribution(df, "dti", title="Debt-to-Income Distribution")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Default rate by group
# ---------------------------------------------------------------------------
render_section_header("Default Rate by Borrower Group")

group_choice = st.selectbox(
    "Group default rate by:", ["grade", "purpose", "home_ownership"],
    format_func=lambda c: c.replace("_", " ").title(),
)
fig, _ = eda_utils.plot_default_rate_by_group(df, group_choice, title=f"Default Rate by {group_choice.replace('_', ' ').title()}")
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
render_section_header("Feature Correlation")

fig, _ = eda_utils.plot_correlation_heatmap(
    df, config.NUMERIC_FEATURES, title="Numeric Feature Correlation Matrix",
)
st.pyplot(fig)

if advanced_statistics_enabled():
    # ---------------------------------------------------------------------------
    # Scatter relationships
    # ---------------------------------------------------------------------------
    render_section_header("Bivariate Relationships")

    col1, col2 = st.columns(2)
    x_options = ["loan_amnt", "int_rate", "annual_inc", "dti"]
    with col1:
        x_var = st.selectbox("X axis", x_options, index=0, key="scatter_x")
    with col2:
        y_var = st.selectbox("Y axis", x_options, index=1, key="scatter_y")

    if x_var != y_var:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(df[x_var], df[y_var], c=df["default_flag"], cmap="RdBu_r", alpha=0.5, s=18)
        ax.set_xlabel(x_var.replace("_", " ").title())
        ax.set_ylabel(y_var.replace("_", " ").title())
        ax.set_title(
            f"{x_var.replace('_', ' ').title()} vs. {y_var.replace('_', ' ').title()} (colored by default)",
            fontsize=12, fontweight="bold", loc="left",
        )
        fig.colorbar(scatter, ax=ax, label="Default (1) / Fully Paid (0)")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Choose two different variables to compare.", icon="ℹ️")

with st.expander("View filtered data table"):
    st.dataframe(df)
