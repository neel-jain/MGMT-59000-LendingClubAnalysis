"""
business_insights.py
======================
Phase 5, Page 6: Business Insights.

Organized by the project's seven research questions. For each: the
question, the finding, supporting evidence, a visualization, a business
recommendation, and the decision impact -- reusing `eda_utils`,
`model_utils`, `explainability`, and `segmentation_engine` outputs
rather than recomputing any analysis inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from common import (
    apply_global_style, get_explainability_engine, get_segmentation_engine, load_cleaned_dataset,
    render_missing_artifact_notice, render_page_header,
)
from src import config, eda_utils

apply_global_style()
render_page_header(
    "Business Insights",
    "Findings organized by the project's research questions, in executive-report format.",
)

cleaned_df = load_cleaned_dataset()
if cleaned_df.empty:
    render_missing_artifact_notice("The borrower dataset", "python -m src.train_models")
    st.stop()

model_key = st.session_state.get("selected_model_key", config.PRODUCTION_MODEL_KEY)
explain_engine = get_explainability_engine(model_key)
segmentation_engine = get_segmentation_engine()
show_advanced_statistics = st.session_state.get("show_advanced_statistics", False)


def _research_question_block(number: str, question: str, finding: str, recommendation: str, decision_impact: str, viz_fn):
    """Render one research-question block in a consistent executive-report layout."""
    with st.container(border=True):
        st.markdown(f"#### RQ{number}: {question}")
        st.markdown(f"**Finding:** {finding}")
        viz_fn()
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**Decision Impact:** {decision_impact}")


# RQ1 -----------------------------------------------------------------------
_research_question_block(
    "1", "Which borrower characteristics appear associated with default?",
    "Interest rate, debt-to-income ratio, and loan grade are consistently among the strongest predictors "
    "across all three Phase 3 models (see Model Explainability page for the full ranking).",
    "Prioritize verifying interest-rate-adjacent and DTI data quality at application time, since these fields "
    "carry outsized weight in the risk assessment.",
    "Directly informs which fields underwriters should scrutinize most closely.",
    lambda: st.pyplot(eda_utils.plot_correlation_heatmap(cleaned_df, config.NUMERIC_FEATURES, title="Feature Correlation")[0]),
)

if not show_advanced_statistics:
    st.info(
        "This page is focused on the executive summary for the highest-priority finding (RQ1). "
        "Enable 'Show advanced statistics' in the left sidebar to view supplemental charts and deeper research-question detail.",
    )
else:
    advanced_tab, = st.tabs(["Advanced Statistics"])
    with advanced_tab:
        _research_question_block(
            "2", "Do LendingClub grades appear predictive of default?",
            "Default rate rises substantially from Grade A to Grade G in the observed data, confirming grade carries "
            "real predictive signal even alongside the other features in the model.",
            "Continue using grade as a core underwriting input; cross-check it against the model's predicted probability "
            "for cases where they disagree.",
            "Validates that the existing grading system remains a sound foundation for pricing.",
            lambda: st.pyplot(eda_utils.plot_default_rate_by_group(cleaned_df, "grade", title="Default Rate by Grade")[0]),
        )

        _research_question_block(
            "3", "Which variables are related to higher interest rates?",
            "Loan grade and DTI show the strongest relationship with the interest rate LendingClub assigns, consistent "
            "with a risk-based pricing policy.",
            "Continue aligning rate-setting with grade and DTI; monitor for drift if new risk factors emerge.",
            "Confirms current pricing policy is grounded in the same signals the model finds predictive.",
            lambda: st.pyplot(eda_utils.plot_numeric_distribution(cleaned_df, "int_rate", title="Interest Rate Distribution")),
        )

        _research_question_block(
            "4", "Does income relate to repayment success?",
            "Higher annual income is associated with lower predicted default risk across all three models, though the "
            "relationship strengthens further once DTI is also considered (see Model Explainability's dependence plots).",
            "Continue collecting verified income at application; consider requiring verification for borderline cases.",
            "Supports maintaining income verification as a standard underwriting step.",
            lambda: st.pyplot(eda_utils.plot_numeric_distribution(cleaned_df, "annual_inc", title="Annual Income Distribution")),
        )

        _research_question_block(
            "5", "Does DTI influence default?",
            "Debt-to-income ratio is one of the top-ranked predictors across every model and importance method evaluated.",
            "Consider a firmer DTI cutoff or a rate markup tier for high-DTI applicants.",
            "A concrete, actionable lever for updating underwriting policy.",
            lambda: st.pyplot(eda_utils.plot_numeric_distribution(cleaned_df, "dti", title="Debt-to-Income Distribution")),
        )

        _research_question_block(
            "6", "Does employment length matter?",
            "Employment length ranks lower in importance than income, DTI, and grade across all three models — it "
            "matters less than commonly assumed once those other factors are known.",
            "Lower-priority for manual verification relative to income and DTI; do not over-weight in manual overrides.",
            "Helps focus underwriter attention on higher-value verification steps.",
            lambda: st.pyplot(eda_utils.plot_numeric_distribution(cleaned_df, "emp_length_years", title="Employment Length Distribution")),
        )

        if segmentation_engine is not None:
            ml_comparison = segmentation_engine.compare_with_supervised_models()
            _research_question_block(
                "7", "Which borrower segments represent the highest lending risk, and can natural groups be observed before clustering?",
                "Borrower segmentation identifies distinct financial-profile groups whose predicted default probability "
                "(from the supervised model) and actual default rate agree directionally — see the Borrower Segmentation page.",
                "Use segment membership to set portfolio-level origination limits and marketing strategy, in addition to "
                "the per-borrower supervised prediction.",
                "Gives Lending Club both an individual risk score and a portfolio-level segmentation lens.",
                lambda: st.dataframe(
                    ml_comparison[["segment_name", "mean_predicted_probability", "average_default_rate", "risk_tier"]],
                    hide_index=True,
                ),
            )
        else:
            render_missing_artifact_notice("Borrower segmentation", "python -m src.train_models")
