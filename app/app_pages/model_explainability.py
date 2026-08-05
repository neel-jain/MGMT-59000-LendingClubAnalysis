"""
model_explainability.py
=========================
Phase 5, Page 7: Model Explainability.

Global model-level explanations, entirely sourced from
`ExplainabilityEngine` (Phase 4A). This page renders SHAP summary,
dependence, and decision plots plus the executive/business
interpretation text the engine already generates -- no SHAP computation
happens in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.common import (
    apply_global_style, download_report_buttons, get_explainability_engine, get_global_explanation,
    load_splits_cached, render_executive_summary_box, render_missing_artifact_notice, render_page_header,
    render_section_header,
)
from src import config

apply_global_style()
render_page_header(
    "Model Explainability",
    "Global SHAP-based explanation of what drives the production model's predictions.",
)

model_key = st.session_state.get("selected_model_key", config.PRODUCTION_MODEL_KEY)
engine = get_explainability_engine(model_key)

if engine is None:
    render_missing_artifact_notice("Model explainability artifacts", "python -m src.train_models")
    st.stop()

_, _, X_test, _, _, _ = load_splits_cached()

# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------
render_section_header("Global Feature Importance")

global_explanation = get_global_explanation(engine, model_key, len(X_test))

st.dataframe(
    global_explanation.importance_table[["feature_label", "mean_abs_shap", "permutation_importance", "research_question"]],
    hide_index=True,
)
render_executive_summary_box(global_explanation.business_summary)

# ---------------------------------------------------------------------------
# SHAP summary / dependence / decision plots
# ---------------------------------------------------------------------------
render_section_header("SHAP Visualizations")

tabs = st.tabs(["Summary (Beeswarm)", "Summary (Bar)", "Dependence Plot", "Decision Plot", "Waterfall (Sample Borrower)"])

with tabs[0]:
    fig = engine.generate_shap_summary(X_test, plot_type="beeswarm")
    st.pyplot(fig)
with tabs[1]:
    fig = engine.generate_shap_summary(X_test, plot_type="bar")
    st.pyplot(fig)
with tabs[2]:
    feature_options = [f for f in config.NUMERIC_FEATURES + config.ORDINAL_CATEGORICAL_FEATURES]
    dep_feature = st.selectbox("Feature:", feature_options, index=feature_options.index("dti") if "dti" in feature_options else 0)
    fig = engine.generate_dependence_plot(dep_feature, X_test)
    st.pyplot(fig)
with tabs[3]:
    fig = engine.generate_decision_plot(X_test, n_samples=25)
    st.pyplot(fig)
with tabs[4]:
    st.caption("Illustrative single-borrower waterfall — visit Borrower Risk Prediction for a live borrower's own explanation.")
    fig = engine.generate_waterfall_plot(X_test.iloc[[0]])
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Business interpretation
# ---------------------------------------------------------------------------
render_section_header("Business Interpretation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Most influential variables:**")
    for f in global_explanation.top_features:
        st.markdown(f"- {f}")
with col2:
    st.markdown("**Least influential variables:**")
    for f in global_explanation.least_influential_features:
        st.markdown(f"- {f}")

render_section_header("Exportable Report")
download_report_buttons(engine.export_global_explanation_report(X_test), "global_model_explanation", "global_explain_report")
