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

from common import (
    apply_global_style, download_dataframe_button, download_report_buttons, get_explainability_engine,
    get_global_explanation, get_learning_curve_figure, get_risk_engine, load_phase3_reports,
    load_splits_cached, render_executive_summary_box, render_missing_artifact_notice,
    render_page_header, render_section_header,
)
from src import config, model_utils

apply_global_style()
render_page_header(
    "Regression Model",
    "Integrated regression diagnostics and SHAP-based explanation for the production scoring model.",
)

model_key = st.session_state.get("selected_model_key", config.PRODUCTION_MODEL_KEY)
engine = get_explainability_engine(model_key)

if engine is None:
    render_missing_artifact_notice("Model explainability artifacts", "python -m src.train_models")
    st.stop()

_, _, X_test, _, _, y_test = load_splits_cached()

# ---------------------------------------------------------------------------
# Regression Model diagnostics
# ---------------------------------------------------------------------------
show_advanced_statistics = st.session_state.get("show_advanced_statistics", False)

if show_advanced_statistics:
    render_section_header("Regression Model Overview")

    reports = load_phase3_reports()
    if not reports:
        render_missing_artifact_notice("Model evaluation reports", "python -m src.train_models")
        st.stop()

    comparison_table = reports["comparison_table"]
    st.dataframe(comparison_table, hide_index=True)
    download_dataframe_button(comparison_table, "⬇ Download Comparison Table (CSV)", "model_comparison_table.csv")

    best_model_row = comparison_table.iloc[0]
    st.success(
        f"**Executive Recommendation:** {best_model_row['model']} ranks #1 by test ROC-AUC "
        f"({best_model_row['roc_auc']:.3f}) and is the current production scoring model.",
        icon="🏆",
    )

    render_section_header("Regression Model Diagnostics")
    model_keys = ["logistic_regression", "random_forest", "xgboost"]
    selected = st.multiselect(
        "Compare models:", model_keys, default=model_keys,
        format_func=lambda k: model_utils.MODEL_DISPLAY_NAMES[k],
    )

    if not selected:
        st.info("Select at least one model to view diagnostics.", icon="ℹ️")
        st.stop()

    proba_by_model = reports["probability_predictions"]

    tab_roc, tab_pr, tab_cm, tab_cal, tab_learning = st.tabs(
        ["ROC Curves", "Precision-Recall Curves", "Confusion Matrices", "Calibration Curves", "Learning Curves"]
    )

    with tab_roc:
        cols = st.columns(len(selected))
        for col, key in zip(cols, selected):
            with col:
                fig = model_utils.plot_roc_curve_chart(y_test, proba_by_model[key], title=model_utils.MODEL_DISPLAY_NAMES[key])
                st.pyplot(fig)

    with tab_pr:
        cols = st.columns(len(selected))
        for col, key in zip(cols, selected):
            with col:
                fig = model_utils.plot_pr_curve_chart(y_test, proba_by_model[key], title=model_utils.MODEL_DISPLAY_NAMES[key])
                st.pyplot(fig)

    with tab_cm:
        cols = st.columns(len(selected))
        for col, key in zip(cols, selected):
            with col:
                y_pred = (proba_by_model[key] >= 0.5).astype(int)
                fig = model_utils.plot_confusion_matrix_chart(y_test, y_pred, title=model_utils.MODEL_DISPLAY_NAMES[key])
                st.pyplot(fig)

    with tab_cal:
        cols = st.columns(len(selected))
        for col, key in zip(cols, selected):
            with col:
                fig = model_utils.plot_calibration_curve_chart(y_test, proba_by_model[key], title=model_utils.MODEL_DISPLAY_NAMES[key])
                st.pyplot(fig)

    with tab_learning:
        st.caption("Learning curves are cached per model (they require refitting across several training-set sizes).")
        learning_key = st.selectbox("Model:", selected, format_func=lambda k: model_utils.MODEL_DISPLAY_NAMES[k], key="learning_curve_model")
        learning_engine = get_risk_engine(learning_key)
        if learning_engine is not None:
            X_train, _, _, y_train, _, _ = load_splits_cached()
            fig = get_learning_curve_figure(learning_engine.pipeline, X_train, y_train, learning_key)
            st.pyplot(fig)

# ---------------------------------------------------------------------------
# Global SHAP feature importance
# ---------------------------------------------------------------------------
render_section_header("Global Feature Importance")

global_explanation = get_global_explanation(engine, model_key, len(X_test))

st.dataframe(
    global_explanation.importance_table[["feature_label", "mean_abs_shap", "permutation_importance", "research_question"]],
    hide_index=True,
)
render_executive_summary_box(global_explanation.business_summary)

# ---------------------------------------------------------------------------
# SHAP visualizations
# ---------------------------------------------------------------------------
if show_advanced_statistics:
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

if show_advanced_statistics:
    render_section_header("Exportable Report")
    download_report_buttons(engine.export_global_explanation_report(X_test), "global_model_explanation", "global_explain_report")
