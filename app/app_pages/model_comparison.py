"""
model_comparison.py
=====================
Phase 5, Page 3: Model Comparison.

Displays Phase 3's evaluation artifacts (ROC/PR curves, confusion
matrix, calibration curve, learning curve, performance table, ranking)
side-by-side for Logistic Regression, Random Forest, and XGBoost.
Re-renders using `model_utils.py`'s existing plotting functions against
the saved probability predictions -- no metrics are recomputed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.common import (
    apply_global_style, download_dataframe_button, get_learning_curve_figure, get_risk_engine,
    load_phase3_reports, load_splits_cached, render_missing_artifact_notice, render_page_header,
    render_section_header,
)
from src import model_utils

apply_global_style()
render_page_header(
    "Model Comparison",
    "Side-by-side evaluation of the three Phase 3 supervised models.",
)

reports = load_phase3_reports()
if not reports:
    render_missing_artifact_notice("Model evaluation reports", "python -m src.train_models")
    st.stop()

_, _, _, _, _, y_test = load_splits_cached()

# ---------------------------------------------------------------------------
# Performance table + ranking
# ---------------------------------------------------------------------------
render_section_header("Performance Table & Ranking")

comparison_table = reports["comparison_table"]
st.dataframe(comparison_table, hide_index=True)
download_dataframe_button(comparison_table, "⬇ Download Comparison Table (CSV)", "model_comparison_table.csv")

best_model_row = comparison_table.iloc[0]
st.success(
    f"**Executive Recommendation:** {best_model_row['model']} ranks #1 by test ROC-AUC "
    f"({best_model_row['roc_auc']:.3f}) and is the current production scoring model.",
    icon="🏆",
)

# ---------------------------------------------------------------------------
# Model selector for detailed diagnostics
# ---------------------------------------------------------------------------
render_section_header("Detailed Diagnostics")

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
    engine = get_risk_engine(learning_key)
    if engine is not None:
        X_train, _, _, y_train, _, _ = load_splits_cached()
        fig = get_learning_curve_figure(engine.pipeline, X_train, y_train, learning_key)
        st.pyplot(fig)
