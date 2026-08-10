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

from common import (
    apply_global_style, download_dataframe_button, get_learning_curve_figure, get_risk_engine,
    load_phase3_reports, load_splits_cached, render_missing_artifact_notice, render_page_header,
    render_section_header,
)
from src import model_utils

apply_global_style()
render_page_header(
    "Model Comparison",
    "Archived model comparison page. Detailed regression diagnostics have been moved to the Regression Model page.",
)

show_advanced_statistics = st.session_state.get("show_advanced_statistics", False)
if not show_advanced_statistics:
    st.info(
        "This page no longer shows comparison diagnostics. Open the Regression Model page for the integrated analysis.",
        icon="ℹ️",
    )
else:
    st.info(
        "Advanced statistics are enabled, but model comparison diagnostics now live on the Regression Model tab.",
        icon="ℹ️",
    )
