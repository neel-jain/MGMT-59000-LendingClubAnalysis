"""
app.py
=======
Phase 5 entry point for the LendingClub Risk Intelligence Streamlit
application.

Run from the project root with:
    streamlit run app/app.py

This file only wires up navigation and the shared sidebar -- it
contains NO machine-learning or business logic itself, consistent with
the project rule that the Streamlit layer only orchestrates the
`RiskScoringEngine`, `ExplainabilityEngine`, and `SegmentationEngine`
built in Phases 4A/4B. Each page under `app_pages/` is a standalone
script that Streamlit's `st.navigation`/`st.Page` API runs in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    apply_global_style, get_production_model_key, load_cleaned_dataset, render_global_sidebar_controls,
)
from src import utils  # noqa: E402

logger = utils.get_logger("app")

# Streamlit re-executes this script on every user interaction, but
# "application startup" should only be logged once per session (not on
# every rerun) -- guarded via session_state, per Phase 6's structured-
# logging requirement.
if "_app_session_started" not in st.session_state:
    logger.info(
        "Streamlit session started. Production model=%s, project root=%s",
        get_production_model_key(), PROJECT_ROOT,
    )
    st.session_state["_app_session_started"] = True

st.set_page_config(
    page_title="LendingClub Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_style()

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

pages = {
    "Overview": [
        st.Page(str(PAGES_DIR / "executive_dashboard.py"), title="Executive Dashboard", icon="📊", default=True),
    ],
    "Analysis": [
        st.Page(str(PAGES_DIR / "exploratory_analysis.py"), title="Exploratory Analysis", icon="🔍"),
        st.Page(str(PAGES_DIR / "business_insights.py"), title="Business Insights", icon="💡"),
        st.Page(str(PAGES_DIR / "model_explainability.py"), title="Model Explainability", icon="🧠"),
    ],
    "Decision Tools": [
        st.Page(str(PAGES_DIR / "borrower_risk_prediction.py"), title="Borrower Risk Prediction", icon="🎯"),
        st.Page(str(PAGES_DIR / "borrower_segmentation.py"), title="Borrower Segmentation", icon="👥"),
    ],
    "Project Info": [
        st.Page(str(PAGES_DIR / "about_project.py"), title="About Project", icon="ℹ️"),
    ],
}

navigation = st.navigation(pages)

st.sidebar.markdown("## 💳 LendingClub Risk Intelligence")
st.sidebar.caption("Indiana Borrower Default Risk Platform")

render_global_sidebar_controls(load_cleaned_dataset())

navigation.run()
