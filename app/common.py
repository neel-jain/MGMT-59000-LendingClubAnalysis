"""
common.py
==========
Phase 5 shared UI utilities for the LendingClub Risk Intelligence
Streamlit application.

This module intentionally contains NO machine-learning or business
logic -- every prediction, explanation, and segment assignment comes
from `RiskScoringEngine`, `ExplainabilityEngine`, and
`SegmentationEngine` (Phases 4A/4B). This module only:
    - loads/caches those engines and shared data so pages don't each
      re-implement caching or re-fit anything,
    - renders shared UI chrome (KPI cards, page headers, download
      buttons, the sidebar's filter/model-selection/about/download
      controls),
    - applies the app's visual style.

Every `app_pages/*.py` page imports from here rather than duplicating
any of this logic, per the project's "avoid duplicated code" standard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Make `from src import ...` resolve regardless of the working directory
# `streamlit run` is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src import config, interpretation_utils, labels, model_utils, utils
from src.configurable_thresholds import RiskThresholdConfig, load_threshold_config
from src.explainability import ExplainabilityEngine
from src.risk_scoring import RiskScoringEngine
from src.segmentation_engine import SegmentationEngine

logger = utils.get_logger(__name__)

MODEL_KEYS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]
MODEL_LABELS: Dict[str, str] = {k: model_utils.MODEL_DISPLAY_NAMES[k] for k in MODEL_KEYS}


# ---------------------------------------------------------------------------
# Cached data / engine loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading borrower dataset...")
def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the winsorized borrower dataset used by the Streamlit app.

    Returns
    -------
    pd.DataFrame
        The winsorized dataset, or an empty frame if it has not yet been generated.
    """
    try:
        return utils.load_dataframe(config.WINSORIZED_DATA_PATH)
    except FileNotFoundError:
        logger.warning("Winsorized dataset not found at %s.", config.WINSORIZED_DATA_PATH)
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading train/validation/test splits...")
def load_splits_cached() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Cached wrapper around `utils.load_splits()` (avoids re-reading six CSVs on every rerun)."""
    return utils.load_splits()


@st.cache_resource(show_spinner="Loading risk scoring engine...")
def get_risk_engine(model_key: str) -> Optional[RiskScoringEngine]:
    """
    Load (and cache, per model_key) a `RiskScoringEngine`. Returns None
    if the requested model hasn't been trained yet (Phase 3 not run),
    so pages can show a setup message instead of crashing.
    """
    try:
        return RiskScoringEngine(model_key=model_key)
    except FileNotFoundError as exc:
        logger.warning("Could not load RiskScoringEngine for '%s': %s", model_key, exc)
        return None


@st.cache_resource(show_spinner="Loading explainability engine (this computes a SHAP background sample)...")
def get_explainability_engine(model_key: str) -> Optional[ExplainabilityEngine]:
    """Load (and cache, per model_key) an `ExplainabilityEngine`. Returns None if unavailable."""
    try:
        return ExplainabilityEngine(model_key=model_key)
    except FileNotFoundError as exc:
        logger.warning("Could not load ExplainabilityEngine for '%s': %s", model_key, exc)
        return None


@st.cache_resource(show_spinner="Fitting the borrower segmentation model...")
def get_segmentation_engine() -> Optional[SegmentationEngine]:
    """
    Load and fit (once, cached for the app's lifetime) a
    `SegmentationEngine` on the Phase 1 training split. Returns None if
    the training split isn't available yet.
    """
    try:
        X_train, _, _, y_train, _, _ = load_splits_cached()
    except FileNotFoundError as exc:
        logger.warning("Could not load splits for SegmentationEngine: %s", exc)
        return None
    engine = SegmentationEngine()
    engine.fit(X_train, default_flags=y_train)
    return engine


# ---------------------------------------------------------------------------
# Cached wrappers for expensive per-page computations
# ---------------------------------------------------------------------------
# These wrap engine calls that are otherwise recomputed from scratch on
# EVERY Streamlit rerun (e.g. any widget interaction on the same page)
# even though their result only depends on which model/engine is
# selected -- t-SNE/UMAP projection and cross-validated learning curves
# are both meaningfully expensive (multi-second) operations. The leading
# underscore on the engine/pipeline parameter tells Streamlit's
# `st.cache_data` not to try to hash that (unhashable) object; the cache
# key is instead the remaining plain arguments (e.g. `model_key`).


@st.cache_data(show_spinner="Computing t-SNE projection (this can take a few seconds)...")
def get_cluster_visualization(_engine: SegmentationEngine, method: str, model_key: str):
    """
    Cached wrapper around `SegmentationEngine.visualize_clusters`. `model_key`
    is included only so the cache key changes if the underlying engine
    instance is ever swapped for a different model's segmentation (the
    segmentation model itself is not model-specific, but this keeps the
    cache correctly scoped per engine instance without hashing the engine).
    """
    return _engine.visualize_clusters(method=method)


@st.cache_data(show_spinner="Computing learning curve (this refits the model across several training sizes)...")
def get_learning_curve_figure(_pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_key: str):
    """Cached wrapper around `model_utils.plot_learning_curve_chart`, keyed on `model_key`."""
    return model_utils.plot_learning_curve_chart(
        _pipeline, X_train, y_train, title=MODEL_LABELS[model_key],
    )


@st.cache_data(show_spinner="Computing global SHAP explanation...")
def get_global_explanation(_engine: ExplainabilityEngine, model_key: str, sample_size: int):
    """Cached wrapper around `ExplainabilityEngine.explain_global_model`, keyed on `model_key`."""
    _, _, X_test, _, _, _ = load_splits_cached()
    sample = X_test if len(X_test) <= sample_size else X_test.sample(n=sample_size, random_state=42)
    return _engine.explain_global_model(sample)


@st.cache_data(show_spinner="Loading model evaluation reports...")
def load_phase3_reports() -> Dict[str, object]:
    """
    Load every Phase 3 evaluation artifact needed by the Model
    Comparison page. Returns an empty dict (with a logged warning) if
    Phase 3 hasn't been run yet.
    """
    try:
        return {
            "evaluation_metrics": utils.load_object(config.EVALUATION_METRICS_PATH),
            "cv_results": utils.load_object(config.CV_RESULTS_PATH),
            "feature_importance": utils.load_object(config.FEATURE_IMPORTANCE_PATH),
            "probability_predictions": utils.load_object(config.PROBABILITY_PREDICTIONS_PATH),
            "threshold_analysis": utils.load_object(config.THRESHOLD_ANALYSIS_PATH),
            "comparison_table": pd.read_csv(config.MODEL_COMPARISON_TABLE_PATH),
        }
    except FileNotFoundError as exc:
        logger.warning("Phase 3 reports not fully available: %s", exc)
        return {}


@st.cache_resource(show_spinner=False)
def get_threshold_config() -> RiskThresholdConfig:
    """Load the (JSON-backed, hot-editable) risk threshold configuration."""
    return load_threshold_config()


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
/* Executive KPI card */
div[data-testid="stMetric"] {
    background-color: #F4F6F8;
    border: 1px solid #E1E5E9;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
}
div[data-testid="stMetric"] label {
    font-weight: 600;
    color: #4A5A68;
}
/* Page headers */
.page-header {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1A2B3C;
    margin-bottom: 0.1rem;
}
.page-subtitle {
    font-size: 1.02rem;
    color: #5B6B78;
    margin-bottom: 1.1rem;
}
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1A2B3C;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
    border-bottom: 2px solid #2E86AB;
    padding-bottom: 0.25rem;
}
.exec-summary-box {
    background-color: #F0F6FA;
    border-left: 4px solid #2E86AB;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 0.6rem 0 1rem 0;
    font-size: 1.0rem;
    line-height: 1.5;
}
</style>
"""


def apply_global_style() -> None:
    """Inject the app's shared CSS (KPI cards, headers, executive-summary callout box). Call once per page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page title + subtitle across every page."""
    st.markdown(f'<div class="page-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_section_header(title: str) -> None:
    """Render a consistent section divider within a page."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_executive_summary_box(text: str) -> None:
    """Render a visually distinct callout box for executive-summary text returned by an engine."""
    st.markdown(f'<div class="exec-summary-box">{text}</div>', unsafe_allow_html=True)


def render_missing_artifact_notice(what: str, phase_hint: str) -> None:
    """
    Consistent, friendly notice shown when a required artifact (model,
    dataset, report) hasn't been generated yet, instead of letting a
    page crash with a raw exception.
    """
    st.warning(
        f"**{what} is not available yet.** Run `{phase_hint}` from the project "
        f"root to generate it, then reload this page.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_dataframe_button(df: pd.DataFrame, label: str, filename: str, key: Optional[str] = None) -> None:
    """Render a CSV download button for any DataFrame."""
    st.download_button(
        label=label, data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=key,
    )


def download_report_buttons(report: "interpretation_utils.ExportableReport", filename_stem: str, key_prefix: str) -> None:
    """
    Render Markdown + JSON download buttons for any
    `interpretation_utils.ExportableReport` (used across the Prediction,
    Segmentation, and Explainability pages for their exportable reports).
    """
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download Report (Markdown)", data=report.to_markdown().encode("utf-8"),
            file_name=f"{filename_stem}.md", mime="text/markdown", key=f"{key_prefix}_md",
        )
    with col2:
        st.download_button(
            "⬇ Download Report (JSON)", data=report.to_json().encode("utf-8"),
            file_name=f"{filename_stem}.json", mime="application/json", key=f"{key_prefix}_json",
        )


def download_figure_button(fig, label: str, filename: str, key: Optional[str] = None) -> None:
    """Render a PNG download button for any matplotlib Figure."""
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    st.download_button(label=label, data=buffer.getvalue(), file_name=filename, mime="image/png", key=key)


# ---------------------------------------------------------------------------
# Sidebar controls (shared across all pages via app.py)
# ---------------------------------------------------------------------------


def render_model_selector() -> str:
    """
    Render the sidebar's model-selection control, backed by
    `st.session_state["selected_model_key"]` so every page reads the
    same choice. Defaults to `config.PRODUCTION_MODEL_KEY`.

    Returns
    -------
    str
        The currently selected model key.
    """
    if "selected_model_key" not in st.session_state:
        st.session_state["selected_model_key"] = config.PRODUCTION_MODEL_KEY

    label = st.sidebar.selectbox(
        "Scoring Model", options=MODEL_KEYS, format_func=lambda k: MODEL_LABELS[k],
        index=MODEL_KEYS.index(st.session_state["selected_model_key"]),
        help="Choose which Phase 3 model powers predictions, explanations, and comparisons on this page.",
        key="model_selector_widget",
    )
    st.session_state["selected_model_key"] = label
    return label


def render_borrower_filters(df: pd.DataFrame) -> Dict[str, object]:
    """
    Render the sidebar's borrower-filter controls (grade, purpose,
    income, employment length, home ownership, interest rate, DTI, loan
    amount) and return the selections as a plain dict, backed by
    `st.session_state["borrower_filters"]` so the choices persist as the
    user moves between pages.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset, used only to set slider/selectbox bounds
        from the actual data range.

    Returns
    -------
    dict
        Filter selections, consumed by `apply_borrower_filters`.
    """
    if df.empty:
        return {}

    with st.sidebar.expander("🔎 Borrower Filters", expanded=False):
        grades = sorted(df["grade"].dropna().unique().tolist())
        purposes = sorted(df["purpose"].dropna().unique().tolist())
        ownerships = sorted(df["home_ownership"].dropna().unique().tolist())

        selected_grades = st.multiselect("Loan Grade", grades, default=grades, key="filter_grades")
        # format_func only changes the displayed text; the multiselect still
        # returns the raw category values that apply_borrower_filters needs.
        selected_purposes = st.multiselect(
            "Loan Purpose", purposes, default=purposes,
            format_func=lambda v: labels.category_label("purpose", v), key="filter_purposes",
        )
        selected_ownership = st.multiselect(
            "Home Ownership", ownerships, default=ownerships,
            format_func=lambda v: labels.category_label("home_ownership", v), key="filter_ownership",
        )

        income_min, income_max = float(df["annual_inc"].min()), float(df["annual_inc"].max())
        income_range = st.slider("Annual Income ($)", income_min, income_max, (income_min, income_max), key="filter_income")

        emp_min, emp_max = float(df["emp_length_years"].min()), float(df["emp_length_years"].max())
        emp_range = st.slider("Employment Length (years)", emp_min, emp_max, (emp_min, emp_max), key="filter_emp")

        rate_min, rate_max = float(df["int_rate"].min()), float(df["int_rate"].max())
        rate_range = st.slider("Interest Rate (%)", rate_min, rate_max, (rate_min, rate_max), key="filter_rate")

        dti_min, dti_max = float(df["dti"].min()), float(df["dti"].max())
        dti_range = st.slider("Debt-to-Income Ratio (%)", dti_min, dti_max, (dti_min, dti_max), key="filter_dti")

        loan_min, loan_max = float(df["loan_amnt"].min()), float(df["loan_amnt"].max())
        loan_range = st.slider("Loan Amount ($)", loan_min, loan_max, (loan_min, loan_max), key="filter_loan")

    filters = {
        "grade": selected_grades, "purpose": selected_purposes, "home_ownership": selected_ownership,
        "annual_inc": income_range, "emp_length_years": emp_range, "int_rate": rate_range,
        "dti": dti_range, "loan_amnt": loan_range,
    }
    st.session_state["borrower_filters"] = filters
    return filters


def apply_borrower_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    """
    Apply the filter dict from `render_borrower_filters` to a DataFrame.
    Pure data-filtering (no modeling logic) -- safe to call from any
    page that wants a filtered view of the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
    filters : dict

    Returns
    -------
    pd.DataFrame
        Filtered copy of `df`. Returns `df` unchanged if `filters` is empty.
    """
    if not filters or df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if filters.get("grade"):
        mask &= df["grade"].isin(filters["grade"])
    if filters.get("purpose"):
        mask &= df["purpose"].isin(filters["purpose"])
    if filters.get("home_ownership"):
        mask &= df["home_ownership"].isin(filters["home_ownership"])
    for col in ("annual_inc", "emp_length_years", "int_rate", "dti", "loan_amnt"):
        lo, hi = filters.get(col, (None, None))
        if lo is not None:
            mask &= df[col].between(lo, hi)
    return df[mask]


def render_advanced_statistics_toggle() -> bool:
    """
    Render the sidebar toggle for advanced statistics content.

    Returns
    -------
    bool
        True when advanced statistics should be shown.
    """
    return st.sidebar.checkbox(
        "Show advanced statistics", value=False,
        help="Reveal deeper analytics and supplemental charts for advanced users.",
        key="show_advanced_statistics",
    )


def advanced_statistics_enabled() -> bool:
    """Return whether the user has enabled advanced statistics in the sidebar."""
    return st.session_state.get("show_advanced_statistics", False)


def render_theme_options() -> str:
    """
    Render the sidebar's (lightweight) display-density option. Full
    color theming is configured via `.streamlit/config.toml` (Streamlit
    does not support hot-swapping its color theme at runtime); this
    control instead toggles chart/table density for different screen
    sizes, backed by `st.session_state["display_density"]`.

    Returns
    -------
    str
        "Comfortable" or "Compact".
    """
    density = st.sidebar.radio(
        "Display Density", ["Comfortable", "Compact"], horizontal=True,
        help="Adjust chart and table sizing. The color theme itself is set in .streamlit/config.toml.",
        key="display_density",
    )
    return density


def render_download_options() -> None:
    """Render the sidebar's global download options (winsorized dataset, model comparison table)."""
    with st.sidebar.expander("⬇ Download Options", expanded=False):
        winsorized = load_cleaned_dataset()
        if not winsorized.empty:
            download_dataframe_button(
                winsorized,
                "Winsorized Borrower Dataset (CSV)",
                config.WINSORIZED_DATA_FILENAME,
                key="dl_winsorized_dataset",
            )
        reports = load_phase3_reports()
        if reports:
            download_dataframe_button(
                reports["comparison_table"], "Model Comparison Table (CSV)", "model_comparison_table.csv", key="dl_comparison_table",
            )
        if not winsorized.empty or reports:
            st.caption("More specific exports (borrower reports, segment summaries) are available on their respective pages.")
        else:
            st.caption("Run the project pipeline to generate downloadable data.")


def render_about_sidebar() -> None:
    """Render the sidebar's persistent 'About Project' summary (full detail lives on the About Project page)."""
    with st.sidebar.expander("ℹ️ About This Project", expanded=False):
        st.markdown(
            "**LendingClub Loan Default Risk — Indiana Borrowers**\n\n"
            "MGMT 59000 capstone (Purdue University). Predicts loan default risk "
            "and supports lending decisions using supervised ML, SHAP-based "
            "explainability, and borrower segmentation.\n\n"
            "See the **About Project** page for the full methodology, research "
            "questions, and technology stack."
        )


def render_global_sidebar_controls(cleaned_df: pd.DataFrame) -> None:
    """
    Render every sidebar control the Phase 5 brief calls for (Borrower
    Filters, Advanced Statistics toggle, Download Options, About Project)
    in one call from `app.py`, so it appears consistently above the
    page-specific content on every page. Navigation itself is rendered
    separately by `st.navigation` in `app.py`.
    """
    st.sidebar.markdown("---")
    render_advanced_statistics_toggle()
    render_borrower_filters(cleaned_df)
    render_download_options()
    render_about_sidebar()
