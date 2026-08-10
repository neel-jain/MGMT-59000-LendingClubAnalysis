"""
borrower_segmentation.py
==========================
Phase 5, Page 5: Borrower Segmentation.

All clustering, profiling, and recommendation logic comes from
`SegmentationEngine` (Phase 4B). This page only offers a way to look up
a segment (either for a manually-entered borrower profile or by
selecting a segment directly) and displays whatever the engine returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from common import (
    apply_global_style, download_dataframe_button, download_report_buttons, get_cluster_visualization,
    get_segmentation_engine, render_missing_artifact_notice, render_page_header, render_section_header,
)
from src import cluster_visualization as cv
from src import utils

logger = utils.get_logger(__name__)

apply_global_style()
render_page_header(
    "Borrower Segmentation",
    "Natural borrower groups identified via unsupervised clustering, complementing the supervised risk models.",
)

engine = get_segmentation_engine()
if engine is None:
    render_missing_artifact_notice("The segmentation model", "python -m src.train_models")
    st.stop()

# ---------------------------------------------------------------------------
# Segment overview
# ---------------------------------------------------------------------------
render_section_header("Segment Overview")

comparison = engine.compare_segments()
st.dataframe(comparison, hide_index=True)
download_dataframe_button(comparison, "⬇ Download Segment Comparison (CSV)", "segment_comparison.csv")

# ---------------------------------------------------------------------------
# Select a segment for detail
# ---------------------------------------------------------------------------
render_section_header("Segment Detail")

segment_names = engine.fit_result.segment_names
selected_name = st.selectbox("Choose a segment to inspect:", list(segment_names.values()))
selected_cluster_id = next(cid for cid, name in segment_names.items() if name == selected_name)

st.markdown(f"**{selected_name}**")
st.write(engine.describe_segment(selected_cluster_id))

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Cluster Characteristics**")
    st.dataframe(engine.generate_cluster_profile().loc[[selected_cluster_id]])
with col2:
    st.markdown("**Business Recommendations**")
    rec = engine.recommend_business_actions(selected_cluster_id)
    st.markdown(f"- **Risk level:** {rec.primary_risk_level}")
    st.markdown(f"- **Lending recommendation:** {rec.lending_recommendation}")
    st.markdown(f"- **Interest rate strategy:** {rec.interest_rate_strategy}")
    st.markdown(f"- **Underwriting strategy:** {rec.underwriting_strategy}")
    st.markdown(f"- **Manual review:** {rec.manual_review_requirement}")
    st.markdown(f"- **Marketing strategy:** {rec.marketing_strategy}")
    st.markdown(f"- **Portfolio notes:** {rec.portfolio_management_notes}")

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
show_advanced_statistics = st.session_state.get("show_advanced_statistics", False)
if show_advanced_statistics:
    render_section_header("Cluster Visualizations")
    viz_tabs = st.tabs(["PCA", "t-SNE", "Radar Chart", "Heatmap"])
    segmentation_cache_key = f"{engine.algorithm}_{engine.n_clusters}"
    try:
        with viz_tabs[0]:
            fig = get_cluster_visualization(engine, "pca", segmentation_cache_key)
            st.pyplot(fig)
        with viz_tabs[1]:
            fig = get_cluster_visualization(engine, "tsne", segmentation_cache_key)
            st.pyplot(fig)
        with viz_tabs[2]:
            profile_df = engine._X_train_raw.copy()
            profile_df["cluster"] = engine.fit_result.labels
            features = ["annual_inc", "dti", "loan_amnt", "int_rate", "emp_length_years", "revol_util"]
            fig = cv.plot_radar_chart(profile_df, features, segment_names=segment_names)
            st.pyplot(fig)
        with viz_tabs[3]:
            profile_df = engine._X_train_raw.copy()
            profile_df["cluster"] = engine.fit_result.labels
            features = ["annual_inc", "dti", "loan_amnt", "int_rate", "emp_length_years", "revol_util"]
            fig = cv.plot_cluster_heatmap(profile_df, features, segment_names=segment_names)
            st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cluster visualization failed: %s", exc, exc_info=True)
        st.error("One or more cluster visualizations could not be rendered. The segment data above remains valid.", icon="🚫")

# ---------------------------------------------------------------------------
# Relationship to supervised models + portfolio recommendations
# ---------------------------------------------------------------------------
render_section_header("Relationship to the Supervised Models")

try:
    ml_comparison = engine.compare_with_supervised_models()
    st.dataframe(
        ml_comparison[["segment_name", "n_borrowers", "mean_predicted_probability", "average_default_rate", "risk_tier"]],
        hide_index=True,
    )
    st.caption(
        "Segments near agreement between `mean_predicted_probability` (Phase 3 model) and "
        "`average_default_rate` (actual outcomes) validate that both analytical lenses agree on relative risk."
    )
except Exception as exc:  # noqa: BLE001
    logger.error("compare_with_supervised_models failed: %s", exc, exc_info=True)
    st.warning("Could not cross-reference segments against the supervised model right now.", icon="⚠️")

render_section_header("Exportable Segmentation Report")
download_report_buttons(engine.export_segment_summary(), "borrower_segmentation_report", "segment_report")
