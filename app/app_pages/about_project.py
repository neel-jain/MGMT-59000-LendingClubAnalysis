"""
about_project.py
==================
Phase 5, Page 8: About Project.

Static project documentation -- business problem, research questions,
methodology, technology stack, workflow, limitations, and future
improvements. Purely descriptive; no engine calls needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from common import apply_global_style, render_page_header, render_section_header

apply_global_style()
render_page_header("About This Project", "LendingClub Loan Default Risk — Indiana Borrowers")

render_section_header("Business Problem")
st.markdown(
    "LendingClub, a peer-to-peer lending platform, needs to accurately assess the default risk of "
    "prospective borrowers to make sound lending decisions, price loans appropriately, and manage "
    "portfolio-level risk. This project builds a decision-support platform focused specifically on "
    "Indiana borrowers, combining predictive modeling, explainability, and segmentation into one "
    "coherent tool for lending executives, underwriters, and portfolio managers."
)

render_section_header("Research Questions")
st.markdown(
  "Which borrower characteristics appear associated with default?"
)

render_section_header("PDID Framework")
st.markdown(
    """
This project follows a **Problem → Data → Insight → Decision (PDID)** framework at every phase:
- **Problem:** a specific lending or risk-management question is stated before any analysis begins.
- **Data:** the Indiana LendingClub extract is validated, cleaned, and transformed to answer it.
- **Insight:** statistical analysis, supervised models, SHAP explanations, and clustering surface
  the underlying patterns.
- **Decision:** every insight is translated into a concrete lending, pricing, underwriting, or
  portfolio-management recommendation (see the Business Insights page).
"""
)

render_section_header("Project Methodology")
st.markdown(
    """
| Phase | Scope |
|---|---|
| 1 | Data architecture: ingestion, validation, cleaning, leakage-safe train/val/test split, preprocessing pipeline |
| 2 | Exploratory data analysis and statistical testing |
| 3 | Supervised modeling: Logistic Regression, Random Forest, XGBoost — tuned, evaluated, and compared |
| 4A | Explainable AI (SHAP) and a configurable risk-scoring engine |
| 4B | Unsupervised borrower segmentation, complementing the supervised models |
| 5 | This Streamlit dashboard |
| 6 | Integration testing, performance optimization, and deployment |
| 7 | Final code review, documentation, and presentation assets |
"""
)

render_section_header("Machine Learning Workflow")
st.markdown(
    """
1. Raw data ingested and validated against an expected schema.
2. Cleaned: duplicates removed, percentages parsed, employment length parsed, binary target
   constructed, leakage-prone columns dropped.
3. Split into training/validation/test sets (stratified, leakage-safe).
4. Preprocessing (median imputation, standardization, one-hot/ordinal encoding) embedded inside a
   scikit-learn `Pipeline` alongside each classifier, so every fold in cross-validation refits its
   own preprocessing statistics.
5. Hyperparameters tuned via `GridSearchCV` (Logistic Regression) and `RandomizedSearchCV` (Random
   Forest, XGBoost) with Stratified 5-fold cross-validation, optimizing ROC-AUC.
6. Models evaluated on a held-out test set across a full metric suite (accuracy, precision, recall,
   specificity, F1, ROC-AUC, balanced accuracy, MCC, log loss, Brier score, calibration error).
7. The best model (by test ROC-AUC) is designated the production scorer and wrapped by
   `RiskScoringEngine` and `ExplainabilityEngine`.
8. Borrower segments are identified separately via K-Means clustering on a curated
   numeric/ordinal feature space, profiled, and named directly from the data.
"""
)

render_section_header("Technology Stack")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Data & ML**")
    st.markdown("- pandas, numpy\n- scikit-learn\n- XGBoost\n- SHAP\n- UMAP")
with col2:
    st.markdown("**Analysis & Visualization**")
    st.markdown("- matplotlib, seaborn\n- scipy, statsmodels\n- Jupyter notebooks")
with col3:
    st.markdown("**Application**")
    st.markdown("- Streamlit\n- joblib (serialization)\n- Python 3.12")

render_section_header("Project Limitations")
st.markdown(
    """
- **No legally protected class attributes** (race, gender, age, etc.) are present in the dataset —
  the fairness assessment (Model Explainability / Phase 4A notebook) can only speak to parity across
  business/financial attributes, not legally protected classes.
- **Indiana-only scope**: findings may not generalize to other states' borrower populations or
  regulatory environments without re-validation.
- **Historical data**: the model reflects patterns in past LendingClub originations and does not
  account for subsequent changes in the broader credit or economic environment.
- **Static thresholds**: risk tiers and lending actions are configurable but require a human
  decision-maker to review and adjust them periodically, not an automated feedback loop.
"""
)

render_section_header("Future Improvements")
st.markdown(
    """
- Incorporate macroeconomic indicators (unemployment rate, interest rate environment at origination)
  as additional model features.
- Extend the fairness assessment with additional business-relevant subgroup analysis as more data
  becomes available.
- Add automated model-drift monitoring so the production model is retrained/re-validated on a
  regular cadence rather than manually.
- Explore ensembling the three Phase 3 models rather than selecting a single production scorer.
"""
)

st.caption("MGMT 59000 Capstone — Purdue University System — Summer 2026")
