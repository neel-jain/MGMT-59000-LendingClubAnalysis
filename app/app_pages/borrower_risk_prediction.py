"""
borrower_risk_prediction.py
=============================
Phase 5, Page 4: Borrower Risk Prediction.

An interactive single-borrower prediction form. All scoring comes from
`RiskScoringEngine`; all explanation comes from `ExplainabilityEngine`.
This page only collects form input, assembles it into the one-row
DataFrame the fitted `Pipeline` expects, and displays whatever the two
engines return -- no probability, threshold, or SHAP logic is
implemented here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from common import (
    apply_global_style, download_report_buttons, get_explainability_engine, get_risk_engine,
    load_cleaned_dataset, render_executive_summary_box, render_missing_artifact_notice,
    render_page_header, render_section_header,
)
from src import config, utils

logger = utils.get_logger(__name__)

apply_global_style()
render_page_header(
    "Borrower Risk Prediction",
    "Enter a borrower's information to generate a live risk assessment.",
)

model_key = st.session_state.get("selected_model_key", config.PRODUCTION_MODEL_KEY)
risk_engine = get_risk_engine(model_key)
explain_engine = get_explainability_engine(model_key)
show_advanced_statistics = st.session_state.get("show_advanced_statistics", False)

if risk_engine is None:
    render_missing_artifact_notice("The scoring model", "python -m src.train_models")
    st.stop()

cleaned_df = load_cleaned_dataset()


def _default(col: str, fallback):
    """Pull a sensible default (median for numeric, mode for categorical) from the cleaned dataset, or fall back."""
    if cleaned_df.empty or col not in cleaned_df.columns:
        return fallback
    series = cleaned_df[col].dropna()
    if series.empty:
        return fallback
    return series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
render_section_header("Borrower Information")

with st.form("borrower_prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        annual_inc = st.number_input("Annual Income ($)", min_value=0.0, value=float(_default("annual_inc", 55000.0)), step=1000.0)
        emp_length_years = st.slider("Employment Length (years)", 0.0, 10.0, float(_default("emp_length_years", 5.0)), step=0.5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)
    with col2:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, value=float(_default("loan_amnt", 15000.0)), step=500.0)
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"],
            index=0,
        )
        int_rate = st.slider("Interest Rate (%)", 5.0, 31.0, float(_default("int_rate", 13.0)), step=0.1)
    with col3:
        grade = st.selectbox("LendingClub Grade", config.ORDINAL_CATEGORY_ORDER[0], index=2)
        dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, float(_default("dti", 18.0)), step=0.5)
        term = st.selectbox("Loan Term", [" 36 months", " 60 months"], index=0)

    with st.expander("Advanced: additional credit-profile fields"):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=float(_default("installment", loan_amnt / 36)), step=10.0)
            open_acc = st.number_input("Open Credit Accounts", min_value=0, value=int(_default("open_acc", 10)), step=1)
            total_acc = st.number_input("Total Credit Accounts", min_value=0, value=int(_default("total_acc", 20)), step=1)
        with adv2:
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, value=float(_default("revol_bal", 12000.0)), step=500.0)
            revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, float(_default("revol_util", 45.0)), step=1.0)
            mort_acc = st.number_input("Mortgage Accounts", min_value=0, value=int(_default("mort_acc", 1)), step=1)
        with adv3:
            delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", min_value=0, value=int(_default("delinq_2yrs", 0)), step=1)
            pub_rec = st.number_input("Public Derogatory Records", min_value=0, value=int(_default("pub_rec", 0)), step=1)
            pub_rec_bankruptcies = st.number_input("Bankruptcies on Record", min_value=0, value=int(_default("pub_rec_bankruptcies", 0)), step=1)
        verification_status = st.selectbox("Income Verification Status", ["Verified", "Source Verified", "Not Verified"], index=0)
        initial_list_status = st.selectbox("Initial Listing Status", ["w", "f"], index=0)
        application_type = st.selectbox("Application Type", ["Individual", "Joint App"], index=0)

    submitted = st.form_submit_button("🎯 Predict Risk")

if submitted:
    borrower = pd.DataFrame([{
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
        "grade": grade, "home_ownership": home_ownership, "annual_inc": annual_inc,
        "verification_status": verification_status, "purpose": purpose, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "initial_list_status": initial_list_status,
        "application_type": application_type, "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies, "emp_length_years": emp_length_years,
    }])
    st.session_state["last_borrower_input"] = borrower

if "last_borrower_input" in st.session_state:
    borrower = st.session_state["last_borrower_input"]

    logger.info("Prediction requested: model=%s, loan_amnt=%.0f, grade=%s", model_key, loan_amnt, grade)
    try:
        with st.spinner("Scoring borrower..."):
            summary = risk_engine.generate_prediction_summary(borrower)
    except Exception as exc:  # noqa: BLE001 -- surface any engine failure as a friendly message, not a crash
        logger.error("RiskScoringEngine.generate_prediction_summary failed: %s", exc, exc_info=True)
        st.error(
            "Something went wrong while scoring this borrower. This is usually caused by an "
            "unexpected combination of input values. Please review the form and try again.",
            icon="🚫",
        )
        st.stop()

    render_section_header("Risk Assessment Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability of Default", f"{summary.default_probability:.1%}")
    col2.metric("Risk Tier", summary.risk_tier)
    col3.metric("Confidence Score", f"{summary.confidence_score:.0f}/100")
    col4.metric("Recommended Action", summary.recommended_action)

    col5, col6 = st.columns(2)
    col5.metric("Suggested Interest Rate", f"{summary.recommended_interest_rate:.2f}%")
    col6.metric("Model-Driven Loan Grade", summary.recommended_loan_grade)

    # --- Probability gauge + risk meter ---
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.barh([0], [1], color="#E1E5E9", height=0.5)
        ax.barh([0], [summary.default_probability], color="#C0392B" if summary.default_probability > 0.35 else "#2E86AB", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Probability of Default")
        ax.set_title("Probability Gauge", fontsize=11, fontweight="bold", loc="left")
        fig.tight_layout()
        st.pyplot(fig)
    with gcol2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.barh([0], [100], color="#E1E5E9", height=0.5)
        ax.barh([0], [summary.risk_score], color="#C0392B" if summary.risk_score > 35 else "#2E86AB", height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Risk Score (0=safest, 100=riskiest)")
        ax.set_title("Risk Meter", fontsize=11, fontweight="bold", loc="left")
        fig.tight_layout()
        st.pyplot(fig)

    # --- Explanation ---
    if explain_engine is not None:
        try:
            with st.spinner("Computing SHAP explanation..."):
                local = explain_engine.explain_prediction(borrower)
        except Exception as exc:  # noqa: BLE001
            logger.error("ExplainabilityEngine.explain_prediction failed: %s", exc, exc_info=True)
            st.error(
                "The risk score above is valid, but the detailed explanation could not be generated "
                "for this borrower. Please try again or contact support if this persists.",
                icon="🚫",
            )
            local = None

        if local is not None:
            render_section_header("Why This Prediction?")
            render_executive_summary_box(local.business_summary)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Top 5 Risk Factors**")
                for f in local.top_risk_factors[:5]:
                    st.markdown(f"- 🔴 {f}")
            with rc2:
                st.markdown("**Top 3 Protective Factors**")
                for f in local.top_protective_factors[:3]:
                    st.markdown(f"- 🟢 {f}")

            if show_advanced_statistics:
                wcol1, wcol2 = st.columns(2)
                with wcol1:
                    fig = explain_engine.generate_waterfall_plot(borrower)
                    st.pyplot(fig)
                with wcol2:
                    fig = explain_engine.generate_force_plot(borrower)
                    st.pyplot(fig)
            else:
                st.info(
                    "Enable 'Show advanced statistics' in the left sidebar to view detailed explanation plots.",
                    icon="ℹ️",
                )

            render_section_header("Exportable Reports")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Risk Assessment Report")
                download_report_buttons(risk_engine.export_prediction_report(borrower), "borrower_risk_assessment", "risk_report")
            with col_b:
                st.caption("Borrower Explanation Report")
                download_report_buttons(explain_engine.export_borrower_explanation_report(borrower), "borrower_explanation", "explain_report")
    else:
        render_missing_artifact_notice("The explainability engine", "python -m src.train_models")
else:
    st.info("Fill in the borrower information above and click **Predict Risk** to generate an assessment.", icon="👆")
