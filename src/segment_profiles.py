"""
segment_profiles.py
=====================
Phase 4B reusable module: comprehensive per-cluster profiling,
data-driven business naming, business-action recommendations, segment
comparison tables, and exportable segment reports.

This module never fits a clustering model itself -- it consumes cluster
LABELS (already produced by `cluster_analysis.py` / `SegmentationEngine`)
plus raw borrower data and turns them into the business-facing artifacts
Phase 4B calls for: comprehensive profiles, intuitive segment names
("Prime Borrowers", "High Risk Borrowers", ...) assigned from the data
rather than assumed in advance, and concrete lending/marketing/portfolio
recommendations per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import interpretation_utils, utils

logger = utils.get_logger(__name__)

# Columns profiled for every segment, in the order requested by the
# Phase 4B brief ("Typical Income", "Typical DTI", ...).
PROFILE_NUMERIC_COLUMNS: List[str] = [
    "annual_inc", "dti", "loan_amnt", "int_rate", "emp_length_years",
    "revol_util", "revol_bal", "open_acc", "total_acc", "mort_acc",
]
PROFILE_CATEGORICAL_COLUMNS: List[str] = ["grade", "home_ownership", "purpose"]


@dataclass
class SegmentProfile:
    """Everything computed/known about ONE borrower segment."""

    cluster_id: int
    segment_name: str
    n_borrowers: int
    pct_of_portfolio: float
    typical_income: float
    typical_dti: float
    typical_loan_amount: float
    typical_interest_rate: float
    typical_loan_grade: str
    typical_employment_length: float
    typical_home_ownership: str
    typical_loan_purpose: str
    average_default_rate: Optional[float]
    average_credit_utilization: float
    risk_tier: str

    def to_dict(self) -> dict:
        """Plain-dict representation (JSON/Streamlit-friendly)."""
        return {
            "cluster_id": self.cluster_id,
            "segment_name": self.segment_name,
            "n_borrowers": self.n_borrowers,
            "pct_of_portfolio": self.pct_of_portfolio,
            "typical_income": self.typical_income,
            "typical_dti": self.typical_dti,
            "typical_loan_amount": self.typical_loan_amount,
            "typical_interest_rate": self.typical_interest_rate,
            "typical_loan_grade": self.typical_loan_grade,
            "typical_employment_length": self.typical_employment_length,
            "typical_home_ownership": self.typical_home_ownership,
            "typical_loan_purpose": self.typical_loan_purpose,
            "average_default_rate": self.average_default_rate,
            "average_credit_utilization": self.average_credit_utilization,
            "risk_tier": self.risk_tier,
        }


@dataclass
class SegmentRecommendation:
    """Business-action recommendations for ONE borrower segment."""

    cluster_id: int
    segment_name: str
    primary_risk_level: str
    lending_recommendation: str
    interest_rate_strategy: str
    underwriting_strategy: str
    manual_review_requirement: str
    marketing_strategy: str
    portfolio_management_notes: str

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "segment_name": self.segment_name,
            "primary_risk_level": self.primary_risk_level,
            "lending_recommendation": self.lending_recommendation,
            "interest_rate_strategy": self.interest_rate_strategy,
            "underwriting_strategy": self.underwriting_strategy,
            "manual_review_requirement": self.manual_review_requirement,
            "marketing_strategy": self.marketing_strategy,
            "portfolio_management_notes": self.portfolio_management_notes,
        }


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _mode_or_nan(series: pd.Series):
    """Return the most frequent value in a Series, or NaN if the Series is empty."""
    modes = series.mode()
    return modes.iloc[0] if len(modes) > 0 else np.nan


def build_cluster_profile_table(
    X: pd.DataFrame, labels: np.ndarray, default_flags: Optional[pd.Series] = None,
    numeric_columns: List[str] = PROFILE_NUMERIC_COLUMNS, categorical_columns: List[str] = PROFILE_CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """
    Build the raw (pre-naming) per-cluster profile table: mean of every
    numeric column and mode of every categorical column, plus cluster
    size and (if provided) average default rate.

    Parameters
    ----------
    X : pd.DataFrame
        Raw (pre-preprocessing) borrower feature rows, aligned row-for-
        row with `labels`.
    labels : np.ndarray
        Cluster assignment per row.
    default_flags : pd.Series, optional
        Binary actual-outcome column (`config.TARGET_COLUMN`), aligned
        with `X`, used to compute each cluster's average default rate.
        Optional because a live/unlabeled scoring batch has no known
        outcome yet.
    numeric_columns : list[str]
    categorical_columns : list[str]

    Returns
    -------
    pd.DataFrame
        Indexed by cluster_id, with columns: n_borrowers,
        pct_of_portfolio, one column per numeric feature (its mean), one
        column per categorical feature (its mode), and
        average_default_rate (NaN if `default_flags` not supplied).
    """
    working = X.copy()
    working["cluster"] = labels
    if default_flags is not None:
        working["_default_flag"] = np.asarray(default_flags)

    n_total = len(working)
    rows = []
    for cluster_id, group in working.groupby("cluster"):
        row: Dict[str, object] = {
            "cluster": cluster_id,
            "n_borrowers": len(group),
            "pct_of_portfolio": len(group) / n_total,
        }
        for col in numeric_columns:
            if col in group.columns:
                row[col] = group[col].mean()
        for col in categorical_columns:
            if col in group.columns:
                row[col] = _mode_or_nan(group[col])
        if default_flags is not None:
            row["average_default_rate"] = group["_default_flag"].mean()
        else:
            row["average_default_rate"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("cluster").sort_index()


# ---------------------------------------------------------------------------
# Data-driven business naming
# ---------------------------------------------------------------------------


def assign_segment_names(profile_table: pd.DataFrame) -> Dict[int, str]:
    """
    Assign an intuitive, DATA-DRIVEN business name to every cluster,
    based on where that cluster's average income, DTI, interest rate,
    and (if available) default rate fall RELATIVE to the other clusters
    -- never a fixed, assumption-based label independent of the actual
    numbers, per the Phase 4B requirement that "names should be based on
    actual data rather than assumptions."

    Naming logic (applied in priority order per cluster, using
    population-relative z-scores of income, DTI, interest rate, and
    default rate where available):
        1. Highest default rate (or, if unavailable, highest interest
           rate + highest DTI) -> "High Risk Borrowers"
        2. Lowest default rate/interest rate + highest income -> "Prime Borrowers"
        3. High DTI relative to income, but not the single riskiest -> "High Debt Borrowers"
        4. Low credit utilization/short history but decent income -> "Credit Rebuilders"
        5. Below-average employment length + above-average income -> "Young Professionals"
        6. Everything else, ranked by risk -> "Moderate Risk Borrowers" (+ index suffix if >1 remain)

    Parameters
    ----------
    profile_table : pd.DataFrame
        Output of `build_cluster_profile_table` (must include at least
        `annual_inc`, `dti`, `int_rate`; `average_default_rate` and
        `emp_length_years` used if present).

    Returns
    -------
    dict[int, str]
        cluster_id -> assigned business name.
    """
    table = profile_table.copy()

    def _z(col: str) -> pd.Series:
        if col not in table.columns or table[col].std(ddof=0) == 0 or table[col].isna().all():
            return pd.Series(0.0, index=table.index)
        return (table[col] - table[col].mean()) / table[col].std(ddof=0)

    risk_signal = (
        _z("average_default_rate") if table["average_default_rate"].notna().any()
        else (_z("int_rate") + _z("dti")) / 2
    )
    income_z = _z("annual_inc")
    dti_z = _z("dti")
    emp_z = _z("emp_length_years")
    util_z = _z("revol_util")

    remaining = set(table.index)
    names: Dict[int, str] = {}

    # 1. Highest risk signal -> High Risk Borrowers
    if remaining:
        candidate = risk_signal.loc[list(remaining)].idxmax()
        if risk_signal.loc[candidate] > 0.5:
            names[candidate] = "High Risk Borrowers"
            remaining.discard(candidate)

    # 2. Lowest risk + high income -> Prime Borrowers
    if remaining:
        prime_score = (-risk_signal + income_z).loc[list(remaining)]
        candidate = prime_score.idxmax()
        if risk_signal.loc[candidate] < 0 and income_z.loc[candidate] > 0:
            names[candidate] = "Prime Borrowers"
            remaining.discard(candidate)

    # 3. High DTI (debt burden) but not already labeled highest risk -> High Debt Borrowers
    if remaining:
        candidate = dti_z.loc[list(remaining)].idxmax()
        if dti_z.loc[candidate] > 0.5:
            names[candidate] = "High Debt Borrowers"
            remaining.discard(candidate)

    # 4. Low utilization, moderate/lower income, average risk -> Credit Rebuilders
    if remaining:
        candidate = (-util_z).loc[list(remaining)].idxmax()
        if util_z.loc[candidate] < -0.3 and risk_signal.loc[candidate] <= 0.5:
            names[candidate] = "Credit Rebuilders"
            remaining.discard(candidate)

    # 5. Short employment tenure + above-average income -> Young Professionals
    if remaining:
        candidate = (-emp_z + income_z).loc[list(remaining)].idxmax()
        if emp_z.loc[candidate] < 0 and income_z.loc[candidate] > 0:
            names[candidate] = "Young Professionals"
            remaining.discard(candidate)

    # 6. Everything else -> Moderate Risk Borrowers (numbered if more than one remains)
    remaining_sorted = sorted(remaining, key=lambda c: risk_signal.loc[c])
    for i, cluster_id in enumerate(remaining_sorted):
        suffix = f" ({i + 1})" if len(remaining_sorted) > 1 else ""
        names[cluster_id] = f"Moderate Risk Borrowers{suffix}"

    logger.info("Assigned data-driven segment names: %s", names)
    return names


# ---------------------------------------------------------------------------
# Full segment profile construction
# ---------------------------------------------------------------------------


def build_segment_profiles(
    profile_table: pd.DataFrame, segment_names: Dict[int, str], risk_tier_lookup: Optional[Dict[int, str]] = None,
) -> Dict[int, SegmentProfile]:
    """
    Combine the raw profile table and assigned names into a
    `SegmentProfile` per cluster -- the structured object
    `SegmentationEngine.generate_cluster_profile()` returns and
    `describe_segment()` renders as text.

    Parameters
    ----------
    profile_table : pd.DataFrame
        Output of `build_cluster_profile_table`.
    segment_names : dict[int, str]
        Output of `assign_segment_names`.
    risk_tier_lookup : dict[int, str], optional
        Maps cluster_id -> a `RiskThresholdConfig` risk-tier label (e.g.
        "High Risk"), typically derived from each cluster's average
        predicted default probability (see `segmentation_engine.py`).
        Falls back to a simple default-rate-based tier if not supplied.

    Returns
    -------
    dict[int, SegmentProfile]
    """
    profiles: Dict[int, SegmentProfile] = {}
    for cluster_id, row in profile_table.iterrows():
        if risk_tier_lookup and cluster_id in risk_tier_lookup:
            risk_tier = risk_tier_lookup[cluster_id]
        elif pd.notna(row.get("average_default_rate")):
            rate = row["average_default_rate"]
            risk_tier = "Low Risk" if rate < 0.15 else "Moderate Risk" if rate < 0.35 else "High Risk" if rate < 0.6 else "Very High Risk"
        else:
            risk_tier = "Unknown"

        profiles[cluster_id] = SegmentProfile(
            cluster_id=int(cluster_id),
            segment_name=segment_names.get(cluster_id, f"Cluster {cluster_id}"),
            n_borrowers=int(row["n_borrowers"]),
            pct_of_portfolio=float(row["pct_of_portfolio"]),
            typical_income=float(row.get("annual_inc", np.nan)),
            typical_dti=float(row.get("dti", np.nan)),
            typical_loan_amount=float(row.get("loan_amnt", np.nan)),
            typical_interest_rate=float(row.get("int_rate", np.nan)),
            typical_loan_grade=str(row.get("grade", "N/A")),
            typical_employment_length=float(row.get("emp_length_years", np.nan)),
            typical_home_ownership=str(row.get("home_ownership", "N/A")),
            typical_loan_purpose=str(row.get("purpose", "N/A")),
            average_default_rate=(float(row["average_default_rate"]) if pd.notna(row.get("average_default_rate")) else None),
            average_credit_utilization=float(row.get("revol_util", np.nan)),
            risk_tier=risk_tier,
        )
    return profiles


def describe_segment_text(profile: SegmentProfile) -> str:
    """
    Render a `SegmentProfile` as an executive-friendly paragraph -- the
    text `SegmentationEngine.describe_segment()` returns.

    Parameters
    ----------
    profile : SegmentProfile

    Returns
    -------
    str
    """
    default_text = (
        f"an average default rate of {profile.average_default_rate:.1%}"
        if profile.average_default_rate is not None else "an unknown default rate (no outcome data available)"
    )
    return (
        f"{profile.segment_name} ({profile.n_borrowers:,} borrowers, "
        f"{profile.pct_of_portfolio:.1%} of the portfolio) typically have "
        f"${profile.typical_income:,.0f} in annual income, a "
        f"{profile.typical_dti:.1f}% debt-to-income ratio, and take "
        f"${profile.typical_loan_amount:,.0f} loans at a "
        f"{profile.typical_interest_rate:.1f}% interest rate "
        f"(typical grade: {profile.typical_loan_grade}). Most are "
        f"{profile.typical_home_ownership.lower()} homeowners/renters seeking loans for "
        f"{profile.typical_loan_purpose.replace('_', ' ')}, with "
        f"{profile.typical_employment_length:.1f} years of typical employment tenure. "
        f"This segment shows {default_text} and is classified as {profile.risk_tier}."
    )


# ---------------------------------------------------------------------------
# Business recommendations
# ---------------------------------------------------------------------------


_RECOMMENDATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Low Risk": {
        "lending_recommendation": "Approve readily under standard terms; strong candidate for pre-approved offers.",
        "interest_rate_strategy": "Offer at or below the base rate to remain competitive for this low-risk, high-value segment.",
        "underwriting_strategy": "Streamlined/automated underwriting; minimal additional documentation required.",
        "manual_review_requirement": "Not required except for loans above standard size limits.",
        "marketing_strategy": "Prioritize for loyalty offers, rate-reduction refinancing, and referral programs.",
        "portfolio_management_notes": "Low expected loss contribution; a larger share of this segment improves overall portfolio quality.",
    },
    "Moderate Risk": {
        "lending_recommendation": "Approve with standard conditions; monitor for early delinquency signals post-origination.",
        "interest_rate_strategy": "Price at or slightly above the base rate to reflect moderate risk.",
        "underwriting_strategy": "Standard underwriting with routine income/employment verification.",
        "manual_review_requirement": "Spot-check a sample of applications; full review not required for most.",
        "marketing_strategy": "Target with rate-competitive offers conditioned on maintaining/improving credit standing.",
        "portfolio_management_notes": "Represents the core of the portfolio; track default-rate trend over time as an early-warning signal.",
    },
    "High Risk": {
        "lending_recommendation": "Approve selectively, typically with reduced principal or added conditions.",
        "interest_rate_strategy": "Apply a meaningful rate markup to price in elevated risk.",
        "underwriting_strategy": "Enhanced underwriting: verify income and debt obligations directly rather than relying on self-reported figures.",
        "manual_review_requirement": "Recommend manual credit-analyst review before approval.",
        "marketing_strategy": "Avoid broad promotional offers; consider secured or co-signed loan products instead.",
        "portfolio_management_notes": "Cap this segment's share of total originations to manage aggregate expected loss.",
    },
    "Very High Risk": {
        "lending_recommendation": "Decline under standard policy; require an exception process for any approval.",
        "interest_rate_strategy": "If approved via exception, price well above standard rates to reflect very high expected loss.",
        "underwriting_strategy": "Full manual underwriting with independent verification of every major input.",
        "manual_review_requirement": "Mandatory senior credit-analyst review for any exception approval.",
        "marketing_strategy": "Do not target with acquisition marketing; consider credit-building/secured-product referrals instead.",
        "portfolio_management_notes": "Minimize origination volume; concentration in this segment materially raises expected portfolio loss.",
    },
    "Unknown": {
        "lending_recommendation": "Insufficient outcome data to recommend a lending policy for this segment; treat as Moderate Risk pending more data.",
        "interest_rate_strategy": "Price at the base rate until sufficient outcome data accumulates.",
        "underwriting_strategy": "Standard underwriting pending further validation.",
        "manual_review_requirement": "Recommend manual review until this segment's risk profile is validated with outcome data.",
        "marketing_strategy": "Limit promotional targeting until risk profile is confirmed.",
        "portfolio_management_notes": "Monitor closely as outcome data accumulates before setting portfolio limits.",
    },
}


def recommend_segment_actions(profile: SegmentProfile) -> SegmentRecommendation:
    """
    Generate concrete business-action recommendations for one segment,
    using `_RECOMMENDATION_TEMPLATES` keyed on the segment's risk tier
    (itself derived from data -- see `build_segment_profiles`), not a
    fixed rule per segment NAME (so a newly-named segment still gets a
    sensible recommendation based on its actual measured risk).

    Parameters
    ----------
    profile : SegmentProfile

    Returns
    -------
    SegmentRecommendation
    """
    template = _RECOMMENDATION_TEMPLATES.get(profile.risk_tier, _RECOMMENDATION_TEMPLATES["Unknown"])
    return SegmentRecommendation(
        cluster_id=profile.cluster_id,
        segment_name=profile.segment_name,
        primary_risk_level=profile.risk_tier,
        lending_recommendation=template["lending_recommendation"],
        interest_rate_strategy=template["interest_rate_strategy"],
        underwriting_strategy=template["underwriting_strategy"],
        manual_review_requirement=template["manual_review_requirement"],
        marketing_strategy=template["marketing_strategy"],
        portfolio_management_notes=template["portfolio_management_notes"],
    )


# ---------------------------------------------------------------------------
# Segment comparison + exportable reports
# ---------------------------------------------------------------------------


def build_segment_comparison_table(profiles: Dict[int, SegmentProfile]) -> pd.DataFrame:
    """
    Build the executive segment-comparison table required by Phase 4B:
    income, interest rate, loan grade, DTI, employment length, default
    rate, risk tier, and cluster size, one row per segment.

    Parameters
    ----------
    profiles : dict[int, SegmentProfile]

    Returns
    -------
    pd.DataFrame
        Sorted by average_default_rate descending (riskiest segment
        first) when default-rate data is available, else by cluster_id.
    """
    rows = [
        {
            "segment_name": p.segment_name,
            "n_borrowers": p.n_borrowers,
            "pct_of_portfolio": p.pct_of_portfolio,
            "typical_income": p.typical_income,
            "typical_interest_rate": p.typical_interest_rate,
            "typical_loan_grade": p.typical_loan_grade,
            "typical_dti": p.typical_dti,
            "typical_employment_length": p.typical_employment_length,
            "average_default_rate": p.average_default_rate,
            "risk_tier": p.risk_tier,
        }
        for p in profiles.values()
    ]
    table = pd.DataFrame(rows)
    sort_col = "average_default_rate" if table["average_default_rate"].notna().any() else "typical_interest_rate"
    return table.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def generate_segment_executive_summary(profiles: Dict[int, SegmentProfile]) -> str:
    """
    Generate an executive-friendly paragraph summarizing the overall
    segmentation: how many segments, which is riskiest/safest, and the
    broad shape of the portfolio.

    Parameters
    ----------
    profiles : dict[int, SegmentProfile]

    Returns
    -------
    str
    """
    comparison = build_segment_comparison_table(profiles)
    n_segments = len(comparison)
    riskiest = comparison.iloc[0]
    safest = comparison.iloc[-1]

    sentences = [
        f"Borrower segmentation identified {n_segments} distinct groups with meaningfully "
        f"different financial profiles and lending risk."
    ]
    if comparison["average_default_rate"].notna().any():
        sentences.append(
            f"{riskiest['segment_name']} carries the highest observed default rate "
            f"({riskiest['average_default_rate']:.1%}, {riskiest['n_borrowers']:,} borrowers), while "
            f"{safest['segment_name']} shows the lowest ({safest['average_default_rate']:.1%}, "
            f"{safest['n_borrowers']:,} borrowers)."
        )
    else:
        sentences.append(
            f"{riskiest['segment_name']} carries the highest typical interest rate "
            f"({riskiest['typical_interest_rate']:.1f}%), while {safest['segment_name']} shows the "
            f"lowest ({safest['typical_interest_rate']:.1f}%), in the absence of outcome data to compute default rates directly."
        )
    sentences.append(
        "These segments complement (rather than replace) the supervised default-risk models: "
        "clustering groups borrowers by overall financial profile, which can guide underwriting "
        "policy and marketing strategy at the segment level even where an individual borrower's "
        "predicted probability, from Phase 3's models, remains the primary basis for a specific "
        "lending decision."
    )
    return " ".join(sentences)


def export_segment_summary_report(
    profiles: Dict[int, SegmentProfile], recommendations: Dict[int, SegmentRecommendation],
) -> "interpretation_utils.ExportableReport":
    """
    Build an `ExportableReport` bundling the segment comparison table,
    executive summary, and every segment's profile + recommendations --
    the "Segment Summary" / "Executive Report" exportable reports
    required by Phase 4B.

    Parameters
    ----------
    profiles : dict[int, SegmentProfile]
    recommendations : dict[int, SegmentRecommendation]

    Returns
    -------
    interpretation_utils.ExportableReport
    """
    comparison_table = build_segment_comparison_table(profiles)
    sections = {
        "Executive Summary": generate_segment_executive_summary(profiles),
        "Segment Comparison": interpretation_utils.dataframe_to_markdown_table(comparison_table, max_rows=20),
    }
    for cluster_id, profile in profiles.items():
        recommendation = recommendations.get(cluster_id)
        section_body = describe_segment_text(profile)
        if recommendation:
            section_body += (
                f"\n\n**Lending recommendation:** {recommendation.lending_recommendation}\n\n"
                f"**Interest rate strategy:** {recommendation.interest_rate_strategy}\n\n"
                f"**Underwriting strategy:** {recommendation.underwriting_strategy}\n\n"
                f"**Manual review requirement:** {recommendation.manual_review_requirement}\n\n"
                f"**Marketing strategy:** {recommendation.marketing_strategy}\n\n"
                f"**Portfolio management notes:** {recommendation.portfolio_management_notes}"
            )
        sections[profile.segment_name] = section_body

    return interpretation_utils.ExportableReport(title="Borrower Segmentation Report", sections=sections)
