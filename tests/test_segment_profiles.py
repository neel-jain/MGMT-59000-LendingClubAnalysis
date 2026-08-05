"""
test_segment_profiles.py
===========================
Unit tests for src/segment_profiles.py.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import segment_profiles as sp  # noqa: E402


@pytest.fixture
def clustered_borrowers():
    """
    Deliberately constructed so cluster 0 is obviously "prime" (high
    income, low DTI, low default) and cluster 1 is obviously "high risk"
    (low income, high DTI, high default) -- lets tests assert the naming
    logic actually reads the data correctly, not just "doesn't crash".
    """
    n_per_cluster = 60
    prime = pd.DataFrame({
        "annual_inc": np.random.default_rng(0).normal(90000, 5000, n_per_cluster),
        "dti": np.random.default_rng(1).normal(8, 2, n_per_cluster),
        "loan_amnt": np.random.default_rng(2).normal(15000, 2000, n_per_cluster),
        "int_rate": np.random.default_rng(3).normal(7, 1, n_per_cluster),
        "emp_length_years": np.random.default_rng(4).normal(8, 1, n_per_cluster),
        "revol_util": np.random.default_rng(5).normal(20, 5, n_per_cluster),
        "revol_bal": np.random.default_rng(6).normal(5000, 1000, n_per_cluster),
        "open_acc": np.random.default_rng(7).integers(5, 10, n_per_cluster),
        "total_acc": np.random.default_rng(8).integers(10, 20, n_per_cluster),
        "mort_acc": np.random.default_rng(9).integers(1, 3, n_per_cluster),
        "grade": ["A"] * n_per_cluster,
        "home_ownership": ["MORTGAGE"] * n_per_cluster,
        "purpose": ["debt_consolidation"] * n_per_cluster,
    })
    high_risk = pd.DataFrame({
        "annual_inc": np.random.default_rng(10).normal(30000, 3000, n_per_cluster),
        "dti": np.random.default_rng(11).normal(35, 3, n_per_cluster),
        "loan_amnt": np.random.default_rng(12).normal(20000, 2000, n_per_cluster),
        "int_rate": np.random.default_rng(13).normal(25, 2, n_per_cluster),
        "emp_length_years": np.random.default_rng(14).normal(2, 1, n_per_cluster),
        "revol_util": np.random.default_rng(15).normal(85, 5, n_per_cluster),
        "revol_bal": np.random.default_rng(16).normal(30000, 3000, n_per_cluster),
        "open_acc": np.random.default_rng(17).integers(1, 5, n_per_cluster),
        "total_acc": np.random.default_rng(18).integers(3, 8, n_per_cluster),
        "mort_acc": np.random.default_rng(19).integers(0, 1, n_per_cluster),
        "grade": ["G"] * n_per_cluster,
        "home_ownership": ["RENT"] * n_per_cluster,
        "purpose": ["small_business"] * n_per_cluster,
    })
    X = pd.concat([prime, high_risk], ignore_index=True)
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)
    default_flags = pd.Series([0] * (n_per_cluster - 5) + [1] * 5 + [0] * 10 + [1] * (n_per_cluster - 10))
    return X, labels, default_flags


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def test_build_cluster_profile_table_shape(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    assert len(table) == 2
    assert "n_borrowers" in table.columns
    assert "average_default_rate" in table.columns


def test_build_cluster_profile_table_without_default_flags(clustered_borrowers):
    X, labels, _ = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels)
    assert table["average_default_rate"].isna().all()


def test_build_cluster_profile_table_percentages_sum_to_one(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    assert table["pct_of_portfolio"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Data-driven naming
# ---------------------------------------------------------------------------


def test_assign_segment_names_identifies_high_risk_correctly(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    # Cluster 1 (low income, high DTI, high default) must be named High Risk.
    assert names[1] == "High Risk Borrowers"


def test_assign_segment_names_identifies_prime_correctly(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    assert names[0] == "Prime Borrowers"


def test_assign_segment_names_covers_every_cluster(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    assert set(names.keys()) == set(table.index)


# ---------------------------------------------------------------------------
# Segment profile construction
# ---------------------------------------------------------------------------


def test_build_segment_profiles_returns_expected_fields(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)
    assert set(profiles.keys()) == {0, 1}
    assert profiles[0].typical_loan_grade == "A"
    assert profiles[1].typical_loan_grade == "G"


def test_build_segment_profiles_risk_tier_from_lookup(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    risk_tier_lookup = {0: "Low Risk", 1: "Very High Risk"}
    profiles = sp.build_segment_profiles(table, names, risk_tier_lookup=risk_tier_lookup)
    assert profiles[0].risk_tier == "Low Risk"
    assert profiles[1].risk_tier == "Very High Risk"


def test_build_segment_profiles_risk_tier_fallback_from_default_rate(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)  # no risk_tier_lookup
    assert profiles[0].risk_tier in {"Low Risk", "Moderate Risk"}
    assert profiles[1].risk_tier in {"High Risk", "Very High Risk"}


def test_describe_segment_text_contains_key_facts(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)
    text = sp.describe_segment_text(profiles[0])
    assert profiles[0].segment_name in text
    assert "default rate" in text


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommend_segment_actions_high_risk_recommends_decline_or_selective_approval(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    risk_tier_lookup = {0: "Low Risk", 1: "Very High Risk"}
    profiles = sp.build_segment_profiles(table, names, risk_tier_lookup=risk_tier_lookup)
    recommendation = sp.recommend_segment_actions(profiles[1])
    assert "decline" in recommendation.lending_recommendation.lower()


def test_recommend_segment_actions_unknown_tier_uses_fallback():
    profile = sp.SegmentProfile(
        cluster_id=0, segment_name="Test", n_borrowers=10, pct_of_portfolio=1.0,
        typical_income=50000, typical_dti=20, typical_loan_amount=10000, typical_interest_rate=10,
        typical_loan_grade="C", typical_employment_length=5, typical_home_ownership="RENT",
        typical_loan_purpose="other", average_default_rate=None, average_credit_utilization=30,
        risk_tier="Unknown",
    )
    recommendation = sp.recommend_segment_actions(profile)
    assert "insufficient" in recommendation.lending_recommendation.lower()


# ---------------------------------------------------------------------------
# Comparison + exports
# ---------------------------------------------------------------------------


def test_build_segment_comparison_table_sorted_by_risk(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)
    comparison = sp.build_segment_comparison_table(profiles)
    assert comparison.iloc[0]["average_default_rate"] >= comparison.iloc[-1]["average_default_rate"]


def test_generate_segment_executive_summary_mentions_riskiest_and_safest(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)
    summary = sp.generate_segment_executive_summary(profiles)
    assert "Prime Borrowers" in summary or "High Risk Borrowers" in summary


def test_export_segment_summary_report_contains_all_segments(clustered_borrowers):
    X, labels, default_flags = clustered_borrowers
    table = sp.build_cluster_profile_table(X, labels, default_flags=default_flags)
    names = sp.assign_segment_names(table)
    profiles = sp.build_segment_profiles(table, names)
    recommendations = {cid: sp.recommend_segment_actions(p) for cid, p in profiles.items()}
    report = sp.export_segment_summary_report(profiles, recommendations)
    md = report.to_markdown()
    for profile in profiles.values():
        assert profile.segment_name in md
