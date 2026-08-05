"""
test_interpretation_utils.py
==============================
Unit tests for src/interpretation_utils.py.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import interpretation_utils as iu  # noqa: E402


# ---------------------------------------------------------------------------
# Feature-name humanization
# ---------------------------------------------------------------------------


def test_humanize_numeric_feature():
    assert iu.humanize_feature_name("numeric__dti") == "Debt-to-income ratio"


def test_humanize_ordinal_feature():
    assert iu.humanize_feature_name("ordinal_categorical__grade") == "LendingClub credit grade"


def test_humanize_onehot_feature_splits_base_and_category():
    result = iu.humanize_feature_name("onehot_categorical__purpose_debt_consolidation")
    assert result == "Loan purpose: Debt Consolidation"


def test_humanize_onehot_multiword_base_column():
    result = iu.humanize_feature_name("onehot_categorical__home_ownership_MORTGAGE")
    assert result.startswith("Home ownership status:")
    assert "Mortgage" in result


def test_humanize_unrecognized_feature_falls_back_gracefully():
    result = iu.humanize_feature_name("numeric__some_unknown_column")
    assert result  # non-empty
    assert "_" not in result  # underscores replaced


def test_humanize_feature_table_adds_label_column():
    df = pd.DataFrame({"feature": ["numeric__dti", "numeric__annual_inc"], "value": [1, 2]})
    result = iu.humanize_feature_table(df)
    assert "feature_label" in result.columns
    assert result.loc[0, "feature_label"] == "Debt-to-income ratio"
    assert list(result.columns).index("feature_label") == list(result.columns).index("feature") + 1


# ---------------------------------------------------------------------------
# Research-question linkage
# ---------------------------------------------------------------------------


def test_link_feature_to_research_question_direct_match():
    result = iu.link_feature_to_research_question("dti")
    assert result is not None and "RQ5" in result


def test_link_feature_to_research_question_technical_name():
    result = iu.link_feature_to_research_question("numeric__dti")
    assert result is not None and "RQ5" in result


def test_link_feature_to_research_question_onehot_name():
    result = iu.link_feature_to_research_question("onehot_categorical__purpose_other")
    assert result is not None and "RQ1" in result


def test_link_feature_to_research_question_no_match():
    assert iu.link_feature_to_research_question("totally_unknown_xyz") is None


# ---------------------------------------------------------------------------
# Business-summary text generation
# ---------------------------------------------------------------------------


def test_generate_borrower_business_summary_contains_key_elements():
    summary = iu.generate_borrower_business_summary(
        risk_tier="High Risk",
        default_probability=0.42,
        top_risk_factors=["Debt-to-income ratio", "Interest rate"],
        top_protective_factors=["Annual income"],
        recommended_action="Manual Review",
    )
    assert "High Risk" in summary
    assert "42%" in summary
    assert "Manual Review" in summary
    assert "SHAP" not in summary  # no ML jargon
    assert "probability distribution" not in summary


def test_generate_borrower_business_summary_no_factors():
    summary = iu.generate_borrower_business_summary(
        risk_tier="Low Risk", default_probability=0.05,
        top_risk_factors=[], top_protective_factors=[], recommended_action="Approve",
    )
    assert "Low Risk" in summary
    assert "Approve" in summary


def test_generate_global_business_summary():
    summary = iu.generate_global_business_summary(
        model_display_name="XGBoost",
        top_features=["Debt-to-income ratio", "Interest rate"],
        least_influential_features=["Application type"],
    )
    assert "XGBoost" in summary
    assert "debt-to-income ratio" in summary or "Debt-to-income ratio" in summary


# ---------------------------------------------------------------------------
# Fairness reporting
# ---------------------------------------------------------------------------


@pytest.fixture
def fairness_data():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], size=n),
        "annual_inc": rng.lognormal(10.8, 0.4, n),
    })
    y_true = rng.integers(0, 2, n)
    y_proba = rng.uniform(0, 1, n)
    return X, pd.Series(y_true), y_proba


def test_fairness_report_shape(fairness_data):
    X, y_true, y_proba = fairness_data
    result = iu.fairness_report(X, y_true, y_proba, group_columns=["home_ownership"])
    assert set(result["group_value"]) <= {"RENT", "MORTGAGE", "OWN"}
    assert "recall" in result.columns
    assert "actual_default_rate" in result.columns


def test_fairness_report_excludes_small_groups():
    X = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 5})  # B has n=5 < 10
    y_true = pd.Series([0, 1] * 27 + [0])
    y_proba = np.random.default_rng(0).uniform(0, 1, 55)
    result = iu.fairness_report(X, y_true, y_proba, group_columns=["group"])
    assert "B" not in set(result["group_value"])
    assert "A" in set(result["group_value"])


def test_bin_column_for_fairness_produces_expected_bins():
    series = pd.Series(range(100))
    binned = iu.bin_column_for_fairness(series, n_bins=4)
    assert binned.nunique() <= 4


def test_summarize_fairness_disparities(fairness_data):
    X, y_true, y_proba = fairness_data
    fairness_table = iu.fairness_report(X, y_true, y_proba, group_columns=["home_ownership"])
    disparities = iu.summarize_fairness_disparities(fairness_table, metric="recall")
    assert "recall_spread" in disparities.columns
    assert (disparities["recall_spread"] >= 0).all()


def test_summarize_fairness_disparities_empty_table():
    empty = pd.DataFrame()
    result = iu.summarize_fairness_disparities(empty, metric="recall")
    assert result.empty


# ---------------------------------------------------------------------------
# Exportable report formatting
# ---------------------------------------------------------------------------


def test_exportable_report_to_markdown():
    report = iu.ExportableReport(title="Test Report", sections={"Section A": "Body text."})
    md = report.to_markdown()
    assert "# Test Report" in md
    assert "## Section A" in md
    assert "Body text." in md


def test_exportable_report_to_json():
    report = iu.ExportableReport(title="Test Report", sections={"Section A": "Body text."})
    json_str = report.to_json()
    assert "Test Report" in json_str
    assert "Section A" in json_str


def test_exportable_report_save_markdown(tmp_path):
    report = iu.ExportableReport(title="Test Report", sections={"A": "B"})
    path = tmp_path / "report.md"
    report.save(path, fmt="markdown")
    assert path.exists()
    assert "# Test Report" in path.read_text()


def test_exportable_report_save_invalid_format_raises(tmp_path):
    report = iu.ExportableReport(title="Test", sections={"A": "B"})
    with pytest.raises(ValueError):
        report.save(tmp_path / "report.xyz", fmt="invalid_format")


def test_dataframe_to_markdown_table():
    df = pd.DataFrame({"feature": ["dti", "int_rate"], "importance": [0.5, 0.3]})
    table = iu.dataframe_to_markdown_table(df)
    assert "| feature | importance |" in table
    assert "dti" in table


def test_dataframe_to_markdown_table_truncates_and_notes(tmp_path):
    df = pd.DataFrame({"x": range(30)})
    table = iu.dataframe_to_markdown_table(df, max_rows=5)
    assert "showing first 5 of 30 rows" in table
