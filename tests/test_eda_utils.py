"""
test_eda_utils.py
==================
Unit tests for src/eda_utils.py (Phase 2 exploratory-analysis, plotting,
and statistical-testing helpers).

Plotting functions are tested only for "does it run and return a
Figure/Axes without raising" — visual correctness is verified manually
in the executed notebook. Statistical-testing and table-building
functions are checked against known, hand-computable inputs so the
numeric results themselves are validated, not just non-crashing code.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for test environments

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, eda_utils  # noqa: E402


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small, deterministic dataset mirroring the cleaned schema."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "loan_amnt": rng.integers(1000, 30000, n).astype(float),
            "int_rate": rng.normal(13, 4, n).clip(5, 30),
            "dti": rng.normal(18, 7, n).clip(0, 45),
            "annual_inc": rng.lognormal(10.8, 0.4, n),
            "grade": rng.choice(list("ABCDE"), size=n),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
            config.TARGET_COLUMN: rng.choice([0, 1], size=n, p=[0.75, 0.25]),
        }
    )


# ---------------------------------------------------------------------------
# Dataset overview / descriptive statistics
# ---------------------------------------------------------------------------


def test_build_dataset_overview_basic_fields(sample_df):
    overview = eda_utils.build_dataset_overview(sample_df)
    assert overview.n_rows == len(sample_df)
    assert overview.n_columns == sample_df.shape[1]
    assert set(overview.columns) == set(sample_df.columns)
    assert overview.duplicate_row_count >= 0


def test_numeric_descriptive_stats_matches_pandas(sample_df):
    stats_df = eda_utils.numeric_descriptive_stats(sample_df, ["loan_amnt", "dti"])
    assert "loan_amnt" in stats_df.index
    assert stats_df.loc["loan_amnt", "mean"] == pytest.approx(sample_df["loan_amnt"].mean(), rel=1e-6)
    assert stats_df.loc["loan_amnt", "median"] == pytest.approx(sample_df["loan_amnt"].median(), rel=1e-6)
    # numeric_descriptive_stats rounds to 3 decimals for readability
    assert stats_df.loc["dti", "max"] == pytest.approx(sample_df["dti"].max(), abs=5e-4)


def test_categorical_frequency_table_sums_to_total(sample_df):
    freq = eda_utils.categorical_frequency_table(sample_df, "grade")
    assert freq["count"].sum() == len(sample_df)
    assert freq["percentage"].sum() == pytest.approx(100.0, abs=0.5)


def test_interpret_skew_kurtosis_returns_string():
    text = eda_utils.interpret_skew_kurtosis(1.5, 2.0)
    assert isinstance(text, str) and "skewed" in text


# ---------------------------------------------------------------------------
# Default-rate business helpers
# ---------------------------------------------------------------------------


def test_default_rate_by_group_known_values():
    df = pd.DataFrame(
        {
            "grade": ["A", "A", "A", "B", "B"],
            config.TARGET_COLUMN: [0, 0, 1, 1, 1],
        }
    )
    result = eda_utils.default_rate_by_group(df, "grade")
    # default_rate_by_group rounds to 4 decimals for readability
    assert result.loc["A", "default_rate"] == pytest.approx(1 / 3, abs=5e-5)
    assert result.loc["B", "default_rate"] == pytest.approx(1.0)
    # Sorted descending by default rate
    assert result.index[0] == "B"


def test_bin_into_quartiles_produces_expected_bin_count(sample_df):
    binned = eda_utils.bin_into_quartiles(sample_df, "annual_inc", q=4)
    assert binned.nunique() <= 4
    assert set(binned.dropna().unique()) <= {"Q1", "Q2", "Q3", "Q4"}


def test_bin_into_bands_assigns_correct_labels():
    df = pd.DataFrame({"int_rate": [5, 12, 18, 25]})
    binned = eda_utils.bin_into_bands(
        df, "int_rate", bins=[0, 10, 15, 20, 30], labels=["low", "mid", "high", "very_high"]
    )
    assert list(binned) == ["low", "mid", "high", "very_high"]


def test_proportion_confidence_interval_bounds_contain_point_estimate():
    point, lo, hi = eda_utils.proportion_confidence_interval(25, 100)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_default_rate_ci_by_group_shape(sample_df):
    result = eda_utils.default_rate_ci_by_group(sample_df, "grade")
    assert set(result.columns) == {"loan_count", "default_count", "default_rate", "ci_lower", "ci_upper"}
    assert (result["ci_lower"] <= result["default_rate"]).all()
    assert (result["default_rate"] <= result["ci_upper"]).all()


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------


def test_cohens_d_zero_for_identical_groups():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert eda_utils.cohens_d(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cramers_v_range(sample_df):
    contingency = pd.crosstab(sample_df["grade"], sample_df[config.TARGET_COLUMN])
    v = eda_utils.cramers_v(contingency)
    assert 0.0 <= v <= 1.0


def test_run_independent_ttest_returns_valid_result(sample_df):
    result = eda_utils.run_independent_ttest(sample_df, "dti", config.TARGET_COLUMN)
    assert isinstance(result.p_value, float)
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size_label == "Cohen's d"
    assert result.is_significant == (result.p_value < result.alpha)


def test_run_anova_returns_valid_result(sample_df):
    result = eda_utils.run_anova(sample_df, "int_rate", "grade")
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size_label == "eta-squared"


def test_run_chi_square_test_returns_valid_result(sample_df):
    result = eda_utils.run_chi_square_test(sample_df, "purpose", config.TARGET_COLUMN)
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size_label == "Cramer's V"


def test_pearson_and_spearman_perfect_positive_correlation():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    pearson_result, spearman_result = eda_utils.pearson_and_spearman(df, "a", "b")
    assert pearson_result.statistic == pytest.approx(1.0)
    assert spearman_result.statistic == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature relationships / multicollinearity
# ---------------------------------------------------------------------------


def test_high_correlation_pairs_detects_known_pair():
    df = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)], "c": np.random.default_rng(1).normal(size=50)})
    corr = df.corr()
    pairs = eda_utils.high_correlation_pairs(corr, threshold=0.9)
    assert len(pairs) >= 1
    assert {"a", "b"} == {pairs.iloc[0]["variable_1"], pairs.iloc[0]["variable_2"]}


def test_high_correlation_pairs_empty_when_no_pairs_exceed_threshold():
    df = pd.DataFrame(np.random.default_rng(2).normal(size=(50, 3)), columns=["x", "y", "z"])
    corr = df.corr()
    pairs = eda_utils.high_correlation_pairs(corr, threshold=0.999)
    assert pairs.empty


def test_variance_inflation_factors_returns_expected_columns(sample_df):
    vif_df = eda_utils.variance_inflation_factors(sample_df, ["loan_amnt", "int_rate", "dti", "annual_inc"])
    assert set(vif_df.columns) == {"variable", "VIF"}
    assert len(vif_df) == 4
    assert (vif_df["VIF"] > 0).all()


# ---------------------------------------------------------------------------
# Plotting functions (smoke tests: run without raising, return expected type)
# ---------------------------------------------------------------------------


def test_plot_categorical_distribution_runs(sample_df):
    fig = eda_utils.plot_categorical_distribution(sample_df, "grade", title="Test")
    assert fig is not None


def test_plot_numeric_distribution_runs(sample_df):
    fig = eda_utils.plot_numeric_distribution(sample_df, "loan_amnt", title="Test")
    assert fig is not None


def test_plot_default_rate_by_group_runs(sample_df):
    fig, summary = eda_utils.plot_default_rate_by_group(sample_df, "grade", title="Test")
    assert fig is not None
    assert "default_rate" in summary.columns


def test_plot_correlation_heatmap_runs(sample_df):
    fig, corr = eda_utils.plot_correlation_heatmap(
        sample_df, ["loan_amnt", "int_rate", "dti", "annual_inc"], title="Test"
    )
    assert fig is not None
    assert corr.shape == (4, 4)
