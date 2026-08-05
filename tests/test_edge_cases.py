"""
test_edge_cases.py
====================
Phase 6 edge-case test suite.

Verifies graceful handling of unusual/invalid inputs and missing/
corrupted artifacts, per the Phase 6 brief:

    - Missing values
    - Empty datasets
    - Invalid loan grades
    - Negative income
    - Extremely large loan amounts
    - Very high DTI
    - Unexpected categories
    - Corrupted serialized models
    - Missing files
    - Incorrect user inputs

The goal of every test here is NOT "does this raise an exception" in
isolation -- it's "does the system fail in a controlled, documented way"
(a clear exception type, a sentinel return value like `None`/empty
DataFrame, or a valid-but-extreme output), never a silent wrong answer
or an unhandled crash with a confusing traceback.

Run with:
    pytest tests/test_edge_cases.py -v
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, utils  # noqa: E402
from src.risk_scoring import RiskScoringEngine  # noqa: E402
from src.explainability import ExplainabilityEngine  # noqa: E402
from src.segmentation_engine import SegmentationEngine  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return utils.load_splits()


@pytest.fixture(scope="module")
def risk_engine():
    return RiskScoringEngine()


@pytest.fixture(scope="module")
def base_borrower(splits):
    """One valid, realistic borrower row used as the template for edge-case mutations below."""
    _, _, X_test, _, _, _ = splits
    return X_test.iloc[[0]].copy()


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_predict_with_missing_numeric_value_does_not_crash(risk_engine, base_borrower):
    """A missing DTI value should be imputed by the pipeline (median strategy), not raise."""
    borrower = base_borrower.copy()
    borrower["dti"] = np.nan
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_all_numeric_missing_still_returns_valid_probability(risk_engine, base_borrower):
    """Every numeric field missing at once should still impute to a valid (if extreme) probability."""
    borrower = base_borrower.copy()
    for col in config.NUMERIC_FEATURES:
        borrower[col] = np.nan
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_missing_categorical_value_does_not_crash(risk_engine, base_borrower):
    """A missing purpose value should be imputed (most-frequent strategy), not raise."""
    borrower = base_borrower.copy()
    borrower["purpose"] = np.nan
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


# ---------------------------------------------------------------------------
# Empty datasets
# ---------------------------------------------------------------------------


def test_predict_probability_on_empty_dataframe_returns_empty_array(risk_engine, base_borrower):
    empty = base_borrower.iloc[0:0]
    proba = risk_engine.predict_probability(empty)
    assert len(proba) == 0


def test_generate_batch_summary_on_empty_dataframe_returns_empty_dataframe(risk_engine, base_borrower):
    empty = base_borrower.iloc[0:0]
    result = risk_engine.generate_batch_summary(empty)
    assert result.empty


def test_generate_prediction_summary_on_empty_dataframe_raises_clear_error(risk_engine, base_borrower):
    """Single-borrower methods must reject a zero-row input with a clear ValueError, not a cryptic pipeline error."""
    empty = base_borrower.iloc[0:0]
    with pytest.raises(ValueError, match="exactly one"):
        risk_engine.generate_prediction_summary(empty)


def test_fairness_report_on_empty_dataframe_returns_empty_dataframe():
    from src import interpretation_utils as iu

    empty_X = pd.DataFrame(columns=["home_ownership"])
    empty_y = pd.Series([], dtype=int)
    empty_proba = np.array([])
    result = iu.fairness_report(empty_X, empty_y, empty_proba, group_columns=["home_ownership"])
    assert result.empty


# ---------------------------------------------------------------------------
# Invalid loan grades / unexpected categories
# ---------------------------------------------------------------------------


def test_predict_with_invalid_loan_grade_handled_via_unknown_encoding(risk_engine, base_borrower):
    """
    An out-of-vocabulary grade (e.g. 'Z') is handled by the ordinal
    encoder's configured `unknown_value=-1` fallback (see
    utils.build_preprocessing_pipeline) rather than raising -- the
    prediction may be extreme/unreliable for such a borrower, but the
    system must not crash.
    """
    borrower = base_borrower.copy()
    borrower["grade"] = "Z"  # not in config.ORDINAL_CATEGORY_ORDER
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_unexpected_categorical_value_handled_gracefully(risk_engine, base_borrower):
    """An unrecognized home_ownership category is handled by OneHotEncoder(handle_unknown='ignore')."""
    borrower = base_borrower.copy()
    borrower["home_ownership"] = "SPACESHIP"
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_recommend_loan_grade_never_raises_across_full_probability_range(risk_engine):
    """Loan-grade assignment must return a valid letter grade across the entire [0, 1] probability range, including the exact boundaries."""
    for p in [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0]:
        grade = risk_engine.recommend_loan_grade(p)
        assert grade in {"A", "B", "C", "D", "E", "F", "G"}


# ---------------------------------------------------------------------------
# Negative income / extreme values
# ---------------------------------------------------------------------------


def test_predict_with_negative_income_does_not_crash(risk_engine, base_borrower):
    """
    Negative income is nonsensical input but must not crash the pipeline
    -- StandardScaler and the model will simply treat it as an extreme
    value. Input VALIDATION (rejecting a negative income before scoring)
    is a UI-layer concern (see app/app_pages/borrower_risk_prediction.py's
    `min_value=0.0` on the income field) -- this test documents the
    engine's own defensive floor.
    """
    borrower = base_borrower.copy()
    borrower["annual_inc"] = -50000.0
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_extremely_large_loan_amount_does_not_crash(risk_engine, base_borrower):
    borrower = base_borrower.copy()
    borrower["loan_amnt"] = 10_000_000.0  # far outside any realistic LendingClub loan size
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_very_high_dti_does_not_crash(risk_engine, base_borrower):
    borrower = base_borrower.copy()
    borrower["dti"] = 500.0  # far beyond any realistic DTI
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0
    # A borrower with an extreme DTI should score no lower risk than a
    # realistic baseline -- a soft sanity check that the model direction
    # is at least not obviously inverted for this feature.
    realistic = base_borrower.copy()
    realistic["dti"] = 15.0
    realistic_proba = risk_engine.predict_probability(realistic)
    assert proba[0] >= realistic_proba[0] - 0.35  # generous tolerance; not a strict monotonicity claim


def test_predict_with_zero_loan_amount_does_not_crash(risk_engine, base_borrower):
    borrower = base_borrower.copy()
    borrower["loan_amnt"] = 0.0
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


# ---------------------------------------------------------------------------
# Missing / corrupted serialized artifacts
# ---------------------------------------------------------------------------


def test_risk_scoring_engine_raises_clear_error_for_unknown_model_key():
    with pytest.raises(ValueError, match="Unknown model_key"):
        RiskScoringEngine(model_key="not_a_real_model")


def test_load_object_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_object(tmp_path / "does_not_exist.joblib")


def test_load_object_corrupted_file_raises_readable_error(tmp_path):
    """A corrupted (non-joblib) file must raise an exception from joblib's own loader, not silently return garbage."""
    corrupted_path = tmp_path / "corrupted_model.joblib"
    corrupted_path.write_bytes(b"this is not a valid joblib/pickle file at all")
    with pytest.raises(Exception):  # noqa: B017 -- joblib may raise several different exception types depending on corruption
        utils.load_object(corrupted_path)


def test_risk_scoring_engine_missing_model_file_raises_file_not_found(tmp_path, monkeypatch):
    """If a model file is missing on disk, constructing RiskScoringEngine for it must raise FileNotFoundError, not a cryptic downstream error."""
    from src import risk_scoring

    monkeypatch.setitem(risk_scoring.MODEL_PATHS, "xgboost", tmp_path / "missing_model.joblib")
    with pytest.raises(FileNotFoundError):
        RiskScoringEngine(model_key="xgboost")


def test_load_dataframe_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_dataframe(tmp_path / "does_not_exist.csv")


def test_load_raw_data_empty_file_raises_value_error(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError):
        utils.load_raw_data(empty_csv)


# ---------------------------------------------------------------------------
# Incorrect / malformed user inputs
# ---------------------------------------------------------------------------


def test_predict_with_missing_required_column_raises_clear_error(risk_engine, base_borrower):
    """Dropping a column the pipeline expects must raise a clear error (from the ColumnTransformer), not silently mis-score."""
    borrower = base_borrower.drop(columns=["dti"])
    with pytest.raises(Exception):  # noqa: B017 -- exact exception type is sklearn's, not ours to pin down
        risk_engine.predict_probability(borrower)


def test_predict_with_extra_unexpected_column_is_ignored_gracefully(risk_engine, base_borrower):
    """An extra column the pipeline doesn't expect should be silently ignored (ColumnTransformer's remainder='drop'), not raise."""
    borrower = base_borrower.copy()
    borrower["some_extra_column_not_in_schema"] = "unexpected_value"
    proba = risk_engine.predict_probability(borrower)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_with_multiple_rows_rejected_by_single_borrower_methods(risk_engine, base_borrower):
    two_borrowers = pd.concat([base_borrower, base_borrower], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        risk_engine.generate_prediction_summary(two_borrowers)


def test_explainability_engine_rejects_multi_row_local_explanation(base_borrower):
    engine = ExplainabilityEngine()
    two_borrowers = pd.concat([base_borrower, base_borrower], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        engine.explain_prediction(two_borrowers)


def test_segmentation_engine_methods_raise_before_fit():
    """Calling any SegmentationEngine method before .fit() must raise a clear RuntimeError, not an AttributeError deep in engine internals."""
    engine = SegmentationEngine()
    with pytest.raises(RuntimeError, match="not been fit"):
        engine.compare_segments()
    with pytest.raises(RuntimeError, match="not been fit"):
        engine.describe_segment(0)
