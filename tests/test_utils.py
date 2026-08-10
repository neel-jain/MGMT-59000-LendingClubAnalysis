"""
test_utils.py
=============
Unit tests for src/utils.py core data-cleaning and pipeline functions.

Run with:
    pytest tests/ -v

These tests use small in-memory DataFrames (not the real or synthetic
LendingClub files) so they run quickly and deterministically, independent
of any data file being present on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, utils  # noqa: E402


def test_clean_percentage_columns_handles_percent_strings():
    df = pd.DataFrame({"int_rate": ["13.5%", "7.2%", None], "revol_util": ["10%", "20%", "30%"]})
    cleaned = utils.clean_percentage_columns(df, columns=["int_rate", "revol_util"])
    assert cleaned["int_rate"].tolist()[:2] == [13.5, 7.2]
    assert np.isnan(cleaned["int_rate"].iloc[2])
    assert cleaned["revol_util"].tolist() == [10.0, 20.0, 30.0]


def test_clean_percentage_columns_idempotent_on_numeric_input():
    df = pd.DataFrame({"int_rate": [13.5, 7.2]})
    cleaned = utils.clean_percentage_columns(df, columns=["int_rate"])
    assert cleaned["int_rate"].tolist() == [13.5, 7.2]


def test_parse_emp_length_maps_known_values():
    df = pd.DataFrame({"emp_length": ["10+ years", "< 1 year", "5 years", "unknown"]})
    result = utils.parse_emp_length(df)
    assert result["emp_length_years"].tolist()[:3] == [10.0, 0.0, 5.0]
    assert np.isnan(result["emp_length_years"].iloc[3])


def test_winsorize_columns_clips_extreme_values():
    df = pd.DataFrame({
        "annual_inc": [30000, 40000, 50000, 1000000],
        "dti": [10.0, 15.0, 20.0, 500.0],
    })
    result = utils.winsorize_columns(df, columns=["annual_inc", "dti"], iqr_multiplier=1.5)

    assert result["annual_inc"].max() < 1000000
    assert result["dti"].max() < 500.0
    assert result["annual_inc"].iloc[:-1].tolist() == [30000, 40000, 50000]
    assert result["dti"].iloc[:-1].tolist() == [10.0, 15.0, 20.0]


def test_create_target_variable_maps_and_filters():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)"],
            "loan_amnt": [1000, 2000, 3000, 4000, 5000],
        }
    )
    result = utils.create_target_variable(df)
    # "Current" and "Late" rows should be dropped
    assert len(result) == 3
    assert set(result["loan_status"]) == {"Fully Paid", "Charged Off", "Default"}
    mapping = dict(zip(result["loan_status"], result[config.TARGET_COLUMN]))
    assert mapping["Fully Paid"] == 0
    assert mapping["Charged Off"] == 1
    assert mapping["Default"] == 1


def test_create_target_variable_missing_column_raises():
    df = pd.DataFrame({"loan_amnt": [1000]})
    with pytest.raises(KeyError):
        utils.create_target_variable(df)


def test_remove_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    result = utils.remove_duplicate_rows(df)
    assert len(result) == 2


def test_filter_to_target_state():
    df = pd.DataFrame({"addr_state": ["IN", "OH", "IN"]})
    result = utils.filter_to_target_state(df)
    assert len(result) == 2
    assert set(result["addr_state"]) == {"IN"}


def test_split_data_no_leakage_and_correct_proportions():
    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feature_1": rng.normal(size=n),
            config.TARGET_COLUMN: rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(df)

    # No overlapping indices between splits (no leakage)
    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    # Proportions roughly match configured sizes
    assert abs(len(X_test) / n - config.TEST_SIZE) < 0.03
    assert abs(len(X_val) / n - config.VALIDATION_SIZE) < 0.03

    # Stratification: default rate should be similar across splits
    assert abs(y_train.mean() - y_test.mean()) < 0.08


def test_build_preprocessing_pipeline_fits_and_transforms():
    df = pd.DataFrame(
        {
            "loan_amnt": [1000, 2000, 3000, 4000],
            "int_rate": [10.0, 12.0, np.nan, 15.0],
            "installment": [30, 60, 90, 120],
            "annual_inc": [40000, 50000, 60000, 70000],
            "dti": [10, 15, 20, 25],
            "delinq_2yrs": [0, 1, 0, 2],
            "open_acc": [5, 6, 7, 8],
            "pub_rec": [0, 0, 1, 0],
            "revol_bal": [1000, 2000, 3000, 4000],
            "revol_util": [10, 20, 30, 40],
            "total_acc": [10, 12, 14, 16],
            "mort_acc": [0, 1, 2, 0],
            "pub_rec_bankruptcies": [0, 0, 0, 1],
            "emp_length_years": [1, 2, np.nan, 10],
            "term": [" 36 months", " 60 months", " 36 months", " 60 months"],
            "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT"],
            "verification_status": ["Verified", "Not Verified", "Verified", "Source Verified"],
            "purpose": ["debt_consolidation", "credit_card", "other", "debt_consolidation"],
            "initial_list_status": ["w", "f", "w", "f"],
            "application_type": ["Individual", "Individual", "Joint App", "Individual"],
            "grade": ["A", "B", "C", "G"],
        }
    )
    preprocessor = utils.build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == len(df)
    assert not np.isnan(transformed).any(), "No NaNs should remain after imputation"

    feature_names = utils.get_output_feature_names(preprocessor)
    assert len(feature_names) == transformed.shape[1]


def test_save_and_load_object_roundtrip(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "test_object.joblib"
    utils.save_object(obj, path)
    loaded = utils.load_object(path)
    assert loaded == obj


def test_load_object_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_object(tmp_path / "does_not_exist.joblib")
