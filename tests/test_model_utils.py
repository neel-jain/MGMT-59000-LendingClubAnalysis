"""
test_model_utils.py
====================
Unit tests for src/model_utils.py (Phase 3 supervised ML framework).

Uses a small synthetic dataset with real (non-random) signal so that
metric/threshold functions can be checked against sensible expected
behavior, not just "doesn't crash". Hyperparameter search tests use
tiny search spaces / low n_iter / low cv folds to keep runtime short.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for test environments
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_utils  # noqa: E402


@pytest.fixture
def signal_df():
    """
    Synthetic dataset with genuine signal: default probability increases
    with int_rate and dti, decreases with annual_inc — lets tests assert
    that ROC-AUC is meaningfully above 0.5 and that feature importance
    surfaces the right variables, not just that the code runs.
    """
    rng = np.random.default_rng(42)
    n = 600
    int_rate = rng.uniform(5, 30, n)
    dti = rng.uniform(0, 40, n)
    annual_inc = rng.lognormal(10.8, 0.4, n)
    loan_amnt = rng.uniform(1000, 30000, n)
    installment = loan_amnt / 36
    grade = rng.choice(list("ABCDE"), size=n)

    logit = (
        -3.0 + 0.12 * (int_rate - 13) + 0.05 * (dti - 18)
        - 0.00003 * (annual_inc - 60000)
    )
    prob = 1 / (1 + np.exp(-logit))
    default_flag = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": installment,
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": rng.poisson(0.3, n),
            "open_acc": rng.integers(1, 25, n),
            "pub_rec": rng.poisson(0.15, n),
            "revol_bal": rng.integers(0, 60000, n),
            "revol_util": rng.uniform(0, 100, n),
            "total_acc": rng.integers(2, 60, n),
            "mort_acc": rng.integers(0, 8, n),
            "pub_rec_bankruptcies": rng.poisson(0.08, n),
            "emp_length_years": rng.uniform(0, 10, n),
            "term": rng.choice([" 36 months", " 60 months"], size=n),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], size=n),
            "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
            "initial_list_status": rng.choice(["w", "f"], size=n),
            "application_type": rng.choice(["Individual", "Joint App"], size=n, p=[0.9, 0.1]),
            "grade": grade,
        }
    ), pd.Series(default_flag, name=config.TARGET_COLUMN)


@pytest.fixture
def train_val_test_split(signal_df):
    X, y = signal_df
    n = len(X)
    i_train, i_val = int(n * 0.6), int(n * 0.8)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    train_idx, val_idx, test_idx = idx[:i_train], idx[i_train:i_val], idx[i_val:]
    return (
        X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def test_build_logistic_regression_pipeline_has_expected_steps():
    pipeline = model_utils.build_logistic_regression_pipeline()
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]


def test_build_random_forest_pipeline_fits_and_predicts(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_build_xgboost_pipeline_fits_and_predicts(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_xgboost_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])
    metrics = model_utils.compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["matthews_corrcoef"] == pytest.approx(1.0)


def test_compute_classification_metrics_contains_all_expected_keys():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_proba = np.array([0.2, 0.7, 0.6, 0.3, 0.1])
    metrics = model_utils.compute_classification_metrics(y_true, y_pred, y_proba)
    expected_keys = {
        "accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc",
        "average_precision", "balanced_accuracy", "matthews_corrcoef", "log_loss",
        "brier_score", "calibration_error",
    }
    assert expected_keys <= set(metrics.keys())


def test_expected_calibration_error_zero_for_perfect_calibration():
    # Probabilities exactly match the empirical bin rates.
    y_true = np.array([0] * 90 + [1] * 10)  # 10% observed rate
    y_proba = np.full(100, 0.10)
    ece = model_utils.expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_positive_for_miscalibrated_model():
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.full(100, 0.90)  # very overconfident predictions
    ece = model_utils.expected_calibration_error(y_true, y_proba, n_bins=10)
    assert ece > 0.5


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------


def test_threshold_metrics_table_shape_and_monotonic_recall():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.uniform(0, 1, 200)
    table = model_utils.threshold_metrics_table(y_true, y_proba, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert len(table) == 5
    # Recall should be non-increasing as threshold rises (stricter cutoff flags fewer positives).
    recalls = table.sort_values("threshold")["recall"].to_numpy()
    assert all(recalls[i] >= recalls[i + 1] - 1e-9 for i in range(len(recalls) - 1))


def test_recommend_threshold_returns_row_with_min_cost():
    table = pd.DataFrame(
        {"threshold": [0.1, 0.3, 0.5], "expected_cost_per_loan": [0.8, 0.2, 0.5],
         "precision": [0.5, 0.6, 0.7], "recall": [0.9, 0.7, 0.5], "f1_score": [0.6, 0.65, 0.6]}
    )
    recommended = model_utils.recommend_threshold(table)
    assert recommended["threshold"] == 0.3


def test_threshold_metrics_table_expected_cost_uses_configured_weights():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.9, 0.1])  # 1 FN, 1 FP at threshold 0.5
    table = model_utils.threshold_metrics_table(
        y_true, y_proba, thresholds=[0.5], cost_fn=5.0, cost_fp=1.0,
    )
    expected = (1 * 5.0 + 1 * 1.0) / 4  # 1 FN + 1 FP over 4 loans
    assert table.iloc[0]["expected_cost_per_loan"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_logistic_regression_coefficients_shape(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    coef_df = model_utils.logistic_regression_coefficients(pipeline)
    n_features = len(model_utils.get_output_feature_names_from_pipeline(pipeline))
    assert len(coef_df) == n_features
    assert {"feature", "coefficient", "odds_ratio"} <= set(coef_df.columns)
    # odds_ratio must be exp(coefficient)
    assert np.allclose(coef_df["odds_ratio"], np.exp(coef_df["coefficient"]))


def test_logistic_regression_coefficients_surfaces_int_rate_as_important(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    coef_df = model_utils.logistic_regression_coefficients(pipeline)
    top_features = coef_df.head(5)["feature"].tolist()
    assert any("int_rate" in f for f in top_features)


def test_impurity_feature_importance_sums_reasonably(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)
    importance_df = model_utils.impurity_feature_importance(pipeline)
    assert importance_df["importance"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (importance_df["importance"] >= 0).all()


def test_xgboost_importance_all_types_has_expected_columns(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_xgboost_pipeline()
    pipeline.fit(X_train, y_train)
    importance_df = model_utils.xgboost_importance_all_types(pipeline)
    assert {"feature", "gain", "weight", "cover"} <= set(importance_df.columns)
    n_features = len(model_utils.get_output_feature_names_from_pipeline(pipeline))
    assert len(importance_df) == n_features


def test_permutation_feature_importance_shape(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)
    result = model_utils.permutation_feature_importance(pipeline, X_val, y_val, n_repeats=3)
    assert len(result) == X_val.shape[1]
    assert {"feature", "importance_mean", "importance_std"} <= set(result.columns)


# ---------------------------------------------------------------------------
# Hyperparameter search (small budgets to keep tests fast)
# ---------------------------------------------------------------------------


def test_run_grid_search_returns_fitted_search(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_logistic_regression_pipeline()
    tiny_grid = {"classifier__C": [0.1, 1.0], "classifier__solver": ["liblinear"]}
    search = model_utils.run_grid_search(pipeline, tiny_grid, X_train, y_train, cv_folds=3)
    assert hasattr(search, "best_estimator_")
    assert search.best_score_ > 0


def test_run_randomized_search_returns_fitted_search(train_val_test_split):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split
    pipeline = model_utils.build_random_forest_pipeline()
    tiny_distributions = {"classifier__n_estimators": [50, 100], "classifier__max_depth": [3, 5]}
    search = model_utils.run_randomized_search(
        pipeline, tiny_distributions, X_train, y_train, n_iter=2, cv_folds=3,
    )
    assert hasattr(search, "best_estimator_")


def test_xgboost_param_distributions_adds_scale_pos_weight():
    y_train = pd.Series([0] * 80 + [1] * 20)
    distributions = model_utils._xgboost_param_distributions(y_train)
    assert "classifier__scale_pos_weight" in distributions
    assert 1.0 in distributions["classifier__scale_pos_weight"]


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------


def test_build_model_comparison_table_ranks_by_roc_auc(train_val_test_split, tmp_path, monkeypatch):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split

    # Redirect model paths to a tmp dir so the test doesn't touch real artifacts.
    monkeypatch.setattr(config, "LOGISTIC_REGRESSION_MODEL_PATH", tmp_path / "lr.joblib")
    monkeypatch.setattr(config, "RANDOM_FOREST_MODEL_PATH", tmp_path / "rf.joblib")
    monkeypatch.setattr(config, "XGBOOST_MODEL_PATH", tmp_path / "xgb.joblib")

    results = []
    for key in ["logistic_regression", "random_forest"]:
        pipeline = model_utils.PIPELINE_BUILDERS[key]()
        pipeline.fit(X_train, y_train)
        proba_test = pipeline.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)
        test_metrics = model_utils.compute_classification_metrics(y_test, pred_test, proba_test)
        cv_fold_results = pd.DataFrame(
            {"fold": ["mean", "std"], config.CV_SCORING: [test_metrics["roc_auc"], 0.01]}
        )
        results.append(
            model_utils.ModelResult(
                model_key=key, best_estimator=pipeline, best_params={},
                cv_fold_results=cv_fold_results, train_metrics=test_metrics,
                val_metrics=test_metrics, test_metrics=test_metrics,
                training_time_sec=0.1, search_time_sec=0.2,
                prediction_time_ms_per_1000=1.0, y_proba_test=proba_test,
                threshold_table=pd.DataFrame(), recommended_threshold=0.5,
            )
        )

    table = model_utils.build_model_comparison_table(results)
    assert list(table["rank"]) == [1, 2]
    assert table.iloc[0]["roc_auc"] >= table.iloc[1]["roc_auc"]


# ---------------------------------------------------------------------------
# Plotting (smoke tests)
# ---------------------------------------------------------------------------


def test_plot_confusion_matrix_chart_runs():
    fig = model_utils.plot_confusion_matrix_chart(
        np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), title="Test",
    )
    assert fig is not None


def test_plot_roc_curve_chart_runs():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    fig = model_utils.plot_roc_curve_chart(y_true, y_proba, title="Test")
    assert fig is not None


def test_plot_threshold_analysis_chart_runs():
    table = model_utils.threshold_metrics_table(
        np.array([0, 1] * 50), np.random.default_rng(0).uniform(0, 1, 100),
    )
    fig = model_utils.plot_threshold_analysis_chart(table, recommended_threshold=0.35, title="Test")
    assert fig is not None
