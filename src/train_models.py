"""
train_models.py
================
Orchestration entry point for the MGMT 590 LendingClub Loan Default Risk
capstone project.

PHASE 1 SCOPE (unchanged since Phase 1)
--------------------------------------------
This phase implements and exercises the full DATA pipeline:
    raw ingestion -> validation -> cleaning -> preprocessing pipeline
    construction -> leakage-safe train/val/test split -> serialization.

Running ``run_phase1_pipeline()`` (or ``python -m src.train_models``) will:
    1. Ensure the project directory structure exists.
    2. Load the raw Indiana LendingClub extract.
    3. Validate it and log a data-quality report.
    4. Clean it (dedupe, percentage parsing, employment-length parsing,
       binary target construction, column pruning).
    5. Split it into train/validation/test sets (stratified, leak-safe).
    6. Fit the preprocessing ColumnTransformer on the TRAINING split only.
    7. Serialize the fitted preprocessor and save all split artifacts.
    8. Save the full cleaned (pre-split) dataset for EDA/reporting use.

PHASE 3 SCOPE (implemented in this file as of this phase)
----------------------------------------------------------------------------
``train_logistic_regression``, ``train_random_forest``, ``train_xgboost``,
and ``evaluate_model`` (previously ``NotImplementedError`` stubs) now
delegate to the reusable machine-learning framework in
``src/model_utils.py``: hyperparameter search with Stratified K-Fold
cross-validation, threshold optimization, and feature-importance
extraction for each of the three models. ``run_phase3_pipeline()``
orchestrates all three end-to-end, builds the executive model-comparison
table, and serializes every artifact (models, metrics, CV results,
feature-importance tables, probability predictions).

This module is designed to be run either as a script or imported:

    from src.train_models import run_phase1_pipeline, run_phase3_pipeline
    phase1 = run_phase1_pipeline()
    phase3 = run_phase3_pipeline(phase1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer

from src import config, model_utils, utils

logger = utils.get_logger(__name__)


@dataclass
class Phase1Artifacts:
    """
    Container for every artifact produced by the Phase 1 data pipeline,
    returned by ``run_phase1_pipeline()`` so callers (notebook, Phase 3
    script, tests) can access in-memory results without re-reading from
    disk if they don't want to.

    Attributes
    ----------
    raw_df : pd.DataFrame
        Unmodified raw dataset as loaded from disk.
    validation_report : dict
        Output of ``utils.validate_dataset`` run on ``raw_df``.
    cleaned_df : pd.DataFrame
        Output of ``utils.clean_dataset`` (post cleaning, pre-split).
    X_train, X_val, X_test : pd.DataFrame
        Feature splits.
    y_train, y_val, y_test : pd.Series
        Target splits.
    preprocessor : ColumnTransformer
        Preprocessing pipeline FIT on X_train only.
    """

    raw_df: pd.DataFrame
    validation_report: Dict[str, Any]
    cleaned_df: pd.DataFrame
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def run_phase1_pipeline(persist: bool = True) -> Phase1Artifacts:
    """
    Execute the complete Phase 1 data pipeline end-to-end.

    Parameters
    ----------
    persist : bool
        If True (default), write the cleaned dataset, split artifacts,
        and fitted preprocessor to disk (config.PROCESSED_DATA_DIR,
        config.SPLITS_DIR, config.PIPELINES_DIR). Set False for a
        dry-run / unit-test invocation that keeps everything in memory
        only.

    Returns
    -------
    Phase1Artifacts
        Every intermediate and final artifact from the pipeline.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE START")
    logger.info("=" * 70)

    utils.ensure_directories()

    # 1. Ingest
    raw_df = utils.load_raw_data()

    # 2. Validate
    validation_report = utils.validate_dataset(raw_df)

    # 3. Clean
    cleaned_df = utils.clean_dataset(raw_df)
    if persist:
        utils.save_dataframe(cleaned_df, config.CLEANED_DATA_PATH)

    # 4. Split (leakage-safe: test/val carved out before any fitting)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(cleaned_df)
    if persist:
        utils.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    # 5. Build + fit preprocessing pipeline on TRAINING data only
    preprocessor = utils.build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    logger.info(
        "Preprocessor fit complete. Output feature count: %d",
        len(utils.get_output_feature_names(preprocessor)),
    )
    if persist:
        utils.save_object(preprocessor, config.PREPROCESSOR_PATH)

    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE COMPLETE")
    logger.info("=" * 70)

    return Phase1Artifacts(
        raw_df=raw_df,
        validation_report=validation_report,
        cleaned_df=cleaned_df,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# PHASE 3 -- SUPERVISED MODEL TRAINING (implemented)
# ---------------------------------------------------------------------------
# These four functions were NotImplementedError stubs through Phase 1-2.
# They now delegate to src/model_utils.py, which holds the actual
# hyperparameter-search / cross-validation / metrics / feature-importance
# logic, so this file stays a thin orchestration layer consistent with
# its Phase 1 role.
# ---------------------------------------------------------------------------


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fit a Logistic Regression pipeline via exhaustive GridSearchCV over
    `config.LOGISTIC_REGRESSION_PARAM_GRID` with Stratified K-Fold CV.

    See `model_utils.build_logistic_regression_pipeline` for the full
    algorithm rationale (why chosen, advantages/disadvantages, business
    tradeoffs, computational complexity).

    Parameters
    ----------
    X_train, y_train : training data.

    Returns
    -------
    GridSearchCV
        Fitted search object; `.best_estimator_` is the tuned Pipeline
        (preprocessor + classifier), already refit on the full training set.
    """
    pipeline = model_utils.build_logistic_regression_pipeline()
    return model_utils.run_grid_search(
        pipeline, config.LOGISTIC_REGRESSION_PARAM_GRID, X_train, y_train,
    )


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fit a Random Forest pipeline via RandomizedSearchCV over
    `config.RANDOM_FOREST_PARAM_DISTRIBUTIONS` with Stratified K-Fold CV.

    See `model_utils.build_random_forest_pipeline` for the full algorithm
    rationale.

    Parameters
    ----------
    X_train, y_train : training data.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object.
    """
    pipeline = model_utils.build_random_forest_pipeline()
    return model_utils.run_randomized_search(
        pipeline, config.RANDOM_FOREST_PARAM_DISTRIBUTIONS, X_train, y_train,
        n_iter=config.RANDOM_FOREST_N_ITER,
    )


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fit an XGBoost pipeline via RandomizedSearchCV over
    `config.XGBOOST_PARAM_DISTRIBUTIONS` (extended at runtime with a
    data-dependent `scale_pos_weight` candidate -- see
    `model_utils._xgboost_param_distributions`) with Stratified K-Fold CV.

    See `model_utils.build_xgboost_pipeline` for the full algorithm
    rationale.

    Parameters
    ----------
    X_train, y_train : training data.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object.
    """
    pipeline = model_utils.build_xgboost_pipeline()
    distributions = model_utils._xgboost_param_distributions(y_train)
    return model_utils.run_randomized_search(
        pipeline, distributions, X_train, y_train, n_iter=config.XGBOOST_N_ITER,
    )


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Compute the full Phase 3 metric suite (accuracy, precision, recall,
    specificity, F1, ROC-AUC, average precision, balanced accuracy, MCC,
    log loss, Brier score, calibration error) for a fitted model /
    Pipeline on a given feature/target set, at the default 0.50 threshold.

    For per-model threshold optimization, cross-validation fold detail,
    and feature importance, use `model_utils.train_and_evaluate_model`
    instead -- this function exists for quick, single-threshold scoring
    (e.g. ad hoc checks, tests).

    Parameters
    ----------
    model : fitted Pipeline (or any estimator exposing predict/predict_proba)
    X, y : evaluation data.

    Returns
    -------
    dict
        Metric name -> value.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return model_utils.compute_classification_metrics(y, y_pred, y_proba)


@dataclass
class Phase3Artifacts:
    """
    Container for everything Phase 3 produces, returned by
    `run_phase3_pipeline()` so the notebook, tests, or a later phase can
    access in-memory results without re-reading from disk.

    Attributes
    ----------
    results : dict[str, model_utils.ModelResult]
        Keyed by model_key ("logistic_regression", "random_forest",
        "xgboost").
    comparison_table : pd.DataFrame
        Executive model-comparison table (see
        `model_utils.build_model_comparison_table`).
    """

    results: Dict[str, "model_utils.ModelResult"]
    comparison_table: pd.DataFrame


def run_phase3_pipeline(
    phase1_artifacts: "Phase1Artifacts", persist: bool = True,
) -> Phase3Artifacts:
    """
    Execute the complete Phase 3 modeling pipeline end-to-end: train and
    evaluate all three models (Logistic Regression, Random Forest,
    XGBoost), build the executive comparison table, and (optionally)
    serialize every artifact via joblib.

    Parameters
    ----------
    phase1_artifacts : Phase1Artifacts
        Output of `run_phase1_pipeline()` -- supplies the leakage-safe
        train/validation/test splits this phase trains and evaluates on.
    persist : bool
        If True (default), serialize the three fitted models
        (`config.LOGISTIC_REGRESSION_MODEL_PATH`, etc.) and the shared
        evaluation artifacts (`config.EVALUATION_METRICS_PATH`,
        `config.CV_RESULTS_PATH`, `config.FEATURE_IMPORTANCE_PATH`,
        `config.PROBABILITY_PREDICTIONS_PATH`,
        `config.THRESHOLD_ANALYSIS_PATH`,
        `config.MODEL_COMPARISON_TABLE_PATH`) to disk.

    Returns
    -------
    Phase3Artifacts
    """
    logger.info("=" * 70)
    logger.info("PHASE 3 PIPELINE START")
    logger.info("=" * 70)

    utils.ensure_directories()

    model_paths = {
        "logistic_regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
        "random_forest": config.RANDOM_FOREST_MODEL_PATH,
        "xgboost": config.XGBOOST_MODEL_PATH,
    }

    results: Dict[str, model_utils.ModelResult] = {}
    for model_key in ("logistic_regression", "random_forest", "xgboost"):
        result = model_utils.train_and_evaluate_model(
            model_key,
            phase1_artifacts.X_train, phase1_artifacts.y_train,
            phase1_artifacts.X_val, phase1_artifacts.y_val,
            phase1_artifacts.X_test, phase1_artifacts.y_test,
        )
        results[model_key] = result
        if persist:
            utils.save_object(result.best_estimator, model_paths[model_key])

    comparison_table = model_utils.build_model_comparison_table(list(results.values()))

    if persist:
        utils.save_object(
            {k: r.test_metrics for k, r in results.items()}, config.EVALUATION_METRICS_PATH,
        )
        utils.save_object(
            {k: r.cv_fold_results for k, r in results.items()}, config.CV_RESULTS_PATH,
        )
        utils.save_object(
            {k: r.feature_importance for k, r in results.items()}, config.FEATURE_IMPORTANCE_PATH,
        )
        utils.save_object(
            {k: r.y_proba_test for k, r in results.items()}, config.PROBABILITY_PREDICTIONS_PATH,
        )
        utils.save_object(
            {k: r.threshold_table for k, r in results.items()}, config.THRESHOLD_ANALYSIS_PATH,
        )
        comparison_table.to_csv(config.MODEL_COMPARISON_TABLE_PATH, index=False)
        logger.info("All Phase 3 artifacts serialized.")

    logger.info("=" * 70)
    logger.info("PHASE 3 PIPELINE COMPLETE")
    logger.info("=" * 70)

    return Phase3Artifacts(results=results, comparison_table=comparison_table)


def main() -> None:
    """
    Script entry point: run the Phase 1 pipeline, then the Phase 3
    modeling pipeline, and print a concise summary of both.
    Intended usage: ``python -m src.train_models``.
    """
    phase1 = run_phase1_pipeline(persist=True)

    print("\n--- PHASE 1 SUMMARY ---")
    print(f"Raw rows loaded:        {len(phase1.raw_df):,}")
    print(f"Rows after cleaning:    {len(phase1.cleaned_df):,}")
    print(f"Train / Val / Test:     {len(phase1.X_train):,} / "
          f"{len(phase1.X_val):,} / {len(phase1.X_test):,}")
    print(f"Default rate (train):   {phase1.y_train.mean():.3%}")
    print(f"Preprocessed feature count: "
          f"{len(utils.get_output_feature_names(phase1.preprocessor)):,}")

    phase3 = run_phase3_pipeline(phase1, persist=True)

    print("\n--- PHASE 3 SUMMARY ---")
    print(phase3.comparison_table[["rank", "model", "roc_auc", "f1_score", "recall"]].to_string(index=False))
    print(f"\nModel artifacts saved to:   {config.MODELS_DIR}")
    print(f"Evaluation reports saved to: {config.REPORTS_DIR}")
    print("Phase 4 (SHAP explanations) and clustering are intentionally not yet implemented.")


if __name__ == "__main__":
    main()
