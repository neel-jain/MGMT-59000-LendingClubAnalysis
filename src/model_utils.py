"""
model_utils.py
===============
Phase 3 reusable module: supervised machine learning framework for loan
default prediction (Logistic Regression, Random Forest, XGBoost).

This module is ADDITIVE to Phases 1-2 — it does not modify `config.py`,
`utils.py`, or `eda_utils.py`. It imports from `config` and `utils` (for
paths, constants, and the Phase 1 preprocessing pipeline builder) and
from `eda_utils` (for the shared visual style), and builds the modeling
layer on top.

Organized into sections:
    1. Pipeline construction (preprocessing + estimator)
    2. Hyperparameter search (Grid / Randomized)
    3. Cross-validation reporting
    4. Classification metrics (incl. calibration error)
    5. Threshold optimization
    6. Feature importance (coefficients/odds ratios, impurity,
       permutation, XGBoost gain/weight/cover)
    7. Visualizations (confusion matrix, ROC, PR, calibration, learning
       curve, validation curve, importance/coefficient plots,
       probability distribution, threshold analysis)
    8. Model comparison table
    9. End-to-end orchestration (train_and_evaluate_model)

Design principle: every function takes a fitted or unfitted
scikit-learn `Pipeline` (preprocessing + classifier bundled together, per
Phase 3's leakage-prevention requirement) so callers never touch a raw
estimator or a raw preprocessor separately. All plotting functions return
the created `matplotlib.figure.Figure` and follow eda_utils' visual
conventions (title/subtitle, labeled axes, consistent default/paid
colors) for a single consistent look across the whole notebook series.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from src import config, utils
from src.eda_utils import (
    COLOR_DEFAULT,
    COLOR_PAID,
    FIGSIZE_STANDARD,
    FIGSIZE_WIDE,
    PALETTE_DIVERGING,
    _apply_titles,
)

logger = utils.get_logger(__name__)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


# ---------------------------------------------------------------------------
# 1. PIPELINE CONSTRUCTION
# ---------------------------------------------------------------------------


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """
    Wrap any scikit-learn-compatible classifier with the Phase 1
    preprocessing ColumnTransformer to form a single, leakage-safe
    scikit-learn `Pipeline`.

    Design decision: rather than reusing the single preprocessor object
    already fit-and-serialized in Phase 1
    (`config.PREPROCESSOR_PATH`, fit once on the full X_train), Phase 3
    builds a FRESH, unfitted preprocessor per model via
    `utils.build_preprocessing_pipeline()` and lets it live *inside* this
    `Pipeline`. This is deliberate: cross-validation and hyperparameter
    search repeatedly refit on different subsets of the training data,
    and only a preprocessor embedded in the `Pipeline` gets refit
    correctly on each fold's training portion — reusing one preprocessor
    fit on the entire X_train up front would leak validation-fold
    statistics (imputation medians, scaler mean/std, encoder categories)
    into that fold's training step. The Phase 1 serialized preprocessor
    remains valid and correct for its original purpose (one-off transforms
    in Phase 2's EDA); Phase 3 reuses the same *construction logic*
    (`utils.build_preprocessing_pipeline`), not the same *fitted object*.

    Parameters
    ----------
    estimator : BaseEstimator
        An unfitted scikit-learn-compatible classifier.

    Returns
    -------
    Pipeline
        Unfitted `Pipeline([("preprocessor", ...), ("classifier", ...)])`.
    """
    preprocessor = utils.build_preprocessing_pipeline()
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build the (unfitted) Logistic Regression pipeline.

    Why this algorithm: Logistic Regression is included as the highly
    interpretable baseline. Its coefficients translate directly into
    odds ratios that a credit-risk committee can read and defend, it
    trains in milliseconds, and it gives every later model a bar to beat.

    Advantages: transparent, well-calibrated by construction (linear log-
    odds), cheap to train/predict, robust with modest data.
    Disadvantages: assumes a linear (in log-odds) relationship between
    features and outcome and no interaction effects unless engineered
    manually; cannot capture nonlinear risk patterns Random Forest/
    XGBoost can.
    Business tradeoff: sacrifices some predictive power for maximum
    auditability — often preferred for the parts of a credit decision
    that must be explained to a regulator or a declined applicant.
    Computational complexity: O(n_features x n_samples) per iteration of
    the solver; trivial at this dataset's scale (thousands of rows).
    """
    estimator = LogisticRegression(
        max_iter=2000, random_state=config.RANDOM_STATE,
    )
    return build_model_pipeline(estimator)


def build_random_forest_pipeline() -> Pipeline:
    """
    Build the (unfitted) Random Forest pipeline.

    Why this algorithm: an ensemble of decision trees that captures
    nonlinear relationships and feature interactions (e.g. "high DTI
    matters more when income is also low") without the analyst having to
    hand-engineer them, while still offering intuitive impurity-based and
    permutation feature importances.

    Advantages: handles nonlinearity/interactions natively, robust to
    outliers and unscaled features, low risk of severe overfitting thanks
    to bagging, minimal preprocessing requirements.
    Disadvantages: individual trees are uninterpretable (though the
    ensemble offers importance measures, not "why this one borrower" —
    that gap is exactly what Phase 4's SHAP analysis will close); larger
    memory footprint than a single linear model or a single boosted
    sequence; impurity importance is biased toward high-cardinality
    numeric features.
    Business tradeoff: meaningfully better discrimination than Logistic
    Regression at the cost of some interpretability and larger serialized
    model size.
    Computational complexity: O(n_trees x n_samples x log(n_samples) x
    n_features) for training; prediction is O(n_trees x tree_depth), fast
    but slower than a single logistic regression scoring pass.
    """
    estimator = RandomForestClassifier(
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return build_model_pipeline(estimator)


def build_xgboost_pipeline() -> Pipeline:
    """
    Build the (unfitted) XGBoost pipeline.

    Why this algorithm: gradient-boosted trees are trained sequentially,
    each new tree correcting the previous ensemble's residual errors,
    which typically yields the strongest raw discriminative performance
    among tree-based methods on structured/tabular data such as this —
    making it the candidate production model.

    Advantages: state-of-the-art tabular performance, built-in
    regularization (L1/L2, min_child_weight, gamma) that helps control
    overfitting, native handling of missing values, fast training via
    histogram-based split finding, multiple importance views (gain/
    weight/cover).
    Disadvantages: the largest hyperparameter surface of the three
    (highest tuning cost), least directly interpretable without an
    additional layer (SHAP, in Phase 4), can overfit small/noisy datasets
    if not regularized carefully.
    Business tradeoff: typically the best accuracy/ROC-AUC of the three,
    at the cost of being a "black box" without supplementary
    explainability tooling and a heavier tuning/maintenance burden.
    Computational complexity: O(n_trees x n_samples x n_features x
    log(n_samples)) for training (similar order to Random Forest, but
    sequential rather than parallel across trees, though each tree's
    construction is itself parallelized internally).
    """
    estimator = XGBClassifier(
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    return build_model_pipeline(estimator)


PIPELINE_BUILDERS: Dict[str, Any] = {
    "logistic_regression": build_logistic_regression_pipeline,
    "random_forest": build_random_forest_pipeline,
    "xgboost": build_xgboost_pipeline,
}


# ---------------------------------------------------------------------------
# 2. HYPERPARAMETER SEARCH
# ---------------------------------------------------------------------------


def _xgboost_param_distributions(y_train: pd.Series) -> Dict[str, list]:
    """
    Extend `config.XGBOOST_PARAM_DISTRIBUTIONS` with a data-dependent
    `scale_pos_weight` candidate set. `scale_pos_weight` rebalances the
    default (minority) class, and the "textbook-neutral" value
    (negative_count / positive_count) can only be computed once y_train
    is known, so it cannot live as a fixed constant in config.py.

    Parameters
    ----------
    y_train : pd.Series

    Returns
    -------
    dict
        Copy of config.XGBOOST_PARAM_DISTRIBUTIONS with
        'classifier__scale_pos_weight' added.
    """
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    neutral_ratio = round(neg / pos, 2) if pos > 0 else 1.0
    distributions = dict(config.XGBOOST_PARAM_DISTRIBUTIONS)
    distributions["classifier__scale_pos_weight"] = [1.0, neutral_ratio]
    return distributions


def run_grid_search(
    pipeline: Pipeline,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
) -> GridSearchCV:
    """
    Exhaustive hyperparameter search via `GridSearchCV` with Stratified
    K-Fold cross-validation. Used for Logistic Regression, whose
    parameter space (a handful of C values x 2 penalties) is small enough
    to search exhaustively.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted preprocessing + classifier pipeline.
    param_grid : dict
        Grid of `classifier__*`-prefixed hyperparameters to search.
    X_train, y_train : training data (features/target).
    cv_folds : int
        Number of stratified folds.
    scoring : str
        scikit-learn scorer name used to rank candidates.

    Returns
    -------
    GridSearchCV
        Fitted search object (`.best_estimator_` already refit on the
        full X_train/y_train; `.refit_time_` gives that refit's wall time).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        pipeline, param_grid=param_grid, scoring=scoring, cv=cv,
        n_jobs=-1, refit=True, return_train_score=True,
    )
    logger.info("Starting GridSearchCV over %d candidate combinations.",
                int(np.prod([len(v) for v in param_grid.values()])))
    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info(
        "GridSearchCV complete in %.2fs. Best %s=%.4f. Best params: %s",
        elapsed, scoring, search.best_score_, search.best_params_,
    )
    return search


def run_randomized_search(
    pipeline: Pipeline,
    param_distributions: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv_folds: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
) -> RandomizedSearchCV:
    """
    Randomized hyperparameter search via `RandomizedSearchCV` with
    Stratified K-Fold cross-validation. Used for Random Forest and
    XGBoost, whose combinatorial parameter spaces are too large to search
    exhaustively within a reasonable compute budget; randomized search
    samples `n_iter` combinations, which explores a wide space far more
    efficiently per unit of compute than a grid restricted to the same
    budget (Bergstra & Bengio, 2012).

    Parameters
    ----------
    pipeline : Pipeline
    param_distributions : dict
    X_train, y_train : training data.
    n_iter : int
        Number of randomly sampled hyperparameter combinations to try.
    cv_folds : int
    scoring : str

    Returns
    -------
    RandomizedSearchCV
        Fitted search object.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_distributions, n_iter=n_iter,
        scoring=scoring, cv=cv, n_jobs=-1, refit=True, return_train_score=True,
        random_state=config.RANDOM_STATE,
    )
    logger.info("Starting RandomizedSearchCV with n_iter=%d.", n_iter)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info(
        "RandomizedSearchCV complete in %.2fs. Best %s=%.4f. Best params: %s",
        elapsed, scoring, search.best_score_, search.best_params_,
    )
    return search


def tune_model(
    model_key: str, X_train: pd.DataFrame, y_train: pd.Series,
) -> Tuple[Any, float]:
    """
    Dispatch to the appropriate search strategy (grid vs. randomized) for
    a given model key, using the search spaces and settings centralized
    in config.py.

    Parameters
    ----------
    model_key : str
        One of "logistic_regression", "random_forest", "xgboost".
    X_train, y_train : training data.

    Returns
    -------
    (search_result, search_wall_time_seconds)
    """
    if model_key not in PIPELINE_BUILDERS:
        raise ValueError(f"Unknown model_key '{model_key}'. Expected one of {list(PIPELINE_BUILDERS)}.")

    pipeline = PIPELINE_BUILDERS[model_key]()
    start = time.perf_counter()

    if model_key == "logistic_regression":
        search = run_grid_search(pipeline, config.LOGISTIC_REGRESSION_PARAM_GRID, X_train, y_train)
    elif model_key == "random_forest":
        search = run_randomized_search(
            pipeline, config.RANDOM_FOREST_PARAM_DISTRIBUTIONS, X_train, y_train,
            n_iter=config.RANDOM_FOREST_N_ITER,
        )
    elif model_key == "xgboost":
        distributions = _xgboost_param_distributions(y_train)
        search = run_randomized_search(
            pipeline, distributions, X_train, y_train, n_iter=config.XGBOOST_N_ITER,
        )
    else:  # pragma: no cover - guarded above
        raise ValueError(model_key)

    elapsed = time.perf_counter() - start
    return search, elapsed


# ---------------------------------------------------------------------------
# 3. CROSS-VALIDATION REPORTING
# ---------------------------------------------------------------------------


def extract_cv_fold_results(search: Any, scoring: str = config.CV_SCORING) -> pd.DataFrame:
    """
    Extract fold-by-fold validation scores for the BEST hyperparameter
    combination found by a fitted GridSearchCV/RandomizedSearchCV, plus
    the mean and standard deviation across folds.

    Parameters
    ----------
    search : GridSearchCV | RandomizedSearchCV
        Fitted search object.
    scoring : str
        Metric name (used only for the column label).

    Returns
    -------
    pd.DataFrame
        One row per fold plus a final "mean"/"std" summary row.
    """
    best_idx = search.best_index_
    cv_results = search.cv_results_
    fold_cols = sorted(
        [c for c in cv_results if c.startswith("split") and c.endswith("_test_score")]
    )
    fold_scores = [cv_results[c][best_idx] for c in fold_cols]

    rows = [{"fold": i + 1, scoring: score} for i, score in enumerate(fold_scores)]
    rows.append({"fold": "mean", scoring: np.mean(fold_scores)})
    rows.append({"fold": "std", scoring: np.std(fold_scores)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION METRICS
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = config.CALIBRATION_BINS,
) -> float:
    """
    Expected Calibration Error (ECE): the weighted average gap between
    predicted probability and observed default rate across probability
    bins. A well-calibrated model's ECE is close to 0, meaning "when the
    model says 30% default risk, roughly 30% of those loans actually
    default" — critical for a lending model whose output probabilities
    may directly inform pricing or approval-rate targets, not just
    rank-ordering.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_proba : array-like of predicted probabilities of the positive class
    n_bins : int

    Returns
    -------
    float
        ECE in [0, 1] (lower is better).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba > lo) & (y_proba <= hi) if lo > 0 else (y_proba >= lo) & (y_proba <= hi)
        if mask.sum() == 0:
            continue
        bin_confidence = y_proba[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full Phase 3 metric suite for one set of predictions at
    a given decision threshold (metrics that don't depend on a threshold
    — ROC-AUC, log loss, Brier score, calibration error — are threshold-
    independent by construction and computed from `y_proba` directly).

    Business rationale for each metric (why we don't stop at accuracy):
    - Accuracy: intuitive but MISLEADING here — with ~20-25% default
      rate, predicting "no default" for every loan already scores ~75-80%.
    - Precision: of loans we FLAG as risky, how many really default? High
      precision means a "decline" policy built on this model wastes few
      good customers.
    - Recall (sensitivity): of loans that actually default, how many did
      we catch? Directly drives how much bad debt is avoided.
    - Specificity: of loans that are actually repaid, how many did we
      correctly clear? Low specificity means good borrowers get declined
      unnecessarily, costing revenue and customer goodwill.
    - F1: harmonic mean of precision/recall — a single number for
      comparing models when both error types matter and the class split
      is imbalanced (unlike accuracy).
    - ROC-AUC: threshold-independent ranking quality — "if you show the
      model one defaulter and one non-defaulter, how often does it score
      the defaulter higher?" Directly relevant since the business will
      choose its own operating threshold after training.
    - Balanced accuracy: average of recall and specificity — a
      class-imbalance-robust alternative to plain accuracy.
    - Matthews Correlation Coefficient (MCC): a single balanced summary
      of the full confusion matrix, considered one of the most reliable
      metrics for imbalanced binary classification (robust even when
      class sizes are very different).
    - Log loss: penalizes confident-but-wrong probability estimates
      heavily — relevant if predicted probabilities feed into automated
      pricing or risk-based approval cutoffs.
    - Brier score: mean squared error of the predicted probabilities —
      a simpler, non-log complement to log loss for probability quality.
    - Calibration error (ECE): are the probabilities themselves
      trustworthy at face value, independent of ranking quality?

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_pred : array-like of {0, 1} (thresholded predictions)
    y_proba : array-like of predicted probabilities of the positive class

    Returns
    -------
    dict
        All metrics above, keyed by lower_snake_case metric name.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
        "calibration_error": expected_calibration_error(np.asarray(y_true), np.asarray(y_proba)),
    }


# ---------------------------------------------------------------------------
# 5. THRESHOLD OPTIMIZATION
# ---------------------------------------------------------------------------


def threshold_metrics_table(
    y_true: np.ndarray, y_proba: np.ndarray,
    thresholds: Sequence[float] = config.THRESHOLD_GRID,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, specificity, F1, accuracy, and expected
    business cost at every candidate decision threshold, rather than
    assuming the default 0.50 cutoff.

    Expected cost per loan at a given threshold:
        (FN_count * cost_fn + FP_count * cost_fp) / n_loans
    using `config.COST_FALSE_NEGATIVE` / `config.COST_FALSE_POSITIVE` as
    the relative cost weights (see config.py for the business rationale:
    missing an actual default is assumed several times costlier than
    declining a borrower who would have repaid).

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_proba : array-like of predicted probabilities
    thresholds : sequence of float
    cost_fn, cost_fp : float
        Relative cost weights for a false negative / false positive.

    Returns
    -------
    pd.DataFrame
        One row per threshold with precision/recall/specificity/f1/
        accuracy/expected_cost_per_loan/n_flagged_as_risky.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else 0.0
        expected_cost = (fn * cost_fn + fp * cost_fp) / n
        rows.append(
            {
                "threshold": t, "precision": precision, "recall": recall,
                "specificity": specificity, "f1_score": f1,
                "accuracy": (tp + tn) / n, "expected_cost_per_loan": expected_cost,
                "n_flagged_as_risky": int(tp + fp),
            }
        )
    return pd.DataFrame(rows)


def recommend_threshold(threshold_table: pd.DataFrame) -> pd.Series:
    """
    Recommend the operating threshold that minimizes expected business
    cost (see `threshold_metrics_table`), rather than defaulting to 0.50
    or maximizing a purely statistical metric like F1.

    Parameters
    ----------
    threshold_table : pd.DataFrame
        Output of `threshold_metrics_table`.

    Returns
    -------
    pd.Series
        The row of `threshold_table` with the lowest expected cost per
        loan.
    """
    return threshold_table.loc[threshold_table["expected_cost_per_loan"].idxmin()]


# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------


def get_output_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """Retrieve output feature names from a fitted Pipeline's preprocessor step."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def logistic_regression_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract Logistic Regression coefficients and their odds ratios.

    Interpretation: for a one-unit increase in a standardized numeric
    feature (or moving into a one-hot category), the odds of default
    multiply by exp(coefficient) -- e.g. odds_ratio=1.20 means a 20%
    increase in the odds of default per unit increase; odds_ratio=0.80
    means a 20% decrease.

    Parameters
    ----------
    pipeline : Pipeline
        FITTED Logistic Regression pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: feature, coefficient, odds_ratio. Sorted by |coefficient|
        descending.
    """
    feature_names = get_output_feature_names_from_pipeline(pipeline)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df.sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)


def impurity_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract impurity-based (Gini/entropy decrease) feature importance
    from a fitted tree-ensemble pipeline (Random Forest).

    Caveat (reported alongside the table in the notebook): impurity
    importance is biased toward high-cardinality / continuous numeric
    features and can overstate importance for features with many
    possible split points, which is exactly why permutation importance
    is computed alongside it as a cross-check.

    Parameters
    ----------
    pipeline : Pipeline
        FITTED Random Forest pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance. Sorted descending.
    """
    feature_names = get_output_feature_names_from_pipeline(pipeline)
    importances = pipeline.named_steps["classifier"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_feature_importance(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
    n_repeats: int = 10, scoring: str = "roc_auc",
) -> pd.DataFrame:
    """
    Compute permutation importance on the ORIGINAL (pre-preprocessing)
    feature columns by permuting the full fitted `Pipeline` (preprocessor
    + classifier together). This measures how much a metric (ROC-AUC by
    default) degrades when a raw input column's values are randomly
    shuffled — a model-agnostic importance measure that, unlike impurity
    importance, is not biased toward high-cardinality features and
    reflects each RAW business variable's contribution (rather than each
    one-hot dummy's contribution separately).

    Parameters
    ----------
    pipeline : Pipeline
        FITTED pipeline (preprocessor + classifier).
    X, y : evaluation data (typically the validation or test split).
    n_repeats : int
        Number of shuffles per feature (averaged for stability).
    scoring : str
        scikit-learn scorer name.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_mean, importance_std. Sorted
        descending by importance_mean.
    """
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=config.RANDOM_STATE,
        scoring=scoring, n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def xgboost_importance_all_types(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract XGBoost's three native importance types — gain (average
    improvement in the loss function contributed by a feature, generally
    the most reliable single measure), weight (how many times a feature
    is used to split, which can overstate importance for features with
    many possible split points), and cover (average number of samples
    affected by splits on a feature) — mapped from XGBoost's internal
    "fN" feature indices back to human-readable feature names.

    Parameters
    ----------
    pipeline : Pipeline
        FITTED XGBoost pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: feature, gain, weight, cover. Sorted descending by gain.
        Features never used as a split get 0 across all three columns.
    """
    feature_names = get_output_feature_names_from_pipeline(pipeline)
    booster = pipeline.named_steps["classifier"].get_booster()

    index_to_name = {f"f{i}": name for i, name in enumerate(feature_names)}
    rows = {name: {"gain": 0.0, "weight": 0.0, "cover": 0.0} for name in feature_names}

    for importance_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=importance_type)
        for f_key, value in scores.items():
            name = index_to_name.get(f_key, f_key)
            if name in rows:
                rows[name][importance_type] = value

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "feature"})
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. VISUALIZATIONS
# ---------------------------------------------------------------------------


def plot_confusion_matrix_chart(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, subtitle: Optional[str] = None,
) -> plt.Figure:
    """Confusion matrix heatmap with raw counts, labeled Predicted/Actual."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap=PALETTE_DIVERGING, cbar=False, ax=ax,
        xticklabels=["Fully Paid (0)", "Default (1)"],
        yticklabels=["Fully Paid (0)", "Default (1)"],
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_roc_curve_chart(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, subtitle: Optional[str] = None,
) -> plt.Figure:
    """ROC curve with the diagonal no-skill reference line and AUC annotated."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(fpr, tpr, color=COLOR_DEFAULT, linewidth=2.2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="No-skill baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(frameon=False, loc="lower right")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_pr_curve_chart(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, subtitle: Optional[str] = None,
) -> plt.Figure:
    """Precision-Recall curve with the no-skill baseline (positive class prevalence)."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(recall, precision, color=COLOR_DEFAULT, linewidth=2.2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, loc="upper right")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_calibration_curve_chart(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, subtitle: Optional[str] = None,
    n_bins: int = config.CALIBRATION_BINS,
) -> plt.Figure:
    """
    Calibration (reliability) curve: mean predicted probability vs.
    observed default rate per bin, against the perfect-calibration
    diagonal.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, observed_rates = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba > lo) & (y_proba <= hi) if lo > 0 else (y_proba >= lo) & (y_proba <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(y_proba[mask].mean())
        observed_rates.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.plot(bin_centers, observed_rates, marker="o", color=COLOR_DEFAULT, linewidth=2, label="Model")
    ax.set_xlabel("Mean Predicted Default Probability")
    ax.set_ylabel("Observed Default Rate")
    ax.legend(frameon=False, loc="upper left")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_learning_curve_chart(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, title: str,
    subtitle: Optional[str] = None, cv_folds: int = config.CV_FOLDS,
    scoring: str = config.CV_SCORING,
    train_sizes: Sequence[float] = config.LEARNING_CURVE_TRAIN_SIZES,
) -> plt.Figure:
    """
    Learning curve: training vs. cross-validated score as a function of
    training-set size, used to diagnose high bias (both curves low and
    close together — more data won't help much) vs. high variance (large
    gap between train and CV score — more data or more regularization
    would likely help).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    # A CalibratedClassifierCV(cv="prefit") cannot be re-fit by
    # learning_curve (the clone it fits is never pre-fit, so every CV
    # fold raises NotFittedError). Unwrap to the base estimator pipeline
    # -- which is exactly what a learning curve is meant to diagnose.
    estimator = pipeline
    if isinstance(pipeline, CalibratedClassifierCV):
        estimator = pipeline.estimator
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes,
        n_jobs=-1, random_state=config.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(sizes, train_scores.mean(axis=1), marker="o", color=COLOR_PAID, label="Training score")
    ax.fill_between(sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), color=COLOR_PAID, alpha=0.15)
    ax.plot(sizes, val_scores.mean(axis=1), marker="o", color=COLOR_DEFAULT, label="Cross-validation score")
    ax.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), color=COLOR_DEFAULT, alpha=0.15)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel(scoring.replace("_", " ").upper())
    ax.legend(frameon=False, loc="lower right")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_validation_curve_chart(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, param_name: str,
    param_range: Sequence[Any], title: str, subtitle: Optional[str] = None,
    cv_folds: int = config.CV_FOLDS, scoring: str = config.CV_SCORING,
) -> plt.Figure:
    """
    Validation curve: training vs. cross-validated score as a single
    hyperparameter varies, holding all others at their tuned value —
    shows exactly where a model starts to overfit (train score keeps
    rising while CV score plateaus or falls) as complexity increases.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline with all OTHER hyperparameters already set to their
        tuned values (e.g. `search.best_estimator_`).
    param_name : str
        Full `classifier__*`-prefixed parameter name to vary.
    param_range : sequence
        Candidate values for `param_name`.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    # Same unwrap as plot_learning_curve_chart: a fitted
    # CalibratedClassifierCV(cv="prefit") cannot be re-fit by the curve
    # helpers, so evaluate the base estimator pipeline instead.
    estimator = pipeline
    if isinstance(pipeline, CalibratedClassifierCV):
        estimator = pipeline.estimator
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(param_range, train_scores.mean(axis=1), marker="o", color=COLOR_PAID, label="Training score")
    ax.plot(param_range, val_scores.mean(axis=1), marker="o", color=COLOR_DEFAULT, label="Cross-validation score")
    ax.set_xlabel(param_name.replace("classifier__", ""))
    ax.set_ylabel(scoring.replace("_", " ").upper())
    ax.legend(frameon=False, loc="best")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_importance_bar(
    importance_df: pd.DataFrame, value_column: str, title: str,
    subtitle: Optional[str] = None, top_n: int = 15,
) -> plt.Figure:
    """Generic horizontal bar chart for any feature-importance table (impurity, permutation, gain/weight/cover)."""
    plot_df = importance_df.nlargest(top_n, value_column).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(plot_df) + 1.5)))
    ax.barh(plot_df["feature"], plot_df[value_column], color=COLOR_PAID)
    ax.set_xlabel(value_column.replace("_", " ").title())
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_coefficient_chart(
    coef_df: pd.DataFrame, title: str, subtitle: Optional[str] = None, top_n: int = 15,
) -> plt.Figure:
    """
    Logistic Regression coefficient plot: horizontal bars colored by
    sign (red = increases default odds, blue = decreases default odds),
    annotated with each feature's odds ratio.
    """
    plot_df = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    plot_df = plot_df.sort_values("coefficient")
    colors = [COLOR_DEFAULT if c > 0 else COLOR_PAID for c in plot_df["coefficient"]]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(plot_df) + 1.5)))
    ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    for i, (coef, odds) in enumerate(zip(plot_df["coefficient"], plot_df["odds_ratio"])):
        ax.text(coef, i, f"  OR={odds:.2f}", va="center",
                ha="left" if coef >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds)")
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_probability_distribution_chart(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, subtitle: Optional[str] = None,
) -> plt.Figure:
    """Overlaid histograms of predicted default probability, split by actual outcome — a well-separated model shows two distinct humps."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.histplot(y_proba[y_true == 0], bins=30, color=COLOR_PAID, alpha=0.6, label="Actually Fully Paid", ax=ax, stat="density")
    sns.histplot(y_proba[y_true == 1], bins=30, color=COLOR_DEFAULT, alpha=0.6, label="Actually Defaulted", ax=ax, stat="density")
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _apply_titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_threshold_analysis_chart(
    threshold_table: pd.DataFrame, recommended_threshold: float,
    title: str, subtitle: Optional[str] = None,
) -> plt.Figure:
    """
    Precision / recall / F1 / expected-cost curves across the full
    threshold grid, with the recommended (cost-minimizing) threshold
    marked — the visual backbone of the Threshold Optimization section.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", color=COLOR_PAID)
    ax1.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", color=COLOR_DEFAULT)
    ax1.plot(threshold_table["threshold"], threshold_table["f1_score"], label="F1", color="gray", linestyle="--")
    ax1.axvline(recommended_threshold, color="black", linestyle=":", linewidth=1.5,
                label=f"Recommended = {recommended_threshold:.2f}")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.set_title("Precision / Recall / F1 vs. Threshold", fontsize=12, fontweight="bold", loc="left")

    ax2.plot(threshold_table["threshold"], threshold_table["expected_cost_per_loan"], color=COLOR_DEFAULT, linewidth=2.2)
    ax2.axvline(recommended_threshold, color="black", linestyle=":", linewidth=1.5,
                label=f"Recommended = {recommended_threshold:.2f}")
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Expected Cost per Loan (relative units)")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.set_title("Expected Business Cost vs. Threshold", fontsize=12, fontweight="bold", loc="left")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    if subtitle:
        fig.text(0.01, 1.0, subtitle, fontsize=10, color="dimgray", style="italic")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. MODEL COMPARISON
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """
    Standardized container for everything Phase 3 produces about one
    trained model, so the comparison table, serialization step, and
    Phase 4/5 hand-off can all read from one consistent structure.
    """

    model_key: str
    best_estimator: Pipeline
    best_params: Dict[str, Any]
    cv_fold_results: pd.DataFrame
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time_sec: float
    search_time_sec: float
    prediction_time_ms_per_1000: float
    y_proba_test: np.ndarray
    threshold_table: pd.DataFrame
    recommended_threshold: float
    feature_importance: Dict[str, pd.DataFrame] = field(default_factory=dict)
    calibrated_estimator: Optional[Pipeline] = None
    calibrated: bool = False

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY_NAMES.get(self.model_key, self.model_key)


# Qualitative deployment attributes that don't come out of a metric
# calculation — centralized here (not invented ad hoc in the notebook)
# so the comparison table's judgment calls are transparent and consistent.
MODEL_QUALITATIVE_ATTRIBUTES: Dict[str, Dict[str, str]] = {
    "logistic_regression": {
        "interpretability": "High — coefficients map directly to odds ratios",
        "complexity": "Low — linear model, few hyperparameters",
        "deployment_readiness": "Very high — trivial to serve, no special runtime",
    },
    "random_forest": {
        "interpretability": "Medium — importances available, individual trees opaque",
        "complexity": "Medium-High — many trees, moderate tuning surface",
        "deployment_readiness": "High — standard scikit-learn artifact",
    },
    "xgboost": {
        "interpretability": "Medium-Low — needs SHAP (Phase 4) for per-loan explanations",
        "complexity": "High — largest hyperparameter surface, sequential boosting",
        "deployment_readiness": "High — widely supported serving runtimes",
    },
}


def build_model_comparison_table(results: List[ModelResult]) -> pd.DataFrame:
    """
    Assemble the executive model-comparison table: performance metrics
    (test-set), cross-validation summary, timing, and qualitative
    deployment attributes, ranked by test ROC-AUC (the primary metric
    this project optimizes for, per config.CV_SCORING).

    Parameters
    ----------
    results : list[ModelResult]

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by test ROC-AUC descending, with a
        `rank` column (1 = best).
    """
    rows = []
    for r in results:
        cv_mean = r.cv_fold_results.loc[r.cv_fold_results["fold"] == "mean", config.CV_SCORING].iloc[0]
        cv_std = r.cv_fold_results.loc[r.cv_fold_results["fold"] == "std", config.CV_SCORING].iloc[0]
        model_path = {
            "logistic_regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
            "random_forest": config.RANDOM_FOREST_MODEL_PATH,
            "xgboost": config.XGBOOST_MODEL_PATH,
        }[r.model_key]
        memory_kb = model_path.stat().st_size / 1024 if model_path.exists() else np.nan

        row = {
            "model": r.display_name,
            "accuracy": r.test_metrics["accuracy"],
            "precision": r.test_metrics["precision"],
            "recall": r.test_metrics["recall"],
            "specificity": r.test_metrics["specificity"],
            "f1_score": r.test_metrics["f1_score"],
            "roc_auc": r.test_metrics["roc_auc"],
            "balanced_accuracy": r.test_metrics["balanced_accuracy"],
            f"cv_{config.CV_SCORING}_mean": cv_mean,
            f"cv_{config.CV_SCORING}_std": cv_std,
            "training_time_sec": r.training_time_sec,
            "prediction_time_ms_per_1000": r.prediction_time_ms_per_1000,
            "memory_usage_kb": round(memory_kb, 1),
            **MODEL_QUALITATIVE_ATTRIBUTES[r.model_key],
        }
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def resolve_production_model_key(comparison_table: pd.DataFrame) -> str:
    """
    Choose the production scoring model from the Phase 3 comparison table:
    the row with the highest TEST ROC-AUC, reverse-mapped from its display
    name back to a machine key via `MODEL_DISPLAY_NAMES`.

    This is the single source of truth for "which model is production" --
    the Streamlit app resolves it at runtime so the constant
    `config.PRODUCTION_MODEL_KEY` no longer determines the app's model.

    Parameters
    ----------
    comparison_table : pd.DataFrame
        As persisted to `config.MODEL_COMPARISON_TABLE_PATH` by
        `build_model_comparison_table` (one row per model, with a `model`
        display-name column and a `roc_auc` column). Input order is
        irrelevant; the max is computed directly, not from the `rank`
        column.

    Returns
    -------
    str
        The model key with the highest test ROC-AUC, or
        `config.PRODUCTION_MODEL_KEY` whenever the winner cannot be
        confidently identified (empty table, missing/non-numeric
        `roc_auc`, or a `model` name that does not map back to a key).
    """
    if comparison_table is None or comparison_table.empty:
        return config.PRODUCTION_MODEL_KEY
    if "model" not in comparison_table.columns or "roc_auc" not in comparison_table.columns:
        return config.PRODUCTION_MODEL_KEY
    roc_auc = pd.to_numeric(comparison_table["roc_auc"], errors="coerce")
    if roc_auc.isna().all():
        return config.PRODUCTION_MODEL_KEY
    # idxmax() returns the first occurrence of the max, so a tie resolves
    # deterministically rather than depending on row order from the CSV.
    best_display_name = comparison_table.loc[roc_auc.idxmax(), "model"]
    reverse_map = {display: key for key, display in MODEL_DISPLAY_NAMES.items()}
    return reverse_map.get(best_display_name, config.PRODUCTION_MODEL_KEY)


# ---------------------------------------------------------------------------
# 9. END-TO-END ORCHESTRATION
# ---------------------------------------------------------------------------


def train_and_evaluate_model(
    model_key: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> ModelResult:
    """
    End-to-end Phase 3 workflow for a single model: hyperparameter
    search with stratified CV -> refit on full training data (handled
    automatically by `refit=True` inside the search) -> metrics on
    train/validation/test -> threshold optimization on the validation
    set (test set stays untouched by any decision made from it) ->
    feature importance extraction.

    Design decision: the THRESHOLD is optimized on the VALIDATION split,
    not the test split — the test split is reserved purely for final,
    unbiased performance reporting. Tuning the operating threshold on
    test data would leak test-set information into a modeling decision.

    Parameters
    ----------
    model_key : str
        One of "logistic_regression", "random_forest", "xgboost".
    X_train, y_train, X_val, y_val, X_test, y_test : the Phase 1 splits.

    Returns
    -------
    ModelResult
    """
    logger.info("=" * 70)
    logger.info("Training and evaluating: %s", MODEL_DISPLAY_NAMES[model_key])
    logger.info("=" * 70)

    search, search_time = tune_model(model_key, X_train, y_train)
    best_estimator: Pipeline = search.best_estimator_
    training_time = getattr(search, "refit_time_", np.nan)

    cv_fold_results = extract_cv_fold_results(search)

    # Metrics at the default 0.50 threshold for train/val (diagnostic —
    # overfitting/underfitting assessment); test metrics are reported at
    # BOTH 0.50 (for like-for-like model comparison) and the recommended
    # threshold (for the realistic business recommendation).
    def _predict_and_score(X, y) -> Dict[str, float]:
        proba = best_estimator.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return compute_classification_metrics(y, pred, proba)

    train_metrics = _predict_and_score(X_train, y_train)
    val_metrics = _predict_and_score(X_val, y_val)

    start = time.perf_counter()
    y_proba_test = best_estimator.predict_proba(X_test)[:, 1]
    elapsed_sec = time.perf_counter() - start
    # ms per 1,000 predictions = (seconds / n_predictions) * 1000 predictions * 1000 ms/s
    prediction_time_ms_per_1000 = (elapsed_sec / len(X_test)) * 1000 * 1000
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)

    # Threshold optimization on the VALIDATION set (test set held out).
    y_proba_val = best_estimator.predict_proba(X_val)[:, 1]
    threshold_table = threshold_metrics_table(y_val, y_proba_val)
    recommended = float(recommend_threshold(threshold_table)["threshold"])

    # Feature importance.
    feature_importance: Dict[str, pd.DataFrame] = {}
    if model_key == "logistic_regression":
        feature_importance["coefficients"] = logistic_regression_coefficients(best_estimator)
        feature_importance["permutation"] = permutation_feature_importance(best_estimator, X_val, y_val)
    elif model_key == "random_forest":
        feature_importance["impurity"] = impurity_feature_importance(best_estimator)
        feature_importance["permutation"] = permutation_feature_importance(best_estimator, X_val, y_val)
    elif model_key == "xgboost":
        feature_importance["gain_weight_cover"] = xgboost_importance_all_types(best_estimator)
        feature_importance["permutation"] = permutation_feature_importance(best_estimator, X_val, y_val)

    logger.info(
        "%s complete. Test ROC-AUC=%.4f | Recommended threshold=%.2f | "
        "Training time=%.3fs | Search time=%.1fs",
        MODEL_DISPLAY_NAMES[model_key], test_metrics["roc_auc"], recommended,
        training_time, search_time,
    )

    # Optional calibration: fit a CalibratedClassifierCV on the VALIDATION
    # split so that final serialized models produce calibrated
    # probabilities. Use cv='prefit' to wrap the already-refit best
    # estimator and fit only the calibration layer on X_val / y_val.
    calibrated_estimator = None
    calibrated_flag = False
    if config.CALIBRATE_MODELS:
        try:
            logger.info("Fitting calibration wrapper (method=%s) on validation set.", config.CALIBRATION_METHOD)

            # Ensure the returned best_estimator is fitted. `cv='prefit'` in
            # CalibratedClassifierCV requires a fitted estimator. If for some
            # reason the search did not refit on the full training set,
            # explicitly fit it here before wrapping.
            try:
                check_is_fitted(best_estimator)
            except NotFittedError:
                logger.warning("Best estimator not fitted after search; fitting on full training set before calibration.")
                best_estimator.fit(X_train, y_train)

            calibrator = CalibratedClassifierCV(estimator=best_estimator, method=config.CALIBRATION_METHOD, cv="prefit")
            calibrator.fit(X_val, y_val)
            calibrated_estimator = calibrator
            calibrated_flag = True

            # Recompute test/val/train probabilities/metrics using calibrated estimator
            def _predict_and_score_calib(X, y) -> Dict[str, float]:
                proba = calibrated_estimator.predict_proba(X)[:, 1]
                pred = (proba >= 0.5).astype(int)
                return compute_classification_metrics(y, pred, proba)

            train_metrics = _predict_and_score_calib(X_train, y_train)
            val_metrics = _predict_and_score_calib(X_val, y_val)
            y_proba_test = calibrated_estimator.predict_proba(X_test)[:, 1]
            y_pred_test = (y_proba_test >= 0.5).astype(int)
            test_metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)
            # Recompute threshold table on calibrated validation probabilities
            y_proba_val = calibrated_estimator.predict_proba(X_val)[:, 1]
            threshold_table = threshold_metrics_table(y_val, y_proba_val)
            recommended = float(recommend_threshold(threshold_table)["threshold"])
        except Exception as exc:  # pragma: no cover - calibration best-effort
            logger.exception("Calibration failed: %s. Proceeding with uncalibrated estimator.", exc)
            calibrated_estimator = None
            calibrated_flag = False

    return ModelResult(
        model_key=model_key,
        best_estimator=best_estimator,
        best_params=search.best_params_,
        cv_fold_results=cv_fold_results,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        training_time_sec=training_time,
        search_time_sec=search_time,
        prediction_time_ms_per_1000=prediction_time_ms_per_1000,
        y_proba_test=y_proba_test,
        threshold_table=threshold_table,
        recommended_threshold=recommended,
        feature_importance=feature_importance,
        calibrated_estimator=calibrated_estimator,
        calibrated=calibrated_flag,
    )
