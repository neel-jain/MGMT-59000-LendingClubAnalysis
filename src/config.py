"""
config.py
=========
Central configuration module for the MGMT 590 LendingClub Loan Default
Risk capstone project.

This module is the SINGLE SOURCE OF TRUTH for:
    - File / directory paths
    - Column name conventions
    - Data cleaning parameters (mappings, valid categories, thresholds)
    - Train / validation / test split parameters
    - Modeling constants (random seed, target column name)
    - Logging configuration

Design intent
-------------
Every later phase (EDA, model training, evaluation, Streamlit deployment)
should import shared constants from this module instead of hard-coding
values. This guarantees that a change made in one place (e.g. moving the
raw data file, changing the random seed, or adding a new categorical
column) propagates consistently through the entire pipeline.

Nothing in this file performs I/O or computation — it is pure
configuration and should have no side effects on import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT & DIRECTORY STRUCTURE
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the top-level project folder regardless of the
# current working directory from which a script/notebook is launched, as
# long as this file stays at <PROJECT_ROOT>/src/config.py. This can be
# overridden via the MGMT590_PROJECT_ROOT environment variable -- useful
# in containerized deployments where data/model artifacts are mounted at
# a different path than the source code itself (e.g. a read-only image
# with a separate writable volume for data/models/reports/logs).
PROJECT_ROOT: Path = Path(os.environ.get("MGMT590_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PIPELINES_DIR: Path = PROJECT_ROOT / "pipelines"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
APP_DIR: Path = PROJECT_ROOT / "app"
# New in Phase 3: non-model evaluation artifacts (metrics tables, CV
# results, feature-importance tables, probability predictions). Kept
# separate from MODELS_DIR so "the three serialized estimators" and
# "everything describing how well they performed" are unambiguous on
# disk, which matters once the Streamlit app (Phase 5) needs to load
# only the former.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
# New in Phase 4A: explainability/risk-scoring artifacts (SHAP importance
# tables, model metadata, business-summary templates, fairness report).
# Kept as a subdirectory of REPORTS_DIR (not a new top-level directory)
# since these are, like Phase 3's reports, downstream *evaluation*
# artifacts rather than a new pipeline stage -- they describe/explain
# the already-trained models rather than producing new ones.
EXPLAINABILITY_DIR: Path = REPORTS_DIR / "explainability"
# New in Phase 4B: borrower-segmentation artifacts (clustering model,
# centroids, dimensionality-reduction objects, segment profiles). Same
# rationale as EXPLAINABILITY_DIR -- a subdirectory of REPORTS_DIR, not
# a new top-level directory, since segmentation is a second downstream
# ANALYSIS of the same fitted production pipeline's feature space,
# parallel to (not a replacement for) supervised prediction.
SEGMENTATION_DIR: Path = REPORTS_DIR / "segmentation"

# Directories that must exist before any pipeline step runs. Created lazily
# (idempotently) by `ensure_directories()` in utils.py rather than at
# import time, so importing config.py never touches the filesystem.
REQUIRED_DIRS: List[Path] = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    PIPELINES_DIR,
    REPORTS_DIR,
    EXPLAINABILITY_DIR,
    SEGMENTATION_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    APP_DIR,
]

# ---------------------------------------------------------------------------
# 2. FILE PATHS
# ---------------------------------------------------------------------------
# Raw input file. The project brief specifies the Indiana-only LendingClub
# extract (~37,515 rows, ~27.5 MB). Replace this file with the real export
# before running the pipeline end-to-end.
RAW_DATA_FILENAME: str = "lendingclub_indiana_raw.csv"
RAW_DATA_PATH: Path = RAW_DATA_DIR / RAW_DATA_FILENAME

# Cleaned / feature-ready dataset (post validation + cleaning, pre-split).
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "lendingclub_indiana_cleaned.csv"

# Train / validation / test split outputs (features and target saved
# separately to make downstream loading explicit and leakage-safe).
X_TRAIN_PATH: Path = SPLITS_DIR / "X_train.csv"
X_VAL_PATH: Path = SPLITS_DIR / "X_val.csv"
X_TEST_PATH: Path = SPLITS_DIR / "X_test.csv"
Y_TRAIN_PATH: Path = SPLITS_DIR / "y_train.csv"
Y_VAL_PATH: Path = SPLITS_DIR / "y_val.csv"
Y_TEST_PATH: Path = SPLITS_DIR / "y_test.csv"

# Serialized fitted preprocessing pipeline (joblib). Fit ONLY on training
# data; reused (via .transform) on validation/test/live inference data to
# prevent data leakage.
PREPROCESSOR_PATH: Path = PIPELINES_DIR / "preprocessing_pipeline.joblib"

# Serialized trained models (populated in Phase 3 — Logistic Regression,
# Random Forest, XGBoost). Filenames fixed now so later phases and the
# Streamlit app can reference stable paths.
LOGISTIC_REGRESSION_MODEL_PATH: Path = MODELS_DIR / "logistic_regression_model.joblib"
RANDOM_FOREST_MODEL_PATH: Path = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_model.joblib"

# Phase 3 evaluation artifacts (joblib). Each is a dict keyed by model
# name ("logistic_regression", "random_forest", "xgboost") so downstream
# phases (SHAP in Phase 4, the Streamlit app in Phase 5) can load exactly
# the slice they need without re-running training or evaluation.
EVALUATION_METRICS_PATH: Path = REPORTS_DIR / "evaluation_metrics.joblib"
CV_RESULTS_PATH: Path = REPORTS_DIR / "cv_results.joblib"
FEATURE_IMPORTANCE_PATH: Path = REPORTS_DIR / "feature_importance.joblib"
PROBABILITY_PREDICTIONS_PATH: Path = REPORTS_DIR / "probability_predictions.joblib"
THRESHOLD_ANALYSIS_PATH: Path = REPORTS_DIR / "threshold_analysis.joblib"
MODEL_COMPARISON_TABLE_PATH: Path = REPORTS_DIR / "model_comparison_table.csv"

# Serialized clustering model (populated in Phase 4B). A single fitted
# clustering estimator, kept alongside the Phase 3 supervised models in
# MODELS_DIR since it is likewise "a fitted model artifact", distinct
# from the descriptive/summary artifacts under SEGMENTATION_DIR.
CLUSTERING_MODEL_PATH: Path = MODELS_DIR / "clustering_model.joblib"

# Log file for the full pipeline run.
PIPELINE_LOG_PATH: Path = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# 3. REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# 4. TARGET / FILTERING DEFINITIONS
# ---------------------------------------------------------------------------
# Only Indiana borrowers are in scope for this project.
TARGET_STATE: str = "IN"
STATE_COLUMN: str = "addr_state"

# Raw loan_status values and how they map onto the binary target.
# Any loan_status value NOT present in this mapping is dropped (e.g.
# "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)")
# because those loans have not reached a final resolution and including
# them would leak future information / mislabel censored outcomes.
LOAN_STATUS_COLUMN: str = "loan_status"
TARGET_COLUMN: str = "default_flag"

LOAN_STATUS_TARGET_MAP: Dict[str, int] = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
}

# ---------------------------------------------------------------------------
# 5. COLUMN GROUPS
# ---------------------------------------------------------------------------
# These lists drive both the cleaning functions in utils.py and the
# ColumnTransformer built in build_preprocessing_pipeline(). Keeping the
# groups here (rather than inferring dtypes dynamically at runtime) makes
# the pipeline deterministic and easy to audit.

# Columns that arrive as "12.5%"-style strings and must be converted to
# numeric floats (e.g. 12.5) before modeling.
PERCENTAGE_COLUMNS: List[str] = [
    "int_rate",
    "revol_util",
]

# Raw employment-length column (e.g. "10+ years", "< 1 year", "3 years")
# and the numeric column it is parsed into.
EMP_LENGTH_RAW_COLUMN: str = "emp_length"
EMP_LENGTH_NUMERIC_COLUMN: str = "emp_length_years"

# Mapping used by parse_emp_length(); values not found default to NaN and
# are imputed later using the median strategy defined in NUMERIC_FEATURES.
EMP_LENGTH_MAP: Dict[str, float] = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}

# Numeric predictor columns fed into the pipeline's numeric branch
# (imputation + scaling). emp_length_years and the parsed percentage
# columns are included here since they are numeric AFTER cleaning.
NUMERIC_FEATURES: List[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "emp_length_years",
]

# Low/medium-cardinality categorical columns -> one-hot encoded.
ONEHOT_CATEGORICAL_FEATURES: List[str] = [
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
]

# Ordinal categorical columns -> ordinal encoded (natural risk ordering).
ORDINAL_CATEGORICAL_FEATURES: List[str] = [
    "grade",
]
ORDINAL_CATEGORY_ORDER: List[List[str]] = [
    ["A", "B", "C", "D", "E", "F", "G"],
]

# Columns intentionally EXCLUDED from modeling (identifiers, free text,
# leakage-prone or post-origination fields, and the raw pre-cleaning
# versions of engineered columns). Kept here explicitly (rather than
# silently dropped) so the rationale is auditable in one place.
EXCLUDED_COLUMNS: List[str] = [
    "id",
    "member_id",
    "emp_title",
    "url",
    "desc",
    "title",
    "zip_code",
    "addr_state",
    "issue_d",
    "earliest_cr_line",
    "sub_grade",
    "loan_status",  # replaced by TARGET_COLUMN
    "emp_length",  # replaced by EMP_LENGTH_NUMERIC_COLUMN
]

# All columns the raw ingestion step expects to find. Used by
# validate_dataset() to flag schema drift early.
EXPECTED_RAW_COLUMNS: List[str] = sorted(
    set(
        NUMERIC_FEATURES
        + ONEHOT_CATEGORICAL_FEATURES
        + ORDINAL_CATEGORICAL_FEATURES
        + EXCLUDED_COLUMNS
        + [LOAN_STATUS_COLUMN, STATE_COLUMN, EMP_LENGTH_RAW_COLUMN]
    )
    - {EMP_LENGTH_NUMERIC_COLUMN}  # engineered, not present in raw file
)

# ---------------------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------
# Proportions must sum to 1.0. Test is held out first, then validation is
# carved out of the remaining training pool, so the test set is never used
# to inform any preprocessing or modeling decision (leakage prevention).
TEST_SIZE: float = 0.15
VALIDATION_SIZE: float = 0.15  # fraction of the ORIGINAL full dataset
STRATIFY_COLUMN: str = TARGET_COLUMN

# ---------------------------------------------------------------------------
# 7. LOGGING
# ---------------------------------------------------------------------------
# Overridable via the MGMT590_LOG_LEVEL environment variable (e.g. set to
# "WARNING" in a production deployment to reduce log volume without a
# code change). Falls back silently to INFO for any unrecognized value.
LOG_LEVEL: int = getattr(logging, os.environ.get("MGMT590_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# 8. PHASE 3 — MODELING CONFIGURATION
# ---------------------------------------------------------------------------
# Everything below parameterizes src/train_models.py's Phase 3 functions
# and src/model_utils.py. Kept in config.py (rather than hard-coded in the
# modeling code) for the same reason as every other constant in this
# file: one place to audit and adjust without touching pipeline logic.

# --- Cross-validation ---
# 5 folds is the standard textbook default and a deliberate choice here:
# with a training set in the low thousands of loans and a minority
# (default) class around 20-25%, 5 folds keeps at least ~40-50 positive
# cases in every validation fold (10 folds would roughly halve that,
# making fold-level AUC/PR estimates noisier for little bias benefit).
# StratifiedKFold is used everywhere (not plain KFold) so every fold
# preserves the overall default rate.
CV_FOLDS: int = 5

# Primary metric optimized during hyperparameter search. ROC-AUC (not
# accuracy) is used because the target is imbalanced (~20-25% default)
# and a model can score >75% accuracy by predicting "no default" for
# everyone -- ROC-AUC instead measures how well the model RANKS
# defaulters above non-defaulters across all thresholds, which is what
# actually matters for a lending decision engine that will pick its own
# operating threshold (see Threshold Optimization section).
CV_SCORING: str = "roc_auc"

# --- Hyperparameter search spaces ---
# Logistic Regression: small, well-understood parameter space -> an
# exhaustive GridSearchCV is cheap and guarantees the global optimum
# over the grid (no need for randomized search here).
LOGISTIC_REGRESSION_PARAM_GRID: Dict[str, list] = {
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear"],  # supports both l1 and l2
    "classifier__class_weight": [None, "balanced"],
}

# Random Forest: a much larger, higher-dimensional space (tree count,
# depth, split/leaf sizes, feature sampling) where a full grid would be
# combinatorially expensive for little added benefit -> RandomizedSearchCV
# samples a fixed budget of combinations, which explores the space far
# more efficiently per unit of compute (Bergstra & Bengio, 2012).
RANDOM_FOREST_PARAM_DISTRIBUTIONS: Dict[str, list] = {
    "classifier__n_estimators": [200, 300, 400, 500],
    "classifier__max_depth": [4, 6, 8, 10, 12, 16, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features": ["sqrt", "log2", 0.5],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
}
RANDOM_FOREST_N_ITER: int = 20

# XGBoost: similarly high-dimensional (tree count, depth, learning rate,
# row/column subsampling, regularization) -> RandomizedSearchCV again,
# for the same efficiency reason as Random Forest.
XGBOOST_PARAM_DISTRIBUTIONS: Dict[str, list] = {
    "classifier__n_estimators": [150, 250, 350, 450, 600],
    "classifier__max_depth": [3, 4, 5, 6, 8],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1.0],
    "classifier__reg_lambda": [0.1, 1.0, 5.0, 10.0],
}
XGBOOST_N_ITER: int = 30

# --- Threshold optimization ---
# Grid of candidate decision thresholds evaluated in Section on Threshold
# Optimization (finer resolution near the likely operating region).
THRESHOLD_GRID: List[float] = [round(t, 2) for t in
                                [i / 100 for i in range(5, 96, 5)]]

# Relative business cost of a False Negative (approving a loan to a
# borrower who actually defaults: principal + accrued interest at risk,
# collections cost) vs. a False Positive (declining a borrower who would
# have repaid: forgone interest margin on one loan). LendingClub-style
# unsecured consumer loans typically lose a much larger multiple of the
# expected interest margin when a loan defaults than they forgo by
# rejecting a single good applicant, so FN is weighted more heavily by
# default. This ratio is a business assumption, not a statistical
# estimate -- Phase 5's app should expose it as an adjustable input for
# credit-policy stakeholders rather than treating 5:1 as fixed truth.
COST_FALSE_NEGATIVE: float = 5.0
COST_FALSE_POSITIVE: float = 1.0

# Number of bins used to compute Expected Calibration Error (ECE) and to
# draw calibration curves.
CALIBRATION_BINS: int = 10

# Whether to fit a calibration wrapper (Platt/isotonic) after model
# training. When True, the training pipeline will fit a calibration
# model on the VALIDATION split and the calibrated estimator will be
# serialized to the standard model path so downstream consumers (the
# Streamlit app, explainability artifacts) receive calibrated
# probabilities. Set False to preserve raw classifier outputs.
CALIBRATE_MODELS: bool = True

# Calibration method: 'sigmoid' (Platt scaling) or 'isotonic'.
# 'sigmoid' is typically more robust with smaller validation sets;
# 'isotonic' is non-parametric and can better fit large validation
# sets but risks overfitting with limited data. Default 'sigmoid'.
CALIBRATION_METHOD: str = "sigmoid"

# Learning-curve training-set-size fractions.
LEARNING_CURVE_TRAIN_SIZES: List[float] = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]

# Cloud-safe settings for the dashboard's learning-curve tab. The full
# multi-fold refit of the production model over the whole training set
# (35 Random Forest fits) can exhaust memory/run time on Streamlit
# Community Cloud's free tier and surface as an HTTP 503 on the app, so
# `get_learning_curve_figure` computes the curve on a capped subsample
# with fewer folds and training-size fractions.
LEARNING_CURVE_APP_MAX_ROWS: int = 2500
LEARNING_CURVE_APP_FOLDS: int = 3
LEARNING_CURVE_APP_TRAIN_SIZES: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]

# ---------------------------------------------------------------------------
# 9. PHASE 4A -- EXPLAINABILITY & RISK SCORING CONFIGURATION
# ---------------------------------------------------------------------------
# Business-POLICY thresholds (risk tiers, lending actions, interest-rate
# adjustments, loan-grade bands) deliberately do NOT live here -- see
# src/configurable_thresholds.py. This section holds only engineering
# constants: which model Phase 4A treats as "production", file paths for
# serialized explainability artifacts, and SHAP computation parameters.

# OFFLINE/FALLBACK default used by notebooks/tests and by the app only
# when reports/model_comparison_table.csv is absent. The Streamlit app
# resolves the real production model at runtime from the comparison
# table (the row with the highest TEST ROC-AUC) via
# model_utils.resolve_production_model_key() -- this constant no longer
# determines the app's production model. Kept in sync with the current
# comparison-table winner so the fallback matches reality.
PRODUCTION_MODEL_KEY: str = "xgboost"

# SHAP computation is O(background_size x n_samples_explained) for
# TreeExplainer and can be considerably more expensive for other
# explainer types -- both sample sizes are capped so global explanations
# stay fast in an interactive Streamlit session while remaining large
# enough for stable importance estimates.
SHAP_BACKGROUND_SAMPLE_SIZE: int = 100
SHAP_GLOBAL_SAMPLE_SIZE: int = 150

# --- Phase 4A artifact paths (all under EXPLAINABILITY_DIR) ---
MODEL_METADATA_PATH: Path = EXPLAINABILITY_DIR / "model_metadata.joblib"
SHAP_IMPORTANCE_PATH: Path = EXPLAINABILITY_DIR / "shap_feature_importance.joblib"
BUSINESS_SUMMARY_TEMPLATES_PATH: Path = EXPLAINABILITY_DIR / "business_summary_templates.joblib"
FAIRNESS_REPORT_PATH: Path = EXPLAINABILITY_DIR / "fairness_report.joblib"
FEATURE_INTERACTION_SUMMARY_PATH: Path = EXPLAINABILITY_DIR / "feature_interaction_summary.joblib"

# Risk-tier / lending-action / interest-rate / loan-grade business rules
# are persisted as human-editable JSON (not joblib) specifically so a
# credit-policy stakeholder can open and change them in a text editor
# without touching Python -- see configurable_thresholds.py.
RISK_THRESHOLD_CONFIG_PATH: Path = REPORTS_DIR / "risk_threshold_config.json"

# ---------------------------------------------------------------------------
# 10. PHASE 4B -- BORROWER SEGMENTATION CONFIGURATION
# ---------------------------------------------------------------------------

# Features used to compute inter-borrower DISTANCE for clustering.
# Deliberately NUMERIC + ORDINAL only (excludes the one-hot categorical
# columns in ONEHOT_CATEGORICAL_FEATURES). Rationale, expanded on in the
# Phase 4B notebook's Data Preparation section: one-hot dummy columns are
# binary (0/1) and, under Euclidean distance, several correlated dummies
# from the same categorical variable can collectively dominate the
# distance calculation over genuinely continuous financial signals like
# income or DTI -- producing clusters that mostly reproduce a categorical
# variable's own categories rather than revealing new financial-behavior
# groupings. Categorical columns (home_ownership, purpose, verification
# status, etc.) are still fully used, just downstream in PROFILING
# (segment_profiles.py) to describe what a cluster looks like, rather
# than upstream in defining cluster membership itself.
CLUSTERING_NUMERIC_FEATURES: List[str] = list(NUMERIC_FEATURES)
CLUSTERING_ORDINAL_FEATURES: List[str] = list(ORDINAL_CATEGORICAL_FEATURES)  # i.e. ["grade"]

# Raw categorical columns retained for cluster PROFILING/business-label
# assignment (mode / distribution per cluster) even though they don't
# drive clustering distance directly.
CLUSTERING_PROFILE_CATEGORICAL_FEATURES: List[str] = list(ONEHOT_CATEGORICAL_FEATURES)

# IQR-based outlier clipping applied before scaling. K-Means (and, to a
# lesser extent, hierarchical/Euclidean-based methods) is sensitive to
# extreme values since a single outlier can pull a centroid noticeably;
# clipping (winsorizing) rather than dropping preserves every borrower's
# ranking/segment membership while preventing a handful of extreme rows
# from distorting cluster boundaries for everyone else.
OUTLIER_IQR_MULTIPLIER: float = 3.0

# Numeric features to winsorize for regression-ready data.
WINSORIZE_FEATURES: List[str] = ["annual_inc", "dti"]
WINSORIZE_IQR_MULTIPLIER: float = 3.0
WINSORIZED_DATA_FILENAME: str = "lendingclub_indiana_winsorized.csv"
WINSORIZED_DATA_PATH: Path = PROCESSED_DATA_DIR / WINSORIZED_DATA_FILENAME

# Candidate cluster counts evaluated by the optimal-k analysis (elbow,
# silhouette, Calinski-Harabasz, Davies-Bouldin). Capped at 8: beyond
# that, borrower segments become too numerous for the business
# applications this phase targets (marketing campaigns, underwriting
# policy tiers) to act on distinctly.
N_CLUSTERS_CANDIDATES: List[int] = [2, 3, 4, 5, 6]

# Default clustering algorithm and cluster count used by
# `SegmentationEngine` unless overridden -- see the Phase 4B notebook's
# "Optimal Number of Clusters" and "Clustering Algorithms" sections for
# the comparative analysis that justifies these defaults.
DEFAULT_CLUSTERING_ALGORITHM: str = "kmeans"
DEFAULT_N_CLUSTERS: int = 4

# Sample size for t-SNE/UMAP visualization. Both are O(n log n) to O(n^2)
# depending on implementation and become slow/memory-heavy well before
# this project's full dataset size; a random sample is representative
# enough for a 2D visual sanity-check without a multi-minute wait in an
# interactive notebook or Streamlit session.
DIMENSIONALITY_REDUCTION_SAMPLE_SIZE: int = 500

# --- Phase 4B artifact paths (all under SEGMENTATION_DIR) ---
PCA_MODEL_PATH: Path = SEGMENTATION_DIR / "pca_model.joblib"
CLUSTER_CENTROIDS_PATH: Path = SEGMENTATION_DIR / "cluster_centroids.joblib"
SEGMENT_DEFINITIONS_PATH: Path = SEGMENTATION_DIR / "segment_definitions.joblib"
CLUSTER_METADATA_PATH: Path = SEGMENTATION_DIR / "cluster_metadata.joblib"
SEGMENT_PROFILES_PATH: Path = SEGMENTATION_DIR / "segment_profiles.joblib"
OPTIMAL_K_ANALYSIS_PATH: Path = SEGMENTATION_DIR / "optimal_k_analysis.joblib"
CLUSTERING_PREPROCESSOR_PATH: Path = SEGMENTATION_DIR / "clustering_preprocessor.joblib"
