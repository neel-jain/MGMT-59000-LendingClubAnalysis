"""
test_integration.py
=====================
Phase 6 integration test suite.

Unlike the per-module unit tests in the rest of `tests/`, these tests
deliberately exercise the SEAMS between components -- verifying that
the preprocessing pipeline, serialized models, `RiskScoringEngine`,
`ExplainabilityEngine`, `SegmentationEngine`, configuration files, and
data loading all communicate correctly end-to-end, using the real
artifacts on disk (not mocks). This directly covers the Phase 6
"Application Integration" checklist:

    - Preprocessing Pipeline
    - Serialized Models
    - RiskScoringEngine
    - ExplainabilityEngine
    - SegmentationEngine
    - Configuration Files
    - Logging
    - Data Loading

Streamlit page navigation/integration is covered separately in
`tests/test_app.py`.

Run with:
    pytest tests/test_integration.py -v
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, utils  # noqa: E402
from src.configurable_thresholds import load_threshold_config  # noqa: E402
from src.explainability import ExplainabilityEngine  # noqa: E402
from src.risk_scoring import RiskScoringEngine  # noqa: E402
from src.segmentation_engine import SegmentationEngine  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return utils.load_splits()


@pytest.fixture(scope="module")
def sample_borrower(splits):
    _, _, X_test, _, _, _ = splits
    return X_test.iloc[[0]]


# ---------------------------------------------------------------------------
# Data loading -> preprocessing pipeline -> serialized models
# ---------------------------------------------------------------------------


def test_data_loading_produces_consistent_schema(splits):
    """Every split must share the same raw feature schema (no drift between train/val/test)."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_serialized_preprocessing_pipeline_loads_and_transforms(splits):
    """The Phase 1 serialized preprocessor must load and transform new data without refitting."""
    X_train, _, _, _, _, _ = splits
    preprocessor = utils.load_object(config.PREPROCESSOR_PATH)
    transformed = preprocessor.transform(X_train.head(20))
    assert transformed.shape[0] == 20
    assert not np.isnan(transformed).any()


@pytest.mark.parametrize("model_key", ["logistic_regression", "random_forest", "xgboost"])
def test_serialized_model_loads_and_predicts(model_key, sample_borrower):
    """Every Phase 3 serialized model must load and produce a valid probability without retraining."""
    model_path = {
        "logistic_regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
        "random_forest": config.RANDOM_FOREST_MODEL_PATH,
        "xgboost": config.XGBOOST_MODEL_PATH,
    }[model_key]
    pipeline = utils.load_object(model_path)
    proba = pipeline.predict_proba(sample_borrower)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


def test_clustering_model_loads_and_transforms(splits):
    """The Phase 4B serialized clustering model + preprocessor must load and assign a cluster label."""
    X_train, _, _, _, _, _ = splits
    from src import cluster_analysis as ca

    clustering_preprocessor = utils.load_object(config.CLUSTERING_PREPROCESSOR_PATH)
    clustering_model = utils.load_object(config.CLUSTERING_MODEL_PATH)
    clipped = ca.clip_outliers(X_train.head(20))
    transformed = clustering_preprocessor.transform(clipped)
    labels = clustering_model.predict(transformed)
    assert len(labels) == 20


# ---------------------------------------------------------------------------
# RiskScoringEngine <-> ExplainabilityEngine <-> SegmentationEngine
# ---------------------------------------------------------------------------


def test_risk_scoring_engine_end_to_end(sample_borrower):
    """RiskScoringEngine must load the production model and produce a complete prediction summary."""
    engine = RiskScoringEngine()
    summary = engine.generate_prediction_summary(sample_borrower)
    assert 0.0 <= summary.default_probability <= 1.0
    assert summary.risk_tier in {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}


def test_explainability_engine_end_to_end(sample_borrower):
    """ExplainabilityEngine must load the production model and produce a complete local explanation."""
    engine = ExplainabilityEngine()
    local = engine.explain_prediction(sample_borrower)
    assert isinstance(local.business_summary, str) and len(local.business_summary) > 0


def test_explainability_engine_uses_same_probability_as_risk_scoring_engine(sample_borrower):
    """The two Phase 4A engines must agree on the same borrower's predicted probability -- they wrap the same production model."""
    risk_engine = RiskScoringEngine()
    explain_engine = ExplainabilityEngine()
    risk_proba = risk_engine.predict_probability(sample_borrower)[0]
    explain_proba = explain_engine.risk_scoring_engine.predict_probability(sample_borrower)[0]
    assert risk_proba == pytest.approx(explain_proba)


def test_segmentation_engine_end_to_end(splits):
    """SegmentationEngine must fit on the training split and assign every borrower to a named segment."""
    X_train, _, X_test, y_train, _, _ = splits
    engine = SegmentationEngine()
    engine.fit(X_train, default_flags=y_train)
    segments = engine.assign_segment(X_test.head(10))
    assert len(segments) == 10
    assert set(segments) <= set(engine.fit_result.segment_names.values())


def test_segmentation_engine_cross_references_risk_scoring_engine(splits):
    """SegmentationEngine.compare_with_supervised_models() must successfully call into RiskScoringEngine internally."""
    X_train, _, _, y_train, _, _ = splits
    engine = SegmentationEngine()
    engine.fit(X_train, default_flags=y_train)
    comparison = engine.compare_with_supervised_models()
    assert "mean_predicted_probability" in comparison.columns
    assert (comparison["mean_predicted_probability"] >= 0).all()
    assert (comparison["mean_predicted_probability"] <= 1).all()


def test_all_three_engines_agree_on_same_production_model_key(sample_borrower, splits):
    """RiskScoringEngine, ExplainabilityEngine, and SegmentationEngine's internal RiskScoringEngine must all default to config.PRODUCTION_MODEL_KEY."""
    X_train, _, _, y_train, _, _ = splits
    risk_engine = RiskScoringEngine()
    explain_engine = ExplainabilityEngine()
    segmentation_engine = SegmentationEngine()
    segmentation_engine.fit(X_train, default_flags=y_train)

    assert risk_engine.model_key == config.PRODUCTION_MODEL_KEY
    assert explain_engine.model_key == config.PRODUCTION_MODEL_KEY
    assert segmentation_engine.risk_scoring_engine.model_key == config.PRODUCTION_MODEL_KEY


# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------


def test_risk_threshold_config_loads_and_validates():
    """The JSON-backed risk threshold configuration must load and pass its own validation."""
    threshold_config = load_threshold_config()
    threshold_config.validate()  # should not raise
    assert threshold_config.get_tier(0.5) in {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}


def test_config_paths_are_all_absolute_and_under_project_root():
    """Every path in config.py must be an absolute path rooted under PROJECT_ROOT -- no hardcoded external paths."""
    path_attrs = [name for name in dir(config) if name.endswith("_PATH") or name.endswith("_DIR")]
    assert len(path_attrs) > 5  # sanity check that we actually found path constants
    for name in path_attrs:
        path_value = getattr(config, name)
        assert isinstance(path_value, Path), f"{name} is not a Path instance"
        assert path_value.is_absolute(), f"{name} is not an absolute path"
        assert str(config.PROJECT_ROOT) in str(path_value), f"{name} is not rooted under PROJECT_ROOT"


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------


def test_get_logger_returns_configured_logger_with_handlers():
    logger = utils.get_logger("test_integration_logger")
    assert len(logger.handlers) >= 2  # console + file handler
    assert logger.level == config.LOG_LEVEL


def test_get_logger_is_idempotent_no_duplicate_handlers():
    """Calling get_logger twice for the same name must not attach duplicate handlers."""
    logger1 = utils.get_logger("test_idempotent_logger")
    handler_count_1 = len(logger1.handlers)
    logger2 = utils.get_logger("test_idempotent_logger")
    handler_count_2 = len(logger2.handlers)
    assert handler_count_1 == handler_count_2


def test_pipeline_log_file_exists_after_running_pipeline():
    """python -m src.train_models must produce a readable pipeline.log."""
    assert config.PIPELINE_LOG_PATH.exists()


# ---------------------------------------------------------------------------
# Full cross-phase workflow smoke test
# ---------------------------------------------------------------------------


def test_full_workflow_load_predict_explain_segment(splits):
    """
    The complete Phase 6 'functional testing' workflow in one test:
    load dataset -> generate prediction -> generate SHAP explanation ->
    assign borrower segment -> generate executive summary -> export report.
    """
    X_train, _, X_test, y_train, _, _ = splits
    borrower = X_test.iloc[[0]]

    # Load dataset
    cleaned = utils.load_dataframe(config.CLEANED_DATA_PATH)
    assert not cleaned.empty

    # Generate prediction
    risk_engine = RiskScoringEngine()
    summary = risk_engine.generate_prediction_summary(borrower)
    assert summary.risk_tier

    # Generate SHAP explanation
    explain_engine = ExplainabilityEngine()
    local = explain_engine.explain_prediction(borrower)
    assert local.business_summary

    # Assign borrower segment
    segmentation_engine = SegmentationEngine()
    segmentation_engine.fit(X_train, default_flags=y_train)
    segment = segmentation_engine.assign_segment(borrower).iloc[0]
    assert segment in segmentation_engine.fit_result.segment_names.values()

    # Generate executive summary (global)
    global_explanation = explain_engine.explain_global_model(X_test.head(50))
    assert global_explanation.business_summary

    # Export reports
    risk_report = risk_engine.export_prediction_report(borrower)
    explanation_report = explain_engine.export_borrower_explanation_report(borrower)
    segment_report = segmentation_engine.export_segment_summary()
    assert "Risk Assessment" in risk_report.to_markdown()
    assert "Top Risk Factors" in explanation_report.to_markdown()
    assert "Executive Summary" in segment_report.to_markdown()
