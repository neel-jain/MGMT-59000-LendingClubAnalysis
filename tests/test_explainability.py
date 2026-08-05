"""
test_explainability.py
========================
Unit tests for src/explainability.py.

Uses a freshly-fit, small Logistic Regression and Random Forest pipeline
(not the full Phase 3 hyperparameter search) as injected pipelines for
speed, since ExplainabilityEngine's constructor accepts a pre-fitted
Pipeline directly.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_utils, utils  # noqa: E402
from src.explainability import ExplainabilityEngine, DEFAULT_INTERACTION_PAIRS  # noqa: E402
from src.risk_scoring import RiskScoringEngine  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return utils.load_splits()


@pytest.fixture(scope="module")
def lr_pipeline(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    pipeline = model_utils.build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="module")
def rf_pipeline(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    pipeline = model_utils.build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture
def lr_engine(lr_pipeline, splits):
    X_train, _, _, y_train, _, _ = splits
    risk_engine = RiskScoringEngine(model_key="logistic_regression", pipeline=lr_pipeline)
    return ExplainabilityEngine(
        model_key="logistic_regression", pipeline=lr_pipeline,
        background_data=X_train.head(50), risk_scoring_engine=risk_engine,
    )


@pytest.fixture
def rf_engine(rf_pipeline, splits):
    X_train, _, _, y_train, _, _ = splits
    risk_engine = RiskScoringEngine(model_key="random_forest", pipeline=rf_pipeline)
    return ExplainabilityEngine(
        model_key="random_forest", pipeline=rf_pipeline,
        background_data=X_train.head(50), risk_scoring_engine=risk_engine,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_engine_construction_linear_explainer(lr_engine):
    import shap
    assert isinstance(lr_engine.explainer, shap.LinearExplainer)


def test_engine_construction_tree_explainer(rf_engine):
    import shap
    assert isinstance(rf_engine.explainer, shap.TreeExplainer)


def test_engine_construction_invalid_model_key_raises(lr_pipeline):
    with pytest.raises(ValueError):
        ExplainabilityEngine(model_key="not_a_model", pipeline=lr_pipeline)


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------


def test_explain_prediction_single_borrower(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    local = lr_engine.explain_prediction(X_test.iloc[[0]])
    assert 0.0 <= local.default_probability <= 1.0
    assert isinstance(local.business_summary, str) and len(local.business_summary) > 0
    assert "feature_label" in local.feature_contributions.columns
    assert "shap_value" in local.feature_contributions.columns


def test_explain_prediction_rejects_multi_row(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    with pytest.raises(ValueError):
        lr_engine.explain_prediction(X_test.head(3))


def test_explain_prediction_risk_and_protective_factors_are_disjoint(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    local = lr_engine.explain_prediction(X_test.iloc[[1]])
    assert set(local.top_risk_factors).isdisjoint(set(local.top_protective_factors))


def test_generate_waterfall_plot_runs(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_waterfall_plot(X_test.iloc[[0]])
    assert fig is not None


def test_generate_force_plot_runs(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_force_plot(X_test.iloc[[0]])
    assert fig is not None


def test_generate_waterfall_plot_tree_model(rf_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = rf_engine.generate_waterfall_plot(X_test.iloc[[0]])
    assert fig is not None


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------


def test_summarize_feature_importance_shape(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    table = lr_engine.summarize_feature_importance(X_test.head(30))
    assert "mean_abs_shap" in table.columns
    assert "feature_label" in table.columns
    assert (table["mean_abs_shap"] >= 0).all()


def test_explain_global_model(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    global_exp = lr_engine.explain_global_model(X_test.head(30))
    assert len(global_exp.top_features) > 0
    assert isinstance(global_exp.business_summary, str)


def test_generate_shap_summary_beeswarm(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_shap_summary(X_test.head(30), plot_type="beeswarm")
    assert fig is not None


def test_generate_shap_summary_bar(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_shap_summary(X_test.head(30), plot_type="bar")
    assert fig is not None


def test_generate_shap_summary_invalid_plot_type_raises(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    with pytest.raises(ValueError):
        lr_engine.generate_shap_summary(X_test.head(10), plot_type="invalid")


def test_generate_dependence_plot_numeric_feature(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_dependence_plot("dti", X_test.head(30), interaction_feature="int_rate")
    assert fig is not None


def test_generate_dependence_plot_rejects_onehot_feature(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    with pytest.raises(ValueError):
        lr_engine.generate_dependence_plot("purpose", X_test.head(10))


def test_generate_decision_plot_runs(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_decision_plot(X_test.head(10))
    assert fig is not None


# ---------------------------------------------------------------------------
# Feature interaction analysis
# ---------------------------------------------------------------------------


def test_analyze_feature_interactions_covers_all_default_pairs(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    results = lr_engine.analyze_feature_interactions(X_test.head(40))
    assert set(results.keys()) == set(DEFAULT_INTERACTION_PAIRS)
    for pair, result in results.items():
        assert result["figure"] is not None
        assert result["kind"] in {"shap_dependence", "heatmap"}
        assert len(result["interpretation"]) > 0


def test_analyze_feature_interactions_numeric_pair_uses_shap_dependence(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    results = lr_engine.analyze_feature_interactions(X_test.head(40), pairs=[("annual_inc", "dti")])
    assert results[("annual_inc", "dti")]["kind"] == "shap_dependence"


def test_analyze_feature_interactions_categorical_pair_uses_heatmap(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    results = lr_engine.analyze_feature_interactions(X_test.head(40), pairs=[("purpose", "grade")])
    assert results[("purpose", "grade")]["kind"] == "heatmap"


# ---------------------------------------------------------------------------
# Partial dependence / ICE
# ---------------------------------------------------------------------------


def test_generate_pdp_ice_plot_runs(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    fig = lr_engine.generate_pdp_ice_plot(X_test.head(30), features=["dti", "int_rate"], kind="both")
    assert fig is not None


def test_generate_pdp_ice_plot_rejects_unsupported_feature(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    with pytest.raises(ValueError):
        lr_engine.generate_pdp_ice_plot(X_test.head(10), features=["purpose"])


# ---------------------------------------------------------------------------
# Business summary dispatch + exports
# ---------------------------------------------------------------------------


def test_generate_business_summary_local_dispatch(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    summary = lr_engine.generate_business_summary(X_test.iloc[[0]])
    assert isinstance(summary, str) and len(summary) > 0


def test_generate_business_summary_global_dispatch(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    summary = lr_engine.generate_business_summary(None)
    assert isinstance(summary, str) and len(summary) > 0


def test_export_borrower_explanation_report(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    report = lr_engine.export_borrower_explanation_report(X_test.iloc[[0]])
    md = report.to_markdown()
    assert "Top Risk Factors" in md


def test_export_global_explanation_report(lr_engine, splits):
    _, _, X_test, _, _, _ = splits
    report = lr_engine.export_global_explanation_report(X_test.head(30))
    md = report.to_markdown()
    assert "Most Influential Variables" in md


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def test_persist_explainability_artifacts_writes_all_files(lr_engine, splits, monkeypatch, tmp_path):
    from src import config

    _, _, X_test, _, _, y_test = splits

    monkeypatch.setattr(config, "SHAP_IMPORTANCE_PATH", tmp_path / "shap_importance.joblib")
    monkeypatch.setattr(config, "BUSINESS_SUMMARY_TEMPLATES_PATH", tmp_path / "templates.joblib")
    monkeypatch.setattr(config, "MODEL_METADATA_PATH", tmp_path / "metadata.joblib")
    monkeypatch.setattr(config, "FAIRNESS_REPORT_PATH", tmp_path / "fairness.joblib")
    monkeypatch.setattr(config, "FEATURE_INTERACTION_SUMMARY_PATH", tmp_path / "interactions.joblib")

    lr_engine.persist_explainability_artifacts(X=X_test.head(40), y=y_test.head(40))

    assert (tmp_path / "shap_importance.joblib").exists()
    assert (tmp_path / "templates.joblib").exists()
    assert (tmp_path / "metadata.joblib").exists()
    assert (tmp_path / "fairness.joblib").exists()
    assert (tmp_path / "interactions.joblib").exists()
