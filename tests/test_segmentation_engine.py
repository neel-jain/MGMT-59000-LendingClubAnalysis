"""
test_segmentation_engine.py
==============================
Unit tests for src/segmentation_engine.py.

Uses the real Phase 1 splits (regenerated via the synthetic-fixture
pipeline) and the real Phase 3/4A production model artifacts, via
`SegmentationEngine`'s default disk-loading path for the shared
`RiskScoringEngine`.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, utils  # noqa: E402
from src.segmentation_engine import SegmentationEngine  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return utils.load_splits()


@pytest.fixture
def fitted_engine(splits):
    X_train, _, _, y_train, _, _ = splits
    engine = SegmentationEngine(n_clusters=4, algorithm="kmeans")
    engine.fit(X_train, default_flags=y_train)
    return engine


# ---------------------------------------------------------------------------
# Construction / fitting
# ---------------------------------------------------------------------------


def test_construction_invalid_algorithm_raises():
    with pytest.raises(ValueError):
        SegmentationEngine(algorithm="not_a_real_algorithm")


def test_methods_raise_before_fit():
    engine = SegmentationEngine()
    with pytest.raises(RuntimeError):
        engine.compare_segments()


def test_fit_returns_self(splits):
    X_train, _, _, y_train, _, _ = splits
    engine = SegmentationEngine(n_clusters=3)
    result = engine.fit(X_train, default_flags=y_train)
    assert result is engine


def test_fit_produces_expected_number_of_segments(fitted_engine):
    assert len(fitted_engine.fit_result.segment_names) == 4


def test_fit_auto_select_k(splits):
    X_train, _, _, y_train, _, _ = splits
    engine = SegmentationEngine()
    engine.fit(X_train, default_flags=y_train, auto_select_k=True, k_candidates=[2, 3, 4])
    assert engine.n_clusters in [2, 3, 4]


# ---------------------------------------------------------------------------
# Prediction / assignment
# ---------------------------------------------------------------------------


def test_predict_cluster_kmeans(fitted_engine, splits):
    _, _, X_test, _, _, _ = splits
    labels = fitted_engine.predict_cluster(X_test.head(10))
    assert len(labels) == 10
    assert set(labels) <= set(fitted_engine.fit_result.segment_names.keys())


def test_predict_cluster_agglomerative_raises_not_implemented(splits):
    X_train, _, X_test, y_train, _, _ = splits
    engine = SegmentationEngine(n_clusters=3, algorithm="agglomerative")
    engine.fit(X_train, default_flags=y_train)
    with pytest.raises(NotImplementedError):
        engine.predict_cluster(X_test.head(5))


def test_assign_segment_works_for_agglomerative(splits):
    """assign_segment (nearest-centroid) must work even for algorithms without .predict()."""
    X_train, _, X_test, y_train, _, _ = splits
    engine = SegmentationEngine(n_clusters=3, algorithm="agglomerative")
    engine.fit(X_train, default_flags=y_train)
    segments = engine.assign_segment(X_test.head(5))
    assert len(segments) == 5
    assert set(segments) <= set(engine.fit_result.segment_names.values())


def test_assign_segment_returns_business_names(fitted_engine, splits):
    _, _, X_test, _, _, _ = splits
    segments = fitted_engine.assign_segment(X_test.head(10))
    assert all(isinstance(s, str) for s in segments)
    assert set(segments) <= set(fitted_engine.fit_result.segment_names.values())


# ---------------------------------------------------------------------------
# Profiling / description
# ---------------------------------------------------------------------------


def test_describe_segment_returns_text(fitted_engine):
    text = fitted_engine.describe_segment(0)
    assert isinstance(text, str) and len(text) > 0


def test_describe_segment_unknown_cluster_raises(fitted_engine):
    with pytest.raises(ValueError):
        fitted_engine.describe_segment(999)


def test_generate_cluster_profile_returns_dataframe(fitted_engine):
    profile = fitted_engine.generate_cluster_profile()
    assert isinstance(profile, pd.DataFrame)
    assert len(profile) == 4


def test_compare_segments_returns_expected_columns(fitted_engine):
    comparison = fitted_engine.compare_segments()
    expected_cols = {
        "segment_name", "n_borrowers", "typical_income", "typical_interest_rate",
        "typical_loan_grade", "typical_dti", "typical_employment_length",
        "average_default_rate", "risk_tier",
    }
    assert expected_cols <= set(comparison.columns)


def test_recommend_business_actions_single_segment(fitted_engine):
    recommendation = fitted_engine.recommend_business_actions(0)
    assert recommendation.cluster_id == 0


def test_recommend_business_actions_all_segments(fitted_engine):
    all_recs = fitted_engine.recommend_business_actions()
    assert len(all_recs) == 4


def test_recommend_business_actions_unknown_cluster_raises(fitted_engine):
    with pytest.raises(ValueError):
        fitted_engine.recommend_business_actions(999)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def test_visualize_clusters_pca(fitted_engine):
    fig = fitted_engine.visualize_clusters(method="pca")
    assert fig is not None


def test_visualize_clusters_invalid_method_raises(fitted_engine):
    with pytest.raises(ValueError):
        fitted_engine.visualize_clusters(method="not_a_method")


# ---------------------------------------------------------------------------
# Relationship to supervised models
# ---------------------------------------------------------------------------


def test_compare_with_supervised_models_shape(fitted_engine):
    comparison = fitted_engine.compare_with_supervised_models()
    assert "mean_predicted_probability" in comparison.columns
    assert len(comparison) == 4
    assert (comparison["mean_predicted_probability"] >= 0).all()
    assert (comparison["mean_predicted_probability"] <= 1).all()


# ---------------------------------------------------------------------------
# Export + persistence
# ---------------------------------------------------------------------------


def test_export_segment_summary(fitted_engine):
    report = fitted_engine.export_segment_summary()
    md = report.to_markdown()
    assert "Executive Summary" in md
    assert "Segment Comparison" in md


def test_persist_segmentation_artifacts_writes_all_files(fitted_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLUSTERING_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "CLUSTERING_PREPROCESSOR_PATH", tmp_path / "preprocessor.joblib")
    monkeypatch.setattr(config, "CLUSTER_CENTROIDS_PATH", tmp_path / "centroids.joblib")
    monkeypatch.setattr(config, "SEGMENT_DEFINITIONS_PATH", tmp_path / "definitions.joblib")
    monkeypatch.setattr(config, "CLUSTER_METADATA_PATH", tmp_path / "metadata.joblib")
    monkeypatch.setattr(config, "SEGMENT_PROFILES_PATH", tmp_path / "profiles.joblib")
    monkeypatch.setattr(config, "OPTIMAL_K_ANALYSIS_PATH", tmp_path / "optimal_k.joblib")

    fitted_engine.persist_segmentation_artifacts()

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "preprocessor.joblib").exists()
    assert (tmp_path / "centroids.joblib").exists()
    assert (tmp_path / "definitions.joblib").exists()
    assert (tmp_path / "metadata.joblib").exists()
    assert (tmp_path / "profiles.joblib").exists()
    assert (tmp_path / "optimal_k.joblib").exists()
