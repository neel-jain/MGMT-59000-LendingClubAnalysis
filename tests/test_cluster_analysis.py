"""
test_cluster_analysis.py
==========================
Unit tests for src/cluster_analysis.py.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cluster_analysis as ca  # noqa: E402


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 30000, n),
        "int_rate": rng.uniform(5, 30, n),
        "installment": rng.uniform(30, 1000, n),
        "annual_inc": rng.lognormal(10.8, 0.4, n),
        "dti": rng.uniform(0, 40, n),
        "delinq_2yrs": rng.poisson(0.3, n),
        "open_acc": rng.integers(1, 25, n),
        "pub_rec": rng.poisson(0.15, n),
        "revol_bal": rng.uniform(0, 60000, n),
        "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.integers(2, 60, n),
        "mort_acc": rng.integers(0, 8, n),
        "pub_rec_bankruptcies": rng.poisson(0.08, n),
        "emp_length_years": rng.uniform(0, 10, n),
        "grade": rng.choice(list("ABCDEFG"), size=n),
    })


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def test_clip_outliers_reduces_extreme_values():
    df = pd.DataFrame({"loan_amnt": [1000, 2000, 3000, 4000, 1_000_000]})
    clipped = ca.clip_outliers(df, columns=["loan_amnt"], iqr_multiplier=1.5)
    assert clipped["loan_amnt"].max() < 1_000_000


def test_clip_outliers_preserves_row_count(sample_df):
    clipped = ca.clip_outliers(sample_df)
    assert len(clipped) == len(sample_df)


def test_clip_outliers_missing_column_logs_and_skips(sample_df):
    result = ca.clip_outliers(sample_df, columns=["nonexistent_column"])
    assert len(result) == len(sample_df)


def test_build_clustering_preprocessor_fits_and_transforms(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    transformed = preprocessor.fit_transform(sample_df)
    assert transformed.shape[0] == len(sample_df)
    assert not np.isnan(transformed).any()


def test_build_clustering_preprocessor_excludes_onehot_features(sample_df):
    df_with_extra = sample_df.copy()
    df_with_extra["purpose"] = "debt_consolidation"
    preprocessor = ca.build_clustering_preprocessor()
    transformed = preprocessor.fit_transform(df_with_extra)
    # Should equal numeric (14) + ordinal grade (1) = 15 columns, ignoring 'purpose'.
    assert transformed.shape[1] == 15


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


def test_fit_pca_shape_and_variance(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_pca(X, n_components=2)
    assert result.coordinates.shape == (len(sample_df), 2)
    assert result.explained_variance_ratio is not None
    assert len(result.explained_variance_ratio) == 2


def test_fit_tsne_shape(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_tsne(X[:100])
    assert result.coordinates.shape == (100, 2)
    assert result.model is None


def test_fit_umap_shape_or_none(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_umap(X[:100])
    if result is not None:  # umap-learn may not be installed in all environments
        assert result.coordinates.shape == (100, 2)


# ---------------------------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------------------------


def test_fit_kmeans_returns_expected_n_clusters(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_kmeans(X, n_clusters=4)
    assert result.n_clusters == 4
    assert len(set(result.labels)) == 4


def test_fit_agglomerative_returns_expected_n_clusters(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_agglomerative(X, n_clusters=3)
    assert result.n_clusters == 3
    assert len(set(result.labels)) == 3


def test_fit_gaussian_mixture_returns_expected_n_clusters(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_gaussian_mixture(X, n_clusters=3)
    assert result.n_clusters == 3


def test_fit_dbscan_excludes_noise_from_cluster_count(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_dbscan(X, eps=0.5, min_samples=5)
    assert result.n_clusters == len(set(result.labels) - {-1})


def test_clustering_algorithms_registry_contains_expected_keys():
    assert set(ca.CLUSTERING_ALGORITHMS.keys()) == {"kmeans", "agglomerative", "gaussian_mixture"}


# ---------------------------------------------------------------------------
# Cluster validity metrics + optimal-k
# ---------------------------------------------------------------------------


def test_evaluate_clustering_returns_expected_keys(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    result = ca.fit_kmeans(X, n_clusters=3)
    metrics = ca.evaluate_clustering(X, result.labels)
    assert set(metrics.keys()) == {"silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"}


def test_evaluate_clustering_single_cluster_returns_nan():
    X = np.random.default_rng(0).normal(size=(50, 3))
    labels = np.zeros(50, dtype=int)
    metrics = ca.evaluate_clustering(X, labels)
    assert all(np.isnan(v) for v in metrics.values())


def test_evaluate_optimal_k_shape(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    table = ca.evaluate_optimal_k(X, k_candidates=[2, 3, 4])
    assert len(table) == 3
    assert {"n_clusters", "inertia", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"} <= set(table.columns)
    # Inertia should decrease monotonically as k increases (within-cluster variance shrinks).
    assert table.sort_values("n_clusters")["inertia"].is_monotonic_decreasing


def test_recommend_optimal_k_returns_valid_candidate(sample_df):
    preprocessor = ca.build_clustering_preprocessor()
    X = preprocessor.fit_transform(sample_df)
    table = ca.evaluate_optimal_k(X, k_candidates=[2, 3, 4, 5])
    best_k, explanation = ca.recommend_optimal_k(table)
    assert best_k in [2, 3, 4, 5]
    assert isinstance(explanation, str) and len(explanation) > 0
