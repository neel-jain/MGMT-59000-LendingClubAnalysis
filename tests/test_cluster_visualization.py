"""
test_cluster_visualization.py
================================
Unit tests for src/cluster_visualization.py.

Plotting functions are smoke-tested (do they run and return a Figure
without raising) -- visual correctness is verified manually via the
executed notebook.

Run with:
    pytest tests/ -v
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cluster_visualization as cv  # noqa: E402


@pytest.fixture
def clustered_data():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 30000, n),
        "int_rate": rng.uniform(5, 30, n),
        "annual_inc": rng.lognormal(10.8, 0.4, n),
        "dti": rng.uniform(0, 40, n),
    })
    labels = rng.integers(0, 4, n)
    return df, labels


def test_cluster_color_cycles_for_high_ids():
    color_0 = cv._cluster_color(0)
    color_wrapped = cv._cluster_color(len(cv.CLUSTER_PALETTE))
    assert color_0 == color_wrapped


def test_plot_dimensionality_reduction_scatter_runs(clustered_data):
    df, labels = clustered_data
    coordinates = np.column_stack([df["loan_amnt"], df["int_rate"]])
    fig = cv.plot_dimensionality_reduction_scatter(coordinates, labels, "PCA")
    assert fig is not None


def test_plot_dimensionality_reduction_scatter_with_segment_names(clustered_data):
    df, labels = clustered_data
    coordinates = np.column_stack([df["loan_amnt"], df["int_rate"]])
    names = {0: "Prime Borrowers", 1: "High Risk Borrowers", 2: "Moderate Risk", 3: "Credit Rebuilders"}
    fig = cv.plot_dimensionality_reduction_scatter(coordinates, labels, "PCA", segment_names=names)
    assert fig is not None


def test_plot_dimensionality_reduction_scatter_handles_noise_label(clustered_data):
    df, labels = clustered_data
    labels_with_noise = labels.copy()
    labels_with_noise[0] = -1
    coordinates = np.column_stack([df["loan_amnt"], df["int_rate"]])
    fig = cv.plot_dimensionality_reduction_scatter(coordinates, labels_with_noise, "PCA")
    assert fig is not None


def test_plot_dimensionality_reduction_comparison_runs(clustered_data):
    df, labels = clustered_data
    coordinates = np.column_stack([df["loan_amnt"], df["int_rate"]])
    fig = cv.plot_dimensionality_reduction_comparison({"PCA": coordinates, "t-SNE": coordinates}, labels)
    assert fig is not None


def test_plot_optimal_k_analysis_runs():
    table = pd.DataFrame({
        "n_clusters": [2, 3, 4, 5],
        "inertia": [100, 80, 65, 55],
        "silhouette_score": [0.3, 0.35, 0.4, 0.38],
        "calinski_harabasz_score": [50, 60, 65, 62],
        "davies_bouldin_score": [1.2, 1.0, 0.9, 0.95],
    })
    fig = cv.plot_optimal_k_analysis(table, recommended_k=4)
    assert fig is not None


def test_plot_cluster_heatmap_runs(clustered_data):
    df, labels = clustered_data
    df = df.copy()
    df["cluster"] = labels
    fig = cv.plot_cluster_heatmap(df, ["loan_amnt", "int_rate", "annual_inc", "dti"])
    assert fig is not None


def test_plot_parallel_coordinates_runs(clustered_data):
    df, labels = clustered_data
    df = df.copy()
    df["cluster"] = labels
    fig = cv.plot_parallel_coordinates(df, ["loan_amnt", "int_rate", "annual_inc", "dti"])
    assert fig is not None


def test_plot_radar_chart_runs(clustered_data):
    df, labels = clustered_data
    df = df.copy()
    df["cluster"] = labels
    fig = cv.plot_radar_chart(df, ["loan_amnt", "int_rate", "annual_inc", "dti"])
    assert fig is not None


def test_plot_cluster_size_distribution_runs(clustered_data):
    df, labels = clustered_data
    fig = cv.plot_cluster_size_distribution(labels)
    assert fig is not None


def test_plot_feature_by_cluster_runs(clustered_data):
    df, labels = clustered_data
    df = df.copy()
    df["cluster"] = labels
    fig = cv.plot_feature_by_cluster(df, "annual_inc", feature_label="Annual Income")
    assert fig is not None


def test_plot_feature_distribution_by_cluster_runs(clustered_data):
    df, labels = clustered_data
    df = df.copy()
    df["cluster"] = labels
    fig = cv.plot_feature_distribution_by_cluster(df, "dti", feature_label="DTI")
    assert fig is not None
