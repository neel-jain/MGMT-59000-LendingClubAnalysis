"""
cluster_visualization.py
==========================
Phase 4B reusable module: executive-quality plotting functions for
borrower segmentation -- dimensionality-reduction scatter plots, cluster
heatmaps, parallel-coordinate and radar charts, and per-cluster business
metric comparisons.

Every function returns a `matplotlib.figure.Figure` (never calls
`plt.show()`), following the same convention as `eda_utils.py` and
`model_utils.py`, so a future Streamlit page can call `st.pyplot(fig)`
directly. Visual style (colors, figure sizes, title/subtitle layout)
reuses `eda_utils.py`'s shared constants for a consistent look across
every notebook in this project.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from src import labels, utils
from src.eda_utils import FIGSIZE_STANDARD, FIGSIZE_WIDE, PALETTE_DIVERGING, _apply_titles

logger = utils.get_logger(__name__)

# A consistent, distinguishable color per cluster index, reused across
# every visualization in this module so "Cluster 2" always means the
# same color throughout a single notebook run.
CLUSTER_PALETTE: List[str] = [
    "#2E86AB", "#C0392B", "#27AE60", "#F39C12",
    "#8E44AD", "#16A085", "#D35400", "#7F8C8D",
]


def _cluster_color(cluster_id: int) -> str:
    """Look up this cluster's consistent color, cycling the palette if there are more clusters than colors."""
    return CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]


def _cluster_labels_for_legend(labels: np.ndarray, segment_names: Optional[Dict[int, str]]) -> List[str]:
    """Build legend labels, using business segment names if provided, else 'Cluster N'."""
    unique = sorted(set(int(l) for l in labels if l != -1))
    if segment_names:
        return [segment_names.get(c, f"Cluster {c}") for c in unique]
    return [f"Cluster {c}" for c in unique]


# ---------------------------------------------------------------------------
# Dimensionality-reduction scatter plots
# ---------------------------------------------------------------------------


def plot_dimensionality_reduction_scatter(
    coordinates: np.ndarray, labels: np.ndarray, method_name: str,
    segment_names: Optional[Dict[int, str]] = None, title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> plt.Figure:
    """
    2D scatter plot of any dimensionality-reduction output (PCA, t-SNE,
    or UMAP coordinates), colored by cluster assignment.

    Parameters
    ----------
    coordinates : np.ndarray
        (n_samples, 2) array, e.g. `DimensionalityReductionResult.coordinates`.
    labels : np.ndarray
        Cluster label per row, aligned with `coordinates`.
    method_name : str
        e.g. "PCA", "t-SNE", "UMAP" -- used in the default title/axis labels.
    segment_names : dict[int, str], optional
        Maps cluster id -> business segment name for the legend.
    title, subtitle : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    unique_labels = sorted(set(int(l) for l in labels))
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        color = "lightgray" if cluster_id == -1 else _cluster_color(cluster_id)
        label = "Noise" if cluster_id == -1 else (segment_names.get(cluster_id, f"Cluster {cluster_id}") if segment_names else f"Cluster {cluster_id}")
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], color=color, label=label, alpha=0.65, s=28, edgecolor="white", linewidth=0.3)

    ax.set_xlabel(f"{method_name} Component 1")
    ax.set_ylabel(f"{method_name} Component 2")
    ax.legend(frameon=False, loc="best", fontsize=9)
    _apply_titles(ax, title or f"Borrower Segments -- {method_name} Projection", subtitle)
    fig.tight_layout()
    return fig


def plot_dimensionality_reduction_comparison(
    results: Dict[str, np.ndarray], labels: np.ndarray, segment_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of multiple dimensionality-reduction methods
    (e.g. PCA vs. t-SNE vs. UMAP) on the same cluster labels, for the
    Phase 4B "compare methods, recommend the best visualization approach"
    requirement.

    Parameters
    ----------
    results : dict[str, np.ndarray]
        Method name -> (n_samples, 2) coordinate array. All arrays must
        correspond row-for-row to `labels`.
    labels : np.ndarray
    segment_names : dict[int, str], optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5.5 * n_methods, 5.5))
    if n_methods == 1:
        axes = [axes]

    unique_labels = sorted(set(int(l) for l in labels))
    for ax, (method_name, coordinates) in zip(axes, results.items()):
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            color = "lightgray" if cluster_id == -1 else _cluster_color(cluster_id)
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], color=color, alpha=0.65, s=22, edgecolor="white", linewidth=0.3)
        ax.set_title(method_name, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")

    legend_elements = [
        Patch(facecolor=_cluster_color(c), label=(segment_names.get(c, f"Cluster {c}") if segment_names else f"Cluster {c}"))
        for c in unique_labels if c != -1
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=min(len(legend_elements), 4), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Dimensionality Reduction Method Comparison", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Elbow / validity-metric plots
# ---------------------------------------------------------------------------


def plot_optimal_k_analysis(optimal_k_table: pd.DataFrame, recommended_k: int) -> plt.Figure:
    """
    Four-panel visualization of the optimal-k evaluation: elbow
    (inertia), silhouette score, Calinski-Harabasz index, and
    Davies-Bouldin index, each vs. candidate k, with the recommended k
    marked -- the full Phase 4B "Optimal Number of Clusters" comparison
    in one figure.

    Parameters
    ----------
    optimal_k_table : pd.DataFrame
        Output of `cluster_analysis.evaluate_optimal_k`.
    recommended_k : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = [
        ("inertia", "Inertia (Elbow Method)", "#2E86AB", "lower change = elbow"),
        ("silhouette_score", "Silhouette Score", "#27AE60", "higher is better"),
        ("calinski_harabasz_score", "Calinski-Harabasz Index", "#F39C12", "higher is better"),
        ("davies_bouldin_score", "Davies-Bouldin Index", "#C0392B", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, label, color, note) in zip(axes.ravel(), panels):
        ax.plot(optimal_k_table["n_clusters"], optimal_k_table[column], marker="o", color=color, linewidth=2.2)
        ax.axvline(recommended_k, color="black", linestyle=":", linewidth=1.3)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({note})", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        f"Optimal Number of Clusters (recommended k={recommended_k}, dotted line)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster heatmap / parallel coordinates / radar
# ---------------------------------------------------------------------------


def plot_cluster_heatmap(
    profile_table: pd.DataFrame, features: Sequence[str], cluster_column: str = "cluster",
    segment_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Heatmap of standardized (z-score) mean feature values per cluster --
    a compact way to see, at a glance, which features define each
    cluster relative to the overall population (red = above average,
    blue = below average).

    Parameters
    ----------
    profile_table : pd.DataFrame
        One row per borrower (or per-cluster means already computed --
        see `cluster_column`), containing `features` and `cluster_column`.
    features : sequence of str
    cluster_column : str
    segment_names : dict[int, str], optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cluster_means = profile_table.groupby(cluster_column)[list(features)].mean()
    z_scores = (cluster_means - cluster_means.mean()) / cluster_means.std(ddof=0)

    if segment_names:
        z_scores.index = [segment_names.get(c, f"Cluster {c}") for c in z_scores.index]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(features)), max(4, 0.7 * len(z_scores) + 1.5)))
    sns.heatmap(
        z_scores, annot=True, fmt=".2f", cmap=PALETTE_DIVERGING, center=0, ax=ax,
        xticklabels=[labels.column_label(f) for f in features],
        cbar_kws={"label": "Standardized value (z-score vs. overall population)"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    _apply_titles(ax, "Cluster Profile Heatmap", "Standardized mean feature value per segment (red = above average, blue = below)")
    fig.tight_layout()
    return fig


def plot_parallel_coordinates(
    profile_table: pd.DataFrame, features: Sequence[str], cluster_column: str = "cluster",
    segment_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Parallel coordinate plot of standardized per-cluster mean feature
    values -- each cluster is one colored line crossing every feature
    axis, making it easy to see which clusters are "high on everything,"
    "low on everything," or have a distinctive up/down profile.

    Parameters
    ----------
    profile_table : pd.DataFrame
    features : sequence of str
    cluster_column : str
    segment_names : dict[int, str], optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cluster_means = profile_table.groupby(cluster_column)[list(features)].mean()
    z_scores = (cluster_means - cluster_means.mean()) / cluster_means.std(ddof=0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x_positions = range(len(features))
    for cluster_id in z_scores.index:
        label = segment_names.get(cluster_id, f"Cluster {cluster_id}") if segment_names else f"Cluster {cluster_id}"
        ax.plot(x_positions, z_scores.loc[cluster_id, list(features)], marker="o", color=_cluster_color(int(cluster_id)), label=label, linewidth=2)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([labels.column_label(f) for f in features], rotation=30, ha="right")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Standardized value (z-score)")
    ax.legend(frameon=False, loc="best", fontsize=9)
    _apply_titles(ax, "Cluster Profiles -- Parallel Coordinates", "Each line is one segment's standardized average across features")
    fig.tight_layout()
    return fig


def plot_radar_chart(
    profile_table: pd.DataFrame, features: Sequence[str], cluster_column: str = "cluster",
    segment_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Radar (spider) chart of standardized per-cluster mean feature
    values -- one polygon per cluster, useful for a quick "shape"
    comparison of segments in an executive presentation.

    Parameters
    ----------
    profile_table : pd.DataFrame
    features : sequence of str
    cluster_column : str
    segment_names : dict[int, str], optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cluster_means = profile_table.groupby(cluster_column)[list(features)].mean()
    z_scores = (cluster_means - cluster_means.mean()) / cluster_means.std(ddof=0)

    n_vars = len(features)
    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    for cluster_id in z_scores.index:
        values = z_scores.loc[cluster_id, list(features)].tolist()
        values += values[:1]
        label = segment_names.get(cluster_id, f"Cluster {cluster_id}") if segment_names else f"Cluster {cluster_id}"
        ax.plot(angles, values, linewidth=2, label=label, color=_cluster_color(int(cluster_id)))
        ax.fill(angles, values, alpha=0.08, color=_cluster_color(int(cluster_id)))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([labels.column_label(f) for f in features], fontsize=9)
    ax.set_title("Cluster Profiles -- Radar Chart", fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster size / feature-distribution / business-metric bar charts
# ---------------------------------------------------------------------------


def plot_cluster_size_distribution(
    labels: np.ndarray, segment_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """Bar chart of borrower count per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(unique)
    unique, counts = unique[order], counts[order]
    names = [segment_names.get(int(c), f"Cluster {c}") if segment_names else f"Cluster {c}" for c in unique]
    colors = [_cluster_color(int(c)) if c != -1 else "lightgray" for c in unique]

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars = ax.bar(names, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of Borrowers")
    _apply_titles(ax, "Cluster Size Distribution", "Number of borrowers assigned to each segment")
    fig.tight_layout()
    return fig


def plot_feature_by_cluster(
    profile_table: pd.DataFrame, feature: str, cluster_column: str = "cluster",
    segment_names: Optional[Dict[int, str]] = None, agg: str = "mean",
    feature_label: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of one feature's average (or other aggregate) value per
    cluster -- the general-purpose building block behind the required
    "Average Default Rate/Income/Interest Rate/DTI by Cluster" charts.

    Parameters
    ----------
    profile_table : pd.DataFrame
    feature : str
        Column to aggregate.
    cluster_column : str
    segment_names : dict[int, str], optional
    agg : str
        Any pandas-groupby-compatible aggregation ("mean", "median", etc.).
    feature_label : str, optional
        Business-friendly label for the y-axis/title (defaults to `feature`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    grouped = profile_table.groupby(cluster_column)[feature].agg(agg).sort_index()
    names = [segment_names.get(int(c), f"Cluster {c}") if segment_names else f"Cluster {c}" for c in grouped.index]
    colors = [_cluster_color(int(c)) for c in grouped.index]
    label = feature_label or labels.column_label(feature)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars = ax.bar(names, grouped.values, color=colors)
    for bar, value in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"{agg.title()} {label}")
    _apply_titles(ax, f"{label} by Cluster", f"{agg.title()} {label.lower()} for borrowers in each segment")
    fig.tight_layout()
    return fig


def plot_feature_distribution_by_cluster(
    profile_table: pd.DataFrame, feature: str, cluster_column: str = "cluster",
    segment_names: Optional[Dict[int, str]] = None, feature_label: Optional[str] = None,
) -> plt.Figure:
    """
    Boxplot of one feature's full distribution (not just the mean) per
    cluster -- reveals within-cluster spread/overlap that a bar-of-means
    chart hides.

    Parameters
    ----------
    profile_table : pd.DataFrame
    feature : str
    cluster_column : str
    segment_names : dict[int, str], optional
    feature_label : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    label = feature_label or labels.column_label(feature)
    plot_df = profile_table[[cluster_column, feature]].copy()
    unique_clusters = sorted(plot_df[cluster_column].unique())
    plot_df["segment_label"] = plot_df[cluster_column].apply(
        lambda c: segment_names.get(int(c), f"Cluster {c}") if segment_names else f"Cluster {c}"
    )
    order = [segment_names.get(int(c), f"Cluster {c}") if segment_names else f"Cluster {c}" for c in unique_clusters]
    palette = {name: _cluster_color(int(c)) for c, name in zip(unique_clusters, order)}

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.boxplot(data=plot_df, x="segment_label", y=feature, order=order, hue="segment_label", palette=palette, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    _apply_titles(ax, f"{label} Distribution by Cluster", "Boxplot shows within-segment spread, not just the average")
    fig.tight_layout()
    return fig
