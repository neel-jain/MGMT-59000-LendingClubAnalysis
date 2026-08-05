"""
cluster_analysis.py
=====================
Phase 4B reusable module: data preparation, dimensionality reduction,
clustering algorithms, and optimal-cluster-count evaluation for borrower
segmentation.

This module holds the ANALYTICAL primitives (fit a scaler, fit a PCA,
fit a KMeans, score a set of cluster labels); `segmentation_engine.py`
composes these primitives into the single reusable `SegmentationEngine`
class that owns the end-to-end workflow and the artifacts it produces.
Keeping the primitives here (rather than inlined in the engine) makes
each one independently unit-testable and reusable outside the engine
(e.g. a one-off notebook experiment with a different algorithm).

Organized into sections:
    1. Data preparation (outlier clipping, clustering preprocessor)
    2. Dimensionality reduction (PCA, t-SNE, UMAP)
    3. Clustering algorithms (K-Means, Agglomerative, GMM, DBSCAN)
    4. Cluster validity metrics + optimal-k evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src import config, utils

logger = utils.get_logger(__name__)

try:
    import umap

    _UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed -- UMAP dimensionality reduction will be unavailable.")


# ---------------------------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------------------------


def clip_outliers(
    df: pd.DataFrame, columns: List[str] = config.CLUSTERING_NUMERIC_FEATURES,
    iqr_multiplier: float = config.OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """
    Winsorize (clip, not drop) numeric columns to
    [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR].

    Design decision: K-Means and hierarchical clustering both operate on
    Euclidean distance, where a single extreme value can pull a centroid
    or merge decision noticeably. Clipping is preferred over dropping
    outlier ROWS: every borrower keeps a segment assignment (needed for
    downstream risk-tier-by-segment reporting), while the clipped value
    still lands at the extreme end of that feature's distribution.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to clip. Defaults to `config.CLUSTERING_NUMERIC_FEATURES`.
    iqr_multiplier : float
        Defaults to `config.OUTLIER_IQR_MULTIPLIER` (3.0 -- a wider,
        more conservative multiplier than the classical 1.5 boxplot
        rule, since 1.5x IQR would flag a large share of financial data
        such as income or revolving balance as "outliers" when it is
        simply right-skewed, legitimate borrower variation).

    Returns
    -------
    pd.DataFrame
        Copy of `df` with the specified columns clipped in place.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Clustering column '%s' not found -- skipping outlier clipping.", col)
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_clipped:
            logger.info("Clipped %d outlier values in '%s' to [%.2f, %.2f].", n_clipped, col, lower, upper)
    return df


def build_clustering_preprocessor(
    numeric_features: List[str] = config.CLUSTERING_NUMERIC_FEATURES,
    ordinal_features: List[str] = config.CLUSTERING_ORDINAL_FEATURES,
    ordinal_categories: List[List[str]] = config.ORDINAL_CATEGORY_ORDER,
) -> ColumnTransformer:
    """
    Build (unfitted) the preprocessing pipeline used specifically for
    CLUSTERING -- distinct from, but consistent in style with,
    `utils.build_preprocessing_pipeline()` (same imputation/scaling
    strategy: median imputation, then standardization). The key
    difference is scope: this preprocessor only touches
    `CLUSTERING_NUMERIC_FEATURES` + `CLUSTERING_ORDINAL_FEATURES`,
    omitting one-hot categorical columns entirely (see the module-level
    rationale in `config.py`'s Phase 4B section).

    StandardScaler (not MinMaxScaler) is used so every feature
    contributes comparably to Euclidean distance regardless of its
    native units (dollars vs. years vs. a ratio) -- without scaling,
    `annual_inc` (tens of thousands) would dominate `dti` (a percentage)
    in any distance-based clustering algorithm.

    Parameters
    ----------
    numeric_features : list[str]
    ordinal_features : list[str]
    ordinal_categories : list[list[str]]

    Returns
    -------
    ColumnTransformer
        Unfitted. Call `.fit()` (or `.fit_transform()`) on the data used
        to fit the clustering model, then `.transform()` on any new data
        scored against that same fitted clustering model.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler()),  # scale the encoded grade too, for comparable distance contribution
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("ordinal", ordinal_pipeline, ordinal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    logger.info(
        "Built clustering preprocessor: %d numeric + %d ordinal features (categorical one-hot columns excluded).",
        len(numeric_features), len(ordinal_features),
    )
    return preprocessor


# ---------------------------------------------------------------------------
# 2. DIMENSIONALITY REDUCTION
# ---------------------------------------------------------------------------


@dataclass
class DimensionalityReductionResult:
    """Container for one dimensionality-reduction method's output."""

    method: str
    coordinates: np.ndarray  # (n_samples, n_components)
    model: object  # fitted transformer (None for methods without a reusable fitted object, e.g. t-SNE)
    explained_variance_ratio: Optional[np.ndarray] = None  # PCA only


def fit_pca(X: np.ndarray, n_components: int = 2, random_state: int = config.RANDOM_STATE) -> DimensionalityReductionResult:
    """
    Fit PCA for 2D visualization (and, optionally, as a lower-dimensional
    clustering input -- see the Phase 4B notebook's Dimensionality
    Reduction section for the comparison of "cluster on PCA components"
    vs. "cluster on the full standardized feature space").

    PCA is a LINEAR projection that preserves GLOBAL variance structure
    (which directions in feature space separate borrowers the most) and,
    unlike t-SNE/UMAP, is deterministic and has a reusable `.transform()`
    for new data -- the right tool when the axes themselves need a
    stable, interpretable meaning (loadings) across re-runs and new data.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed (scaled) feature matrix.
    n_components : int
    random_state : int

    Returns
    -------
    DimensionalityReductionResult
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coordinates = pca.fit_transform(X)
    logger.info(
        "Fitted PCA(n_components=%d): explained variance ratio = %s (total %.1f%%)",
        n_components, np.round(pca.explained_variance_ratio_, 3),
        pca.explained_variance_ratio_.sum() * 100,
    )
    return DimensionalityReductionResult(
        method="PCA", coordinates=coordinates, model=pca,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def fit_tsne(
    X: np.ndarray, n_components: int = 2, perplexity: float = 30.0, random_state: int = config.RANDOM_STATE,
) -> DimensionalityReductionResult:
    """
    Fit t-SNE for 2D visualization ONLY -- t-SNE has no `.transform()`
    for new data (each run re-embeds the full input jointly) and does
    NOT preserve global distances (only local neighborhood structure),
    so it is never used to feed the clustering algorithm itself, only to
    visually sanity-check whether clusters already found (e.g. via
    K-Means on the full feature space) look visually separated.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed (scaled) feature matrix. Caller should subsample to
        `config.DIMENSIONALITY_REDUCTION_SAMPLE_SIZE` rows first for
        interactive-speed performance (t-SNE is O(n^2) in the naive case).
    n_components : int
    perplexity : float
        Roughly, the effective number of near neighbors considered per
        point; must be less than the number of samples.
    random_state : int

    Returns
    -------
    DimensionalityReductionResult
        `model` is None (no reusable fitted transformer for new data).
    """
    effective_perplexity = min(perplexity, max(5.0, (len(X) - 1) / 3))
    tsne = TSNE(n_components=n_components, perplexity=effective_perplexity, random_state=random_state, init="pca")
    coordinates = tsne.fit_transform(X)
    logger.info("Fitted t-SNE on %d samples (perplexity=%.1f).", len(X), effective_perplexity)
    return DimensionalityReductionResult(method="t-SNE", coordinates=coordinates, model=None)


def fit_umap(
    X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
    random_state: int = config.RANDOM_STATE,
) -> Optional[DimensionalityReductionResult]:
    """
    Fit UMAP for 2D visualization. Unlike t-SNE, UMAP better preserves
    both local AND some global structure and DOES support
    `.transform()` on new data via the fitted model -- included as the
    third comparison point in the Phase 4B notebook's dimensionality-
    reduction comparison, and a viable candidate for the recommended
    visualization method if it best separates the clusters found.

    Parameters
    ----------
    X : np.ndarray
    n_components : int
    n_neighbors : int
    min_dist : float
    random_state : int

    Returns
    -------
    DimensionalityReductionResult or None
        None if `umap-learn` is not installed in the current environment
        (optional dependency -- see the module-level import guard).
    """
    if not _UMAP_AVAILABLE:
        logger.warning("Skipping UMAP: umap-learn is not installed.")
        return None
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist, random_state=random_state,
    )
    coordinates = reducer.fit_transform(X)
    logger.info("Fitted UMAP on %d samples (n_neighbors=%d, min_dist=%.2f).", len(X), n_neighbors, min_dist)
    return DimensionalityReductionResult(method="UMAP", coordinates=coordinates, model=reducer)


# ---------------------------------------------------------------------------
# 3. CLUSTERING ALGORITHMS
# ---------------------------------------------------------------------------


@dataclass
class ClusteringResult:
    """Container for one clustering algorithm's fitted output."""

    algorithm: str
    model: object
    labels: np.ndarray
    n_clusters: int  # actual number of clusters found (DBSCAN may differ from a requested k)


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = config.RANDOM_STATE) -> ClusteringResult:
    """
    Fit K-Means.

    Advantages: fast (near-linear in n_samples), scales well, produces
    convex/globular clusters that are easy to describe to a business
    audience ("borrowers near this centroid").
    Disadvantages: assumes roughly spherical, similarly-sized clusters;
    sensitive to feature scaling (mitigated by `build_clustering_preprocessor`)
    and outliers (mitigated by `clip_outliers`); requires `n_clusters`
    chosen in advance.
    Business applicability: the default choice here -- lending segments
    (income/DTI/rate tiers) are reasonably continuous and don't have the
    highly irregular, non-convex shapes that would favor DBSCAN.
    Computational considerations: O(n_samples x n_clusters x n_features)
    per iteration -- cheap at this project's scale.

    Parameters
    ----------
    X : np.ndarray
    n_clusters : int
    random_state : int

    Returns
    -------
    ClusteringResult
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return ClusteringResult(algorithm="K-Means", model=model, labels=labels, n_clusters=n_clusters)


def fit_agglomerative(X: np.ndarray, n_clusters: int, linkage: str = "ward") -> ClusteringResult:
    """
    Fit Agglomerative (hierarchical) Clustering.

    Advantages: no assumption of spherical clusters (with non-ward
    linkage), produces a dendrogram showing how clusters merge at every
    granularity (useful for justifying a chosen `n_clusters` visually),
    deterministic (no random initialization sensitivity as K-Means has).
    Disadvantages: O(n^2) memory/time in the naive case -- noticeably
    slower than K-Means at scale; no `.predict()` for new data (a fresh
    borrower requires refitting on the combined dataset, or falling back
    to nearest-centroid assignment).
    Business applicability: useful as a CROSS-CHECK against K-Means (do
    the two algorithms agree on borrower groupings?) and for the
    dendrogram visualization, less suited than K-Means as the deployed
    production segmentation algorithm given the missing `.predict()`.
    Computational considerations: not recommended beyond a few thousand
    rows without a specialized linkage/connectivity structure.

    Parameters
    ----------
    X : np.ndarray
    n_clusters : int
    linkage : str
        "ward" (minimizes within-cluster variance, requires Euclidean
        distance -- consistent with this project's standardized,
        Euclidean-distance clustering feature space), "average", or
        "complete".

    Returns
    -------
    ClusteringResult
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return ClusteringResult(algorithm="Agglomerative", model=model, labels=labels, n_clusters=n_clusters)


def fit_gaussian_mixture(X: np.ndarray, n_clusters: int, random_state: int = config.RANDOM_STATE) -> ClusteringResult:
    """
    Fit a Gaussian Mixture Model.

    Advantages: SOFT (probabilistic) cluster assignment -- a borrower
    can have, say, 60% membership in "Prime Borrowers" and 40% in
    "Moderate Risk," often a more honest representation of a continuous
    risk spectrum than a hard boundary; can model elliptical (not just
    spherical) clusters via a full covariance matrix.
    Disadvantages: more hyperparameters/assumptions than K-Means
    (covariance type), can be sensitive to initialization and slower to
    converge, harder to explain "why" a borrower is in a cluster to a
    non-technical audience than a simple nearest-centroid story.
    Business applicability: useful as a secondary lens when a hard
    segment boundary feels too rigid for portfolio risk reporting (e.g.
    reporting "expected segment mix" using membership probabilities
    rather than a single label), but K-Means remains the primary/default
    algorithm for its simplicity and speed.
    Computational considerations: O(n_samples x n_clusters x n_features^2)
    per EM iteration (the squared term from covariance estimation) --
    noticeably more expensive than K-Means as feature count grows.

    Parameters
    ----------
    X : np.ndarray
    n_clusters : int
    random_state : int

    Returns
    -------
    ClusteringResult
        `model.predict_proba(X)` gives the soft membership probabilities.
    """
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return ClusteringResult(algorithm="Gaussian Mixture", model=model, labels=labels, n_clusters=n_clusters)


def fit_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> ClusteringResult:
    """
    Fit DBSCAN (density-based clustering), included as an OPTIONAL
    comparison point.

    Advantages: does not require `n_clusters` chosen in advance,
    naturally identifies noise/outlier points (label -1) rather than
    forcing every borrower into a segment, can find non-convex cluster
    shapes.
    Disadvantages: highly sensitive to `eps`/`min_samples` choice, tends
    to produce one large "majority" cluster plus scattered noise points
    on financial data that doesn't have well-separated density regions
    (as is typical here -- borrower financial characteristics form a
    fairly continuous cloud, not naturally dense/sparse pockets), and
    provides no direct centroid-based interpretation for profiling.
    Business applicability: LIMITED for this project's goal (a small
    number of ACTIONABLE, roughly-equal-sized business segments for
    underwriting/marketing policy) -- DBSCAN's tendency toward one
    dominant cluster plus noise does not map well onto "N distinct
    borrower personas." Included in the comparison for completeness and
    as a check on whether the borrower population has genuinely
    dense/sparse structure K-Means might be masking; NOT used as
    `SegmentationEngine`'s default algorithm.
    Computational considerations: O(n log n) with a spatial index for
    moderate dimensionality; degrades toward O(n^2) as dimensionality
    grows (the "curse of dimensionality" also hurts DBSCAN's core
    distance-based density estimate here).

    Parameters
    ----------
    X : np.ndarray
    eps : float
    min_samples : int

    Returns
    -------
    ClusteringResult
        `n_clusters` excludes the noise label (-1) from the count.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return ClusteringResult(algorithm="DBSCAN", model=model, labels=labels, n_clusters=n_clusters)


CLUSTERING_ALGORITHMS = {
    "kmeans": fit_kmeans,
    "agglomerative": fit_agglomerative,
    "gaussian_mixture": fit_gaussian_mixture,
}


# ---------------------------------------------------------------------------
# 4. CLUSTER VALIDITY METRICS + OPTIMAL-K EVALUATION
# ---------------------------------------------------------------------------


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute the standard cluster-validity metric suite for one labeling.

    - Silhouette score [-1, 1]: how similar each point is to its own
      cluster vs. the nearest other cluster -- higher is better
      separated/cohesive; near 0 means overlapping clusters; negative
      means many points are probably misassigned.
    - Calinski-Harabasz index [0, inf): ratio of between-cluster to
      within-cluster dispersion -- higher is better; more sensitive to
      convex, similarly-sized clusters (a good fit for K-Means' own
      assumptions).
    - Davies-Bouldin index [0, inf): average similarity between each
      cluster and its most-similar other cluster -- LOWER is better
      (unlike the two metrics above), penalizing clusters that are close
      together or not compact.

    Parameters
    ----------
    X : np.ndarray
        The same feature matrix the clustering was fit on.
    labels : np.ndarray

    Returns
    -------
    dict
        {"silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"}.
        Returns NaN for all three if fewer than 2 distinct (non-noise)
        clusters are present (metrics are undefined in that case).
    """
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        logger.warning("Fewer than 2 clusters present -- validity metrics undefined.")
        return {"silhouette_score": np.nan, "calinski_harabasz_score": np.nan, "davies_bouldin_score": np.nan}

    mask = labels != -1  # exclude DBSCAN noise points from validity scoring
    return {
        "silhouette_score": float(silhouette_score(X[mask], labels[mask])),
        "calinski_harabasz_score": float(calinski_harabasz_score(X[mask], labels[mask])),
        "davies_bouldin_score": float(davies_bouldin_score(X[mask], labels[mask])),
    }


def evaluate_optimal_k(
    X: np.ndarray, k_candidates: List[int] = config.N_CLUSTERS_CANDIDATES,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Fit K-Means across every candidate k and compute the full validity-
    metric suite plus inertia (for the elbow method), so all four
    "Optimal Number of Clusters" evaluation methods required by Phase 4B
    can be compared side-by-side from one table.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed (scaled) feature matrix.
    k_candidates : list[int]
        Defaults to `config.N_CLUSTERS_CANDIDATES`.
    random_state : int

    Returns
    -------
    pd.DataFrame
        One row per k: inertia, silhouette_score, calinski_harabasz_score,
        davies_bouldin_score.
    """
    rows = []
    for k in k_candidates:
        result = fit_kmeans(X, n_clusters=k, random_state=random_state)
        metrics = evaluate_clustering(X, result.labels)
        rows.append({"n_clusters": k, "inertia": result.model.inertia_, **metrics})
        logger.info(
            "k=%d: inertia=%.1f, silhouette=%.3f, calinski_harabasz=%.1f, davies_bouldin=%.3f",
            k, result.model.inertia_, metrics["silhouette_score"],
            metrics["calinski_harabasz_score"], metrics["davies_bouldin_score"],
        )
    return pd.DataFrame(rows)


def recommend_optimal_k(optimal_k_table: pd.DataFrame) -> Tuple[int, str]:
    """
    Recommend a final cluster count from the optimal-k evaluation table
    using a simple, transparent voting scheme: rank each candidate k by
    silhouette score (higher better), Calinski-Harabasz (higher better),
    and Davies-Bouldin (lower better), then pick the k with the best
    average rank across all three -- more robust than trusting any
    single metric alone, since each has known biases (e.g.
    Calinski-Harabasz tends to favor more clusters; Davies-Bouldin can
    favor fewer).

    The elbow method (inertia) is reported alongside for visual
    corroboration but not included in the vote, since "the elbow" is a
    visual judgment call without a single unambiguous numeric rule.

    Parameters
    ----------
    optimal_k_table : pd.DataFrame
        Output of `evaluate_optimal_k`.

    Returns
    -------
    (int, str)
        Recommended k, and a short text explanation of the vote.
    """
    ranked = optimal_k_table.copy()
    ranked["silhouette_rank"] = ranked["silhouette_score"].rank(ascending=False)
    ranked["calinski_harabasz_rank"] = ranked["calinski_harabasz_score"].rank(ascending=False)
    ranked["davies_bouldin_rank"] = ranked["davies_bouldin_score"].rank(ascending=True)
    ranked["average_rank"] = ranked[["silhouette_rank", "calinski_harabasz_rank", "davies_bouldin_rank"]].mean(axis=1)

    best_row = ranked.loc[ranked["average_rank"].idxmin()]
    best_k = int(best_row["n_clusters"])

    explanation = (
        f"k={best_k} has the best average rank ({best_row['average_rank']:.2f}) across silhouette "
        f"(rank {best_row['silhouette_rank']:.0f}), Calinski-Harabasz (rank {best_row['calinski_harabasz_rank']:.0f}), "
        f"and Davies-Bouldin (rank {best_row['davies_bouldin_rank']:.0f}) among candidates "
        f"{sorted(ranked['n_clusters'].tolist())}."
    )
    return best_k, explanation
