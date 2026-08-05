"""
segmentation_engine.py
========================
Phase 4B reusable module: `SegmentationEngine`, the single class that
owns the end-to-end borrower-segmentation workflow -- data preparation,
clustering, dimensionality reduction for visualization, data-driven
segment naming, profiling, business recommendations, and comparison
against Phase 3's supervised models -- composed from the primitives in
`cluster_analysis.py`, `cluster_visualization.py`, and
`segment_profiles.py`.

Design decisions
-----------------
- **One engine, one fitted clustering model.** Mirrors
  `RiskScoringEngine` and `ExplainabilityEngine`'s design from Phase 4A:
  construct once (`fit()`), then call cheap methods against the fitted
  state, ready to be cached (e.g. via Streamlit's `st.cache_resource`)
  in Phase 5.
- **Clustering feature space is intentionally NARROWER than the
  supervised models' feature space** (numeric + ordinal `grade` only,
  no one-hot categoricals) -- see the rationale in `config.py`'s Phase
  4B section and `cluster_analysis.build_clustering_preprocessor`.
  Categorical columns are still fully used for PROFILING each segment
  (`segment_profiles.py`), just not for defining cluster membership.
- **Complements, not replaces, supervised prediction.** `SegmentationEngine`
  never predicts an individual borrower's default probability -- that
  remains `RiskScoringEngine`'s job. `compare_segments()` and
  `compare_with_supervised_models()` explicitly cross-reference
  Phase 3/4A's predicted probabilities against cluster membership so the
  two analytical lenses (supervised risk score, unsupervised segment)
  are used together, per the Phase 4B brief's "Relationship to Machine
  Learning" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src import cluster_analysis as ca
from src import cluster_visualization as cv
from src import config, interpretation_utils, segment_profiles as sprof, utils
from src.configurable_thresholds import RiskThresholdConfig, load_threshold_config
from src.risk_scoring import RiskScoringEngine

logger = utils.get_logger(__name__)


@dataclass
class SegmentationFitResult:
    """
    Everything produced by `SegmentationEngine.fit()`, held as engine
    state and also returned for convenience.
    """

    labels: np.ndarray
    profile_table: pd.DataFrame
    segment_names: Dict[int, str]
    profiles: Dict[int, "sprof.SegmentProfile"]
    recommendations: Dict[int, "sprof.SegmentRecommendation"]
    optimal_k_table: pd.DataFrame
    recommended_k: int
    recommended_k_explanation: str


class SegmentationEngine:
    """
    Fits a borrower-segmentation model and exposes methods to predict,
    profile, visualize, compare, and export borrower segments.

    Parameters
    ----------
    n_clusters : int, optional
        Defaults to `config.DEFAULT_N_CLUSTERS`. Ignored if
        `auto_select_k=True` is passed to `fit()`.
    algorithm : str
        One of `cluster_analysis.CLUSTERING_ALGORITHMS` keys
        ("kmeans", "agglomerative", "gaussian_mixture"). Defaults to
        `config.DEFAULT_CLUSTERING_ALGORITHM`. See each algorithm's
        docstring in `cluster_analysis.py` for the comparative
        advantages/disadvantages/business-applicability discussion.
    risk_scoring_engine : RiskScoringEngine, optional
        Reused for `compare_with_supervised_models()`. Constructed
        lazily (on first use) with `config.PRODUCTION_MODEL_KEY` if not
        supplied, mirroring `ExplainabilityEngine`'s pattern from
        Phase 4A.
    threshold_config : RiskThresholdConfig, optional
        Reused to translate each segment's average predicted default
        probability into a familiar risk-tier label for profiling.
    """

    def __init__(
        self,
        n_clusters: int = config.DEFAULT_N_CLUSTERS,
        algorithm: str = config.DEFAULT_CLUSTERING_ALGORITHM,
        risk_scoring_engine: Optional[RiskScoringEngine] = None,
        threshold_config: Optional[RiskThresholdConfig] = None,
    ) -> None:
        if algorithm not in ca.CLUSTERING_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Expected one of {list(ca.CLUSTERING_ALGORITHMS)}.")

        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.threshold_config = threshold_config or load_threshold_config()
        self._risk_scoring_engine = risk_scoring_engine  # lazily constructed -- see risk_scoring_engine property

        self.preprocessor: Optional[ColumnTransformer] = None
        self.clustering_result: Optional[ca.ClusteringResult] = None
        self.fit_result: Optional[SegmentationFitResult] = None
        self._X_train_raw: Optional[pd.DataFrame] = None  # retained for describe_segment()/visualize_clusters() defaults

        logger.info("SegmentationEngine constructed: algorithm=%s, n_clusters=%d", algorithm, n_clusters)

    @property
    def risk_scoring_engine(self) -> RiskScoringEngine:
        """Lazily construct the shared `RiskScoringEngine` (avoids loading the production model until actually needed)."""
        if self._risk_scoring_engine is None:
            self._risk_scoring_engine = RiskScoringEngine(threshold_config=self.threshold_config)
        return self._risk_scoring_engine

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self, X: pd.DataFrame, default_flags: Optional[pd.Series] = None,
        auto_select_k: bool = False, k_candidates: List[int] = config.N_CLUSTERS_CANDIDATES,
    ) -> "SegmentationEngine":
        """
        Fit the full segmentation workflow: clip outliers, fit the
        clustering preprocessor, (optionally) evaluate the optimal
        cluster count, fit the clustering algorithm, build the profile
        table, assign data-driven segment names, and generate business
        recommendations.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows -- typically
            the Phase 1 training split.
        default_flags : pd.Series, optional
            Binary actual-outcome column, aligned with `X`, used to
            compute each segment's average default rate and to derive
            each segment's risk tier via the production model's
            predicted probabilities (see `_compute_risk_tier_lookup`).
            If omitted, segment risk tiers fall back to a simpler
            default-rate-based rule inside `segment_profiles.py`, or
            "Unknown" if no outcome data is available at all.
        auto_select_k : bool
            If True, run `cluster_analysis.evaluate_optimal_k` /
            `recommend_optimal_k` and use the recommended k instead of
            `self.n_clusters`.
        k_candidates : list[int]
            Candidates evaluated if `auto_select_k=True`.

        Returns
        -------
        SegmentationEngine
            self, for chaining (`engine = SegmentationEngine().fit(X)`).
        """
        logger.info("Fitting SegmentationEngine on %d borrowers.", len(X))
        self._X_train_raw = X.reset_index(drop=True)
        default_flags = default_flags.reset_index(drop=True) if default_flags is not None else None

        clipped = ca.clip_outliers(X)
        self.preprocessor = ca.build_clustering_preprocessor()
        X_transformed = self.preprocessor.fit_transform(clipped)

        optimal_k_table = ca.evaluate_optimal_k(X_transformed, k_candidates=k_candidates)
        recommended_k, explanation = ca.recommend_optimal_k(optimal_k_table)
        if auto_select_k:
            self.n_clusters = recommended_k
            logger.info("auto_select_k=True: using recommended k=%d. %s", recommended_k, explanation)

        fit_fn = ca.CLUSTERING_ALGORITHMS[self.algorithm]
        self.clustering_result = fit_fn(X_transformed, n_clusters=self.n_clusters)

        profile_table = sprof.build_cluster_profile_table(self._X_train_raw, self.clustering_result.labels, default_flags=default_flags)
        risk_tier_lookup = self._compute_risk_tier_lookup(default_flags)
        segment_names = sprof.assign_segment_names(profile_table)
        profiles = sprof.build_segment_profiles(profile_table, segment_names, risk_tier_lookup=risk_tier_lookup)
        recommendations = {cid: sprof.recommend_segment_actions(p) for cid, p in profiles.items()}

        self.fit_result = SegmentationFitResult(
            labels=self.clustering_result.labels, profile_table=profile_table, segment_names=segment_names,
            profiles=profiles, recommendations=recommendations, optimal_k_table=optimal_k_table,
            recommended_k=recommended_k, recommended_k_explanation=explanation,
        )
        logger.info("SegmentationEngine fit complete: %d segments -> %s", self.n_clusters, segment_names)
        return self

    def _compute_risk_tier_lookup(self, default_flags: Optional[pd.Series]) -> Optional[Dict[int, str]]:
        """
        Derive each cluster's risk tier from the PRODUCTION MODEL's mean
        predicted default probability within that cluster (via
        `RiskThresholdConfig.get_tier`), rather than only from the
        cluster's raw historical default rate -- this ties segmentation
        risk tiers to the same supervised-model-driven definition
        `RiskScoringEngine` uses for individual borrowers, keeping the
        two engines' risk language consistent (see
        `compare_with_supervised_models` for the fuller cross-check).
        """
        if default_flags is None or self._X_train_raw is None:
            return None
        try:
            proba = self.risk_scoring_engine.predict_probability(self._X_train_raw)
        except FileNotFoundError as exc:
            logger.warning("Could not load production model for risk-tier lookup (%s) -- falling back to default-rate-based tiers.", exc)
            return None

        working = pd.DataFrame({"cluster": self.clustering_result.labels, "predicted_probability": proba})
        mean_proba_by_cluster = working.groupby("cluster")["predicted_probability"].mean()
        return {cluster_id: self.threshold_config.get_tier(p) for cluster_id, p in mean_proba_by_cluster.items()}

    # ------------------------------------------------------------------
    # Prediction / assignment
    # ------------------------------------------------------------------

    def _require_fit(self) -> None:
        if self.fit_result is None or self.clustering_result is None:
            raise RuntimeError("SegmentationEngine has not been fit yet. Call .fit(X) first.")

    def predict_cluster(self, X: pd.DataFrame) -> np.ndarray:
        """
        Assign cluster labels to new borrower rows.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows.

        Returns
        -------
        np.ndarray
            Cluster id per row.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        NotImplementedError
            If the fitted algorithm has no `.predict()` (Agglomerative
            Clustering has no out-of-sample prediction -- see its
            docstring in `cluster_analysis.py`); use nearest-centroid
            assignment via `assign_segment` instead, or refit including
            the new rows.
        """
        self._require_fit()
        clipped = ca.clip_outliers(X)
        X_transformed = self.preprocessor.transform(clipped)
        model = self.clustering_result.model
        if hasattr(model, "predict"):
            return model.predict(X_transformed)
        raise NotImplementedError(
            f"The fitted '{self.algorithm}' model has no out-of-sample .predict(). "
            f"Use assign_segment() for nearest-centroid assignment, or refit including these rows."
        )

    def assign_segment(self, X: pd.DataFrame) -> pd.Series:
        """
        Assign each row the business SEGMENT NAME (not just the numeric
        cluster id) it belongs to, via nearest-centroid distance in the
        clustering feature space -- works for every algorithm (including
        Agglomerative Clustering, which has no native `.predict()`),
        since centroids can always be computed post hoc from labeled
        training data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows.

        Returns
        -------
        pd.Series
            Segment name per row, aligned with `X`'s index.
        """
        self._require_fit()
        clipped = ca.clip_outliers(X)
        X_transformed = self.preprocessor.transform(clipped)

        train_transformed = self.preprocessor.transform(ca.clip_outliers(self._X_train_raw))
        cluster_ids = sorted(set(self.clustering_result.labels) - {-1})
        centroids = np.array([
            train_transformed[self.clustering_result.labels == cluster_id].mean(axis=0)
            for cluster_id in cluster_ids
        ])

        distances = np.linalg.norm(X_transformed[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.array(cluster_ids)[distances.argmin(axis=1)]
        names = [self.fit_result.segment_names.get(c, f"Cluster {c}") for c in nearest]
        return pd.Series(names, index=X.index, name="segment")

    # ------------------------------------------------------------------
    # Profiling / description
    # ------------------------------------------------------------------

    def describe_segment(self, cluster_id: int) -> str:
        """
        Return the executive-friendly paragraph describing one segment.

        Parameters
        ----------
        cluster_id : int

        Returns
        -------
        str
        """
        self._require_fit()
        if cluster_id not in self.fit_result.profiles:
            raise ValueError(f"Unknown cluster_id {cluster_id}. Known clusters: {list(self.fit_result.profiles)}.")
        return sprof.describe_segment_text(self.fit_result.profiles[cluster_id])

    def generate_cluster_profile(self) -> pd.DataFrame:
        """
        Return the full per-cluster profile table (typical income, DTI,
        loan amount, interest rate, grade, employment length, home
        ownership, loan purpose, default rate, credit utilization,
        borrower count) -- the raw data backing every `SegmentProfile`.

        Returns
        -------
        pd.DataFrame
        """
        self._require_fit()
        return self.fit_result.profile_table

    def compare_segments(self) -> pd.DataFrame:
        """
        Return the executive segment-comparison table: income, interest
        rate, loan grade, DTI, employment length, default rate, risk
        tier, and cluster size, one row per segment, ranked by risk.

        Returns
        -------
        pd.DataFrame
        """
        self._require_fit()
        return sprof.build_segment_comparison_table(self.fit_result.profiles)

    def recommend_business_actions(
        self, cluster_id: Optional[int] = None,
    ) -> Union["sprof.SegmentRecommendation", Dict[int, "sprof.SegmentRecommendation"]]:
        """
        Return business-action recommendations (risk level, lending/
        rate/underwriting strategy, manual-review requirement, marketing
        strategy, portfolio notes) for one segment, or all segments if
        `cluster_id` is omitted.

        Parameters
        ----------
        cluster_id : int, optional

        Returns
        -------
        SegmentRecommendation or dict[int, SegmentRecommendation]
        """
        self._require_fit()
        if cluster_id is not None:
            if cluster_id not in self.fit_result.recommendations:
                raise ValueError(f"Unknown cluster_id {cluster_id}.")
            return self.fit_result.recommendations[cluster_id]
        return self.fit_result.recommendations

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_clusters(self, method: str = "pca", X: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        2D scatter plot of the fitted clusters via the requested
        dimensionality-reduction method.

        Parameters
        ----------
        method : str
            "pca", "tsne", or "umap".
        X : pd.DataFrame, optional
            Defaults to the data `fit()` was called on. For t-SNE/UMAP,
            automatically subsampled to
            `config.DIMENSIONALITY_REDUCTION_SAMPLE_SIZE` rows for speed.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_fit()
        X_source = X if X is not None else self._X_train_raw
        labels = self.clustering_result.labels if X is None else self.predict_cluster(X)

        clipped = ca.clip_outliers(X_source)
        X_transformed = self.preprocessor.transform(clipped)

        if method == "pca":
            result = ca.fit_pca(X_transformed)
            coordinates, plot_labels = result.coordinates, labels
        elif method in ("tsne", "umap"):
            n = min(config.DIMENSIONALITY_REDUCTION_SAMPLE_SIZE, len(X_transformed))
            rng = np.random.default_rng(config.RANDOM_STATE)
            idx = rng.choice(len(X_transformed), size=n, replace=False)
            result = ca.fit_tsne(X_transformed[idx]) if method == "tsne" else ca.fit_umap(X_transformed[idx])
            if result is None:
                raise RuntimeError("UMAP is unavailable in this environment (umap-learn not installed).")
            coordinates, plot_labels = result.coordinates, labels[idx]
        else:
            raise ValueError(f"Unknown method '{method}'. Expected 'pca', 'tsne', or 'umap'.")

        return cv.plot_dimensionality_reduction_scatter(
            coordinates, plot_labels, result.method, segment_names=self.fit_result.segment_names,
        )

    # ------------------------------------------------------------------
    # Relationship to supervised models
    # ------------------------------------------------------------------

    def compare_with_supervised_models(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cross-reference each segment against the PRODUCTION supervised
        model's predicted default probability -- directly addresses the
        Phase 4B "Relationship to Machine Learning" requirement: do
        high-risk clusters align with high predicted default
        probabilities, and which clusters concentrate the most defaults?

        Parameters
        ----------
        X : pd.DataFrame, optional
            Defaults to the data `fit()` was called on.

        Returns
        -------
        pd.DataFrame
            One row per segment: segment_name, n_borrowers,
            mean_predicted_probability (from the production supervised
            model), average_default_rate (actual outcome, if available
            from `fit()`), and risk_tier.
        """
        self._require_fit()
        X_source = X if X is not None else self._X_train_raw
        labels = self.clustering_result.labels if X is None else self.predict_cluster(X)
        proba = self.risk_scoring_engine.predict_probability(X_source)

        working = pd.DataFrame({"cluster": labels, "predicted_probability": proba})
        grouped = working.groupby("cluster")["predicted_probability"].mean().rename("mean_predicted_probability")

        name_to_cluster_id = {name: cid for cid, name in self.fit_result.segment_names.items()}
        comparison = self.compare_segments().copy()
        comparison["cluster_id"] = comparison["segment_name"].map(name_to_cluster_id)
        comparison = comparison.merge(grouped, left_on="cluster_id", right_index=True, how="left")
        return comparison.sort_values("mean_predicted_probability", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_segment_summary(self) -> "interpretation_utils.ExportableReport":
        """
        Build the full exportable segmentation report (executive
        summary, segment comparison table, per-segment profile +
        recommendations) -- ready for `.to_markdown()` / `.to_json()` /
        `.save()`, e.g. behind a future Streamlit download button.

        Returns
        -------
        interpretation_utils.ExportableReport
        """
        self._require_fit()
        return sprof.export_segment_summary_report(self.fit_result.profiles, self.fit_result.recommendations)

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def persist_segmentation_artifacts(self) -> None:
        """
        Serialize every reusable Phase 4B artifact a future Streamlit
        dashboard needs: the fitted clustering model, the clustering
        preprocessor, cluster centroids, segment definitions
        (name/profile/recommendation per cluster), cluster metadata, and
        the segment-profile table.

        Artifacts written (all under `config.SEGMENTATION_DIR` /
        `config.MODELS_DIR`):
            - `config.CLUSTERING_MODEL_PATH` -- the fitted clustering estimator
            - `config.CLUSTERING_PREPROCESSOR_PATH` -- the fitted clustering preprocessor
            - `config.CLUSTER_CENTROIDS_PATH` -- per-cluster centroid vectors
            - `config.SEGMENT_DEFINITIONS_PATH` -- {cluster_id: {profile, recommendation}}
            - `config.CLUSTER_METADATA_PATH` -- algorithm, k, feature list, segment names
            - `config.SEGMENT_PROFILES_PATH` -- the raw profile table
            - `config.OPTIMAL_K_ANALYSIS_PATH` -- the optimal-k evaluation table
        """
        self._require_fit()
        utils.ensure_directories()

        utils.save_object(self.clustering_result.model, config.CLUSTERING_MODEL_PATH)
        utils.save_object(self.preprocessor, config.CLUSTERING_PREPROCESSOR_PATH)

        train_transformed = self.preprocessor.transform(ca.clip_outliers(self._X_train_raw))
        cluster_ids = sorted(set(self.clustering_result.labels) - {-1})
        centroids = {
            cluster_id: train_transformed[self.clustering_result.labels == cluster_id].mean(axis=0)
            for cluster_id in cluster_ids
        }
        utils.save_object(centroids, config.CLUSTER_CENTROIDS_PATH)

        segment_definitions = {
            cluster_id: {
                "profile": self.fit_result.profiles[cluster_id].to_dict(),
                "recommendation": self.fit_result.recommendations[cluster_id].to_dict(),
            }
            for cluster_id in self.fit_result.profiles
        }
        utils.save_object(segment_definitions, config.SEGMENT_DEFINITIONS_PATH)

        metadata = {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "clustering_numeric_features": config.CLUSTERING_NUMERIC_FEATURES,
            "clustering_ordinal_features": config.CLUSTERING_ORDINAL_FEATURES,
            "segment_names": self.fit_result.segment_names,
            "recommended_k": self.fit_result.recommended_k,
            "recommended_k_explanation": self.fit_result.recommended_k_explanation,
        }
        utils.save_object(metadata, config.CLUSTER_METADATA_PATH)

        utils.save_object(self.fit_result.profile_table, config.SEGMENT_PROFILES_PATH)
        utils.save_object(self.fit_result.optimal_k_table, config.OPTIMAL_K_ANALYSIS_PATH)

        logger.info("All Phase 4B segmentation artifacts persisted to %s / %s", config.SEGMENTATION_DIR, config.MODELS_DIR)
