"""
explainability.py
===================
Phase 4A reusable module: `ExplainabilityEngine`, a single class that
wraps a fitted Phase 3 model with SHAP-based attribution, producing both
global ("what drives this model overall") and local ("why did this
borrower get this score") explanations, ready to be plotted directly in
a future Streamlit app.

Design decisions
-----------------
- **One engine, one model.** `ExplainabilityEngine` is constructed
  around exactly one Phase 3 model (default: `config.PRODUCTION_MODEL_KEY`).
  This mirrors `RiskScoringEngine`'s design and keeps SHAP's explainer
  setup (which differs meaningfully between tree ensembles and linear
  models -- see `_build_explainer`) simple and explicit rather than
  branching on model type inside every public method.
- **SHAP explainer choice by model family.** Tree-based models
  (Random Forest, XGBoost) use `shap.TreeExplainer`, which computes
  EXACT Shapley values efficiently by exploiting tree structure rather
  than sampling -- the right tool whenever it's available. Logistic
  Regression uses `shap.LinearExplainer`, exact for linear models given
  a background distribution. Both compute in the model's raw/margin
  (log-odds) output space (`feature_perturbation="tree_path_dependent"`
  for trees, no `model_output="probability"` override) rather than
  probability space: this is faster, does not require passing a
  background dataset to `TreeExplainer` (avoiding a known slowness/
  compatibility rough edge with probability-output SHAP on sklearn
  ensembles), and Shapley values remain directly comparable and additive
  in this space. Every plot/table generated here is labeled accordingly.
- **One-way dependency on `RiskScoringEngine`.** `generate_business_summary`
  needs both "why" (SHAP, computed here) and "what to do" (risk tier /
  action, computed by `RiskScoringEngine`) -- so this module imports
  `risk_scoring`, never the reverse (see the design note at the top of
  `risk_scoring.py`).
- **Streamlit-ready by construction.** Every `generate_*_plot` method
  returns a `matplotlib.figure.Figure` (never calls `plt.show()`), so a
  Streamlit page can call `st.pyplot(fig)` directly without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from src import config, interpretation_utils, model_utils, utils
from src.risk_scoring import RiskScoringEngine, MODEL_PATHS

logger = utils.get_logger(__name__)

# Default feature-interaction pairs required by Phase 4A. Each tuple is
# (primary_feature, interacting_feature); see `analyze_feature_interactions`
# for how pairs with a multi-column one-hot member are handled differently
# from pairs where both sides are single numeric/ordinal columns.
DEFAULT_INTERACTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("annual_inc", "dti"),
    ("grade", "int_rate"),
    ("emp_length_years", "annual_inc"),
    ("purpose", "grade"),
    ("home_ownership", "dti"),
)

# Numeric features Partial Dependence / ICE plots are supported for
# (sklearn's PartialDependenceDisplay needs continuous columns; one-hot
# categorical raw columns are excluded -- see `generate_pdp_ice_plot`).
PDP_SUPPORTED_FEATURES: Tuple[str, ...] = tuple(config.NUMERIC_FEATURES)


@dataclass
class LocalExplanation:
    """Everything `ExplainabilityEngine.explain_prediction()` produces for one borrower."""

    default_probability: float
    base_value: float
    shap_explanation: shap.Explanation  # single-instance Explanation, ready for shap.plots.*
    feature_contributions: pd.DataFrame  # feature, feature_label, shap_value, feature_value
    top_risk_factors: List[str]          # humanized labels, strongest positive SHAP first
    top_protective_factors: List[str]    # humanized labels, strongest negative SHAP first
    business_summary: str


@dataclass
class GlobalExplanation:
    """Everything `ExplainabilityEngine.explain_global_model()` produces for the whole model."""

    importance_table: pd.DataFrame  # feature, feature_label, mean_abs_shap, permutation_importance, research_question
    top_features: List[str]
    least_influential_features: List[str]
    positive_contributors: List[str]  # features whose higher values push risk UP on average
    negative_contributors: List[str]  # features whose higher values push risk DOWN on average
    business_summary: str


class ExplainabilityEngine:
    """
    Loads one fitted Phase 3 model and provides SHAP-based global and
    local explanations, diagnostic plots, and executive-friendly text
    summaries.

    Parameters
    ----------
    model_key : str
        One of "logistic_regression", "random_forest", "xgboost".
        Defaults to `config.PRODUCTION_MODEL_KEY`.
    pipeline : Pipeline, optional
        Inject an already-fitted `Pipeline` directly instead of loading
        one from disk (mirrors `RiskScoringEngine`'s constructor, for the
        same notebook/testing convenience).
    background_data : pd.DataFrame, optional
        Raw (pre-preprocessing) borrower rows used as the SHAP background
        distribution. Defaults to a random sample of
        `config.SHAP_BACKGROUND_SAMPLE_SIZE` rows from the Phase 1
        training split (`utils.load_splits()`).
    risk_scoring_engine : RiskScoringEngine, optional
        Reused for `generate_business_summary`'s risk-tier/action text.
        If not supplied, one is constructed for the same `model_key`
        (and, if `pipeline` was injected, the same injected pipeline, so
        the two engines never disagree about which fitted model they're
        describing).
    """

    def __init__(
        self,
        model_key: str = config.PRODUCTION_MODEL_KEY,
        pipeline: Optional[Pipeline] = None,
        background_data: Optional[pd.DataFrame] = None,
        risk_scoring_engine: Optional[RiskScoringEngine] = None,
    ) -> None:
        if model_key not in MODEL_PATHS:
            raise ValueError(f"Unknown model_key '{model_key}'. Expected one of {list(MODEL_PATHS)}.")

        self.model_key = model_key
        self.model_display_name = model_utils.MODEL_DISPLAY_NAMES[model_key]

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            logger.info("Loading %s pipeline from %s", self.model_display_name, MODEL_PATHS[model_key])
            self.pipeline = utils.load_object(MODEL_PATHS[model_key])

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names: List[str] = model_utils.get_output_feature_names_from_pipeline(self.pipeline)

        self.background_data = background_data if background_data is not None else self._default_background_sample()
        self._background_transformed = self.preprocessor.transform(self.background_data)

        self.explainer = self._build_explainer()

        self.risk_scoring_engine = risk_scoring_engine or RiskScoringEngine(
            model_key=model_key, pipeline=self.pipeline,
        )

        logger.info(
            "ExplainabilityEngine ready: model=%s, explainer=%s, n_features=%d, background_n=%d",
            self.model_display_name, type(self.explainer).__name__,
            len(self.feature_names), len(self.background_data),
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _default_background_sample(self) -> pd.DataFrame:
        """Sample `config.SHAP_BACKGROUND_SAMPLE_SIZE` rows from the Phase 1 training split as the SHAP background distribution."""
        X_train, _, _, _, _, _ = utils.load_splits()
        n = min(config.SHAP_BACKGROUND_SAMPLE_SIZE, len(X_train))
        return X_train.sample(n=n, random_state=config.RANDOM_STATE).reset_index(drop=True)

    def _build_explainer(self):
        """
        Construct the model-family-appropriate SHAP explainer. See the
        module docstring for why tree models use `TreeExplainer` (exact,
        no background needed) while Logistic Regression uses
        `LinearExplainer` (exact given a background distribution).
        """
        if self.model_key in ("random_forest", "xgboost"):
            return shap.TreeExplainer(self.classifier)
        if self.model_key == "logistic_regression":
            masker = shap.maskers.Independent(self._background_transformed)
            return shap.LinearExplainer(self.classifier, masker)
        raise ValueError(f"No SHAP explainer strategy defined for model_key '{self.model_key}'.")  # pragma: no cover

    def _compute_shap_explanation(self, X_raw: pd.DataFrame) -> shap.Explanation:
        """
        Preprocess raw borrower rows and compute their SHAP `Explanation`
        (values, base_values, data), with human-readable feature names
        already attached.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows.

        Returns
        -------
        shap.Explanation
        """
        X_transformed = self.preprocessor.transform(X_raw)
        explanation = self.explainer(X_transformed)

        # TreeExplainer on a binary classifier occasionally returns a
        # (n_samples, n_features, n_classes) array; normalize to the
        # positive-class (index 1) slice so every downstream method can
        # assume a plain (n_samples, n_features) shape.
        if explanation.values.ndim == 3:
            explanation = explanation[:, :, 1]

        explanation.feature_names = self.feature_names
        return explanation

    # ------------------------------------------------------------------
    # Local explanations
    # ------------------------------------------------------------------

    def explain_prediction(self, borrower: pd.DataFrame, top_n: int = 5) -> LocalExplanation:
        """
        Produce a full local explanation for exactly one borrower: SHAP
        attribution per feature, the top risk-increasing and top
        risk-decreasing factors (humanized), and a plain-language
        business summary.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw borrower features.
        top_n : int
            Number of top risk/protective factors to surface.

        Returns
        -------
        LocalExplanation

        Raises
        ------
        ValueError
            If `borrower` does not contain exactly one row.
        """
        if len(borrower) != 1:
            raise ValueError(f"explain_prediction expects exactly one borrower row, got {len(borrower)}.")

        explanation = self._compute_shap_explanation(borrower)
        shap_values = explanation.values[0]
        base_value = float(np.atleast_1d(explanation.base_values)[0])
        feature_values = explanation.data[0]

        contributions = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": shap_values,
            "feature_value": feature_values,
        })
        contributions = interpretation_utils.humanize_feature_table(contributions)
        contributions = contributions.reindex(
            contributions["shap_value"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        top_risk = contributions[contributions["shap_value"] > 0].head(top_n)["feature_label"].tolist()
        top_protective = contributions[contributions["shap_value"] < 0].head(top_n)["feature_label"].tolist()

        prediction_summary = self.risk_scoring_engine.generate_prediction_summary(borrower)
        business_summary = interpretation_utils.generate_borrower_business_summary(
            risk_tier=prediction_summary.risk_tier,
            default_probability=prediction_summary.default_probability,
            top_risk_factors=top_risk,
            top_protective_factors=top_protective,
            recommended_action=prediction_summary.recommended_action,
        )

        return LocalExplanation(
            default_probability=prediction_summary.default_probability,
            base_value=base_value,
            shap_explanation=explanation[0],
            feature_contributions=contributions,
            top_risk_factors=top_risk,
            top_protective_factors=top_protective,
            business_summary=business_summary,
        )

    def generate_waterfall_plot(self, borrower: pd.DataFrame, max_display: int = 12) -> plt.Figure:
        """
        SHAP waterfall plot for one borrower: how each feature pushes
        the prediction from the model's average (base value) output to
        this borrower's specific output, in the model's log-odds
        (margin) space.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw borrower features.
        max_display : int
            Maximum number of individual features shown before the rest
            are grouped into a single "other features" bar.

        Returns
        -------
        matplotlib.figure.Figure
        """
        explanation = self._compute_shap_explanation(borrower)
        plt.figure()
        shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"{self.model_display_name}: Prediction Breakdown (log-odds scale)",
            fontsize=12, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        return fig

    def generate_force_plot(self, borrower: pd.DataFrame) -> plt.Figure:
        """
        SHAP force plot (matplotlib rendering) for one borrower: the same
        additive attribution as the waterfall plot, laid out as opposing
        "push" arrows from the base value to the final prediction.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw borrower features.

        Returns
        -------
        matplotlib.figure.Figure
        """
        explanation = self._compute_shap_explanation(borrower)
        shap.plots.force(explanation[0], matplotlib=True, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"{self.model_display_name}: Force Plot (log-odds scale)",
            fontsize=11, fontweight="bold", y=1.15,
        )
        return fig

    # ------------------------------------------------------------------
    # Global explanations
    # ------------------------------------------------------------------

    def _global_sample(self, X: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Resolve the sample used for global SHAP computation: caller-supplied `X`, or a fresh random sample from the training split."""
        if X is not None:
            if len(X) > config.SHAP_GLOBAL_SAMPLE_SIZE:
                return X.sample(n=config.SHAP_GLOBAL_SAMPLE_SIZE, random_state=config.RANDOM_STATE).reset_index(drop=True)
            return X.reset_index(drop=True)
        X_train, _, _, _, _, _ = utils.load_splits()
        n = min(config.SHAP_GLOBAL_SAMPLE_SIZE, len(X_train))
        return X_train.sample(n=n, random_state=config.RANDOM_STATE).reset_index(drop=True)

    def summarize_feature_importance(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build a single comparison table of mean |SHAP value| (this
        engine) against Phase 3's permutation importance
        (`reports/feature_importance.joblib`) and each model's native
        importance (Logistic Regression coefficients, Random Forest
        impurity, or XGBoost gain) -- cross-referencing an
        attribution-based measure against Phase 3's model-native measures
        is the same "don't trust one importance method alone" principle
        Phase 3 applied to impurity vs. permutation importance.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Raw feature rows to compute SHAP values over. Defaults to a
            fresh sample from the training split (see `_global_sample`).

        Returns
        -------
        pd.DataFrame
            Columns: feature, feature_label, mean_abs_shap,
            permutation_importance, native_importance,
            native_importance_type, research_question. Sorted by
            mean_abs_shap descending.
        """
        sample = self._global_sample(X)
        explanation = self._compute_shap_explanation(sample)
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)

        table = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs_shap})
        table = interpretation_utils.humanize_feature_table(table)
        table["research_question"] = table["feature"].apply(interpretation_utils.link_feature_to_research_question)

        try:
            phase3_importance = utils.load_object(config.FEATURE_IMPORTANCE_PATH)[self.model_key]
            if self.model_key == "logistic_regression":
                native = phase3_importance["coefficients"][["feature", "coefficient"]].rename(
                    columns={"coefficient": "native_importance"})
                native_type = "Coefficient (log-odds)"
            elif self.model_key == "random_forest":
                native = phase3_importance["impurity"][["feature", "importance"]].rename(
                    columns={"importance": "native_importance"})
                native_type = "Impurity importance"
            else:  # xgboost
                native = phase3_importance["gain_weight_cover"][["feature", "gain"]].rename(
                    columns={"gain": "native_importance"})
                native_type = "Gain importance"
            # native importance tables key on the same TECHNICAL
            # (preprocessed) feature names as this table -- merge directly.
            table = table.merge(native, on="feature", how="left")
            table["native_importance_type"] = native_type

            # Phase 3's permutation importance (model_utils.permutation_feature_importance)
            # was computed on RAW (pre-preprocessing) columns, so its
            # "feature" values ("dti") don't match this table's technical
            # names ("numeric__dti"). Bridge the two naming schemes via
            # the shared humanized label instead of the raw feature name.
            permutation = phase3_importance["permutation"][["feature", "importance_mean"]].copy()
            permutation["feature_label"] = permutation["feature"].apply(interpretation_utils.humanize_feature_name)
            permutation = permutation.rename(columns={"importance_mean": "permutation_importance"})
            table = table.merge(
                permutation[["feature_label", "permutation_importance"]], on="feature_label", how="left",
            )
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                "Could not load Phase 3 feature-importance artifacts (%s) -- "
                "returning SHAP-only importance table.", exc,
            )
            table["permutation_importance"] = np.nan
            table["native_importance"] = np.nan
            table["native_importance_type"] = None

        return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def explain_global_model(self, X: Optional[pd.DataFrame] = None, top_n: int = 8) -> GlobalExplanation:
        """
        Produce a full global explanation: ranked feature-importance
        table, top/least influential features, average positive vs.
        negative contributors (does a higher value of this feature tend
        to raise or lower predicted default risk, on average, across the
        sample), and a plain-language business summary.

        Parameters
        ----------
        X : pd.DataFrame, optional
            See `summarize_feature_importance`.
        top_n : int

        Returns
        -------
        GlobalExplanation
        """
        sample = self._global_sample(X)
        explanation = self._compute_shap_explanation(sample)
        importance_table = self.summarize_feature_importance(sample)

        mean_signed_shap = explanation.values.mean(axis=0)
        direction_table = pd.DataFrame({"feature": self.feature_names, "mean_signed_shap": mean_signed_shap})
        direction_table = interpretation_utils.humanize_feature_table(direction_table)

        top_features = importance_table.head(top_n)["feature_label"].tolist()
        least_influential = importance_table.tail(top_n)["feature_label"].tolist()
        positive_contributors = (
            direction_table[direction_table["mean_signed_shap"] > 0]
            .sort_values("mean_signed_shap", ascending=False)
            .head(top_n)["feature_label"].tolist()
        )
        negative_contributors = (
            direction_table[direction_table["mean_signed_shap"] < 0]
            .sort_values("mean_signed_shap")
            .head(top_n)["feature_label"].tolist()
        )

        business_summary = interpretation_utils.generate_global_business_summary(
            model_display_name=self.model_display_name,
            top_features=top_features,
            least_influential_features=least_influential,
        )

        return GlobalExplanation(
            importance_table=importance_table,
            top_features=top_features,
            least_influential_features=least_influential,
            positive_contributors=positive_contributors,
            negative_contributors=negative_contributors,
            business_summary=business_summary,
        )

    def generate_shap_summary(
        self, X: Optional[pd.DataFrame] = None, plot_type: str = "beeswarm", max_display: int = 15,
    ) -> plt.Figure:
        """
        Global SHAP summary plot -- "beeswarm" (every sample's SHAP value
        per feature, colored by feature value) or "bar" (mean |SHAP
        value| per feature) -- covering both the "SHAP Beeswarm Plot" and
        "SHAP Bar Plot" deliverables from one method via `plot_type`.

        Parameters
        ----------
        X : pd.DataFrame, optional
        plot_type : str
            "beeswarm" or "bar".
        max_display : int

        Returns
        -------
        matplotlib.figure.Figure
        """
        if plot_type not in ("beeswarm", "bar"):
            raise ValueError(f"plot_type must be 'beeswarm' or 'bar', got '{plot_type}'.")

        sample = self._global_sample(X)
        explanation = self._compute_shap_explanation(sample)
        explanation.feature_names = [interpretation_utils.humanize_feature_name(f) for f in self.feature_names]

        plt.figure()
        shap.summary_plot(
            explanation, plot_type=("dot" if plot_type == "beeswarm" else "bar"),
            max_display=max_display, show=False,
        )
        fig = plt.gcf()
        title = "Beeswarm" if plot_type == "beeswarm" else "Bar"
        fig.suptitle(
            f"{self.model_display_name}: SHAP {title} Summary (log-odds scale, n={len(sample)})",
            fontsize=12, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        return fig

    def generate_dependence_plot(
        self, feature: str, X: Optional[pd.DataFrame] = None, interaction_feature: Optional[str] = "auto",
    ) -> plt.Figure:
        """
        SHAP dependence plot for one feature: the feature's raw value on
        the x-axis, its SHAP value (contribution to predicted risk) on
        the y-axis, colored by an interacting feature -- a richer
        extension of a classical partial dependence plot that also
        reveals interaction effects via vertical dispersion / color
        gradient.

        Parameters
        ----------
        feature : str
            RAW column name (e.g. "dti") -- resolved to its technical
            (preprocessed) column name internally. Must map to a single
            output column (numeric or ordinal); one-hot categorical
            features are not supported by SHAP's scatter-style
            dependence plot (see `analyze_feature_interactions` for how
            those are handled instead).
        X : pd.DataFrame, optional
        interaction_feature : str, optional
            RAW column name to color by, or "auto" (SHAP picks the
            strongest-interacting feature automatically), or None (no
            coloring).

        Returns
        -------
        matplotlib.figure.Figure
        """
        technical_feature = self._resolve_single_feature_column(feature)
        technical_interaction = (
            self._resolve_single_feature_column(interaction_feature)
            if interaction_feature not in (None, "auto") else interaction_feature
        )

        sample = self._global_sample(X)
        explanation = self._compute_shap_explanation(sample)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        shap.dependence_plot(
            technical_feature, explanation.values, explanation.data,
            feature_names=self.feature_names, interaction_index=technical_interaction,
            ax=ax, show=False,
        )
        label = interpretation_utils.humanize_feature_name(technical_feature)
        ax.set_title(
            f"{self.model_display_name}: SHAP Dependence -- {label}",
            fontsize=12, fontweight="bold", loc="left",
        )
        fig.tight_layout()
        return fig

    def generate_decision_plot(self, X: Optional[pd.DataFrame] = None, n_samples: int = 20) -> plt.Figure:
        """
        SHAP decision plot: cumulative SHAP contributions traced as a
        line per borrower from the model's base value to their final
        prediction -- useful for spotting a handful of borrowers whose
        risk drivers diverge from the typical pattern at a glance.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Borrowers to plot. Capped at `n_samples` for readability.
        n_samples : int

        Returns
        -------
        matplotlib.figure.Figure
        """
        sample = self._global_sample(X)
        if len(sample) > n_samples:
            sample = sample.sample(n=n_samples, random_state=config.RANDOM_STATE).reset_index(drop=True)
        explanation = self._compute_shap_explanation(sample)
        base_value = float(np.atleast_1d(explanation.base_values)[0])

        plt.figure(figsize=(9, 7))
        shap.decision_plot(
            base_value, explanation.values, explanation.data,
            feature_names=[interpretation_utils.humanize_feature_name(f) for f in self.feature_names],
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"{self.model_display_name}: Decision Plot (n={len(sample)} borrowers, log-odds scale)",
            fontsize=12, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        return fig

    def _resolve_single_feature_column(self, raw_feature: str) -> str:
        """
        Map a raw column name to its technical (preprocessed) output
        column name, for features that produce exactly ONE output
        column (numeric or ordinal). Raises for one-hot categorical
        features, which produce multiple columns and cannot be resolved
        to a single technical name.
        """
        if raw_feature in config.NUMERIC_FEATURES:
            candidate = f"numeric__{raw_feature}"
        elif raw_feature in config.ORDINAL_CATEGORICAL_FEATURES:
            candidate = f"ordinal_categorical__{raw_feature}"
        elif raw_feature in config.ONEHOT_CATEGORICAL_FEATURES:
            raise ValueError(
                f"'{raw_feature}' is a one-hot encoded categorical feature and does not "
                f"map to a single output column. Use analyze_feature_interactions() instead."
            )
        else:
            raise ValueError(f"Unrecognized feature '{raw_feature}'.")

        if candidate not in self.feature_names:
            raise ValueError(f"Resolved technical feature '{candidate}' not found in pipeline output features.")
        return candidate

    # ------------------------------------------------------------------
    # Feature interaction analysis
    # ------------------------------------------------------------------

    def analyze_feature_interactions(
        self, X: Optional[pd.DataFrame] = None, pairs: Sequence[Tuple[str, str]] = DEFAULT_INTERACTION_PAIRS,
    ) -> Dict[Tuple[str, str], Dict[str, object]]:
        """
        Analyze each requested (primary_feature, interacting_feature)
        pair. Pairs where BOTH features resolve to a single numeric/
        ordinal output column (e.g. Income x DTI, Grade x Interest Rate,
        Employment Length x Income) use a SHAP dependence plot colored by
        the interacting feature -- vertical dispersion in that plot IS
        the interaction effect. Pairs involving a one-hot categorical
        feature (Purpose x Grade, Home Ownership x DTI) instead use a
        group-mean heatmap of predicted default probability across the
        two dimensions (continuous sides quartile-binned), since SHAP's
        scatter-style dependence plot cannot represent a multi-column
        categorical feature on a single axis.

        Parameters
        ----------
        X : pd.DataFrame, optional
        pairs : sequence of (str, str)
            Defaults to `DEFAULT_INTERACTION_PAIRS` (the five pairs
            named in the Phase 4A brief).

        Returns
        -------
        dict
            Keyed by (primary_feature, interacting_feature); each value
            is {"figure": Figure, "kind": "shap_dependence" | "heatmap",
            "interpretation": str}.
        """
        sample = self._global_sample(X)
        results: Dict[Tuple[str, str], Dict[str, object]] = {}

        for primary, interacting in pairs:
            both_single_column = (
                primary in config.NUMERIC_FEATURES + config.ORDINAL_CATEGORICAL_FEATURES
                and interacting in config.NUMERIC_FEATURES + config.ORDINAL_CATEGORICAL_FEATURES
            )
            if both_single_column:
                fig = self.generate_dependence_plot(primary, X=sample, interaction_feature=interacting)
                kind = "shap_dependence"
                interpretation = (
                    f"Vertical spread of points at a given {interpretation_utils.humanize_feature_name(primary)} "
                    f"value, and the color gradient by "
                    f"{interpretation_utils.humanize_feature_name(interacting)}, both indicate how much "
                    f"{interpretation_utils.humanize_feature_name(interacting)} changes the effect of "
                    f"{interpretation_utils.humanize_feature_name(primary)} on predicted risk -- a flat, "
                    f"uncolored-looking band would indicate little interaction, while a clear color gradient "
                    f"at fixed x-values indicates a meaningful interaction."
                )
            else:
                fig, interpretation = self._interaction_heatmap(primary, interacting, sample)
                kind = "heatmap"
            results[(primary, interacting)] = {"figure": fig, "kind": kind, "interpretation": interpretation}

        return results

    def _interaction_heatmap(self, primary: str, interacting: str, sample: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """Build a mean-predicted-probability heatmap across two raw features (quartile-binning any continuous side)."""
        plot_df = sample[[primary, interacting]].copy()
        plot_df["predicted_probability"] = self.pipeline.predict_proba(sample)[:, 1]

        for col in (primary, interacting):
            if col in config.NUMERIC_FEATURES:
                plot_df[col] = interpretation_utils.bin_column_for_fairness(plot_df[col], n_bins=4)

        pivot = plot_df.pivot_table(index=primary, columns=interacting, values="predicted_probability", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.to_numpy()[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5)
        fig.colorbar(im, ax=ax, label="Mean predicted default probability")
        primary_label = interpretation_utils.humanize_feature_name(primary)
        interacting_label = interpretation_utils.humanize_feature_name(interacting)
        ax.set_xlabel(interacting_label)
        ax.set_ylabel(primary_label)
        ax.set_title(
            f"{self.model_display_name}: Mean Predicted Risk by {primary_label} x {interacting_label}",
            fontsize=12, fontweight="bold", loc="left",
        )
        fig.tight_layout()

        interpretation = (
            f"Each cell is the average predicted default probability for borrowers in that "
            f"{primary_label} / {interacting_label} combination. A pattern where risk climbs "
            f"faster in one row/column than others (rather than uniformly) indicates the two "
            f"variables interact -- e.g. if risk is flat across {interacting_label} within the "
            f"safest {primary_label} category but rises sharply within the riskiest one, "
            f"{interacting_label} matters more for already-risky borrowers than for safe ones."
        )
        return fig, interpretation

    # ------------------------------------------------------------------
    # Partial dependence / ICE
    # ------------------------------------------------------------------

    def generate_pdp_ice_plot(
        self, X: Optional[pd.DataFrame] = None, features: Sequence[str] = ("dti", "int_rate", "annual_inc"),
        kind: str = "both",
    ) -> plt.Figure:
        """
        Partial Dependence (PD) and/or Individual Conditional Expectation
        (ICE) plots for one or more numeric features, computed directly
        on the fitted `Pipeline` (preprocessing included) via
        scikit-learn's `PartialDependenceDisplay` -- independent of SHAP,
        included as a second, model-agnostic lens on marginal effects.

        Parameters
        ----------
        X : pd.DataFrame, optional
        features : sequence of str
            RAW numeric column names (must be in `config.NUMERIC_FEATURES`
            -- one-hot categorical raw columns are not supported by
            scikit-learn's PDP without additional categorical-feature
            configuration and are out of scope here).
        kind : str
            "average" (PD only), "individual" (ICE only), or "both".

        Returns
        -------
        matplotlib.figure.Figure
        """
        unsupported = [f for f in features if f not in PDP_SUPPORTED_FEATURES]
        if unsupported:
            raise ValueError(
                f"generate_pdp_ice_plot only supports numeric features: {unsupported} not in "
                f"{PDP_SUPPORTED_FEATURES}."
            )

        sample = self._global_sample(X)
        fig, axes = plt.subplots(1, len(features), figsize=(5.5 * len(features), 5), squeeze=False)
        PartialDependenceDisplay.from_estimator(
            self.pipeline, sample, features=list(features), kind=kind,
            ax=axes[0], response_method="predict_proba",
        )
        for ax, feature in zip(axes[0], features):
            ax.set_xlabel(interpretation_utils.humanize_feature_name(feature))
            ax.set_ylabel("Predicted default probability")
        fig.suptitle(
            f"{self.model_display_name}: Partial Dependence" + (" & ICE" if kind == "both" else ""),
            fontsize=13, fontweight="bold", y=1.05,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Business summary (local or global) + export
    # ------------------------------------------------------------------

    def generate_business_summary(self, borrower: Optional[pd.DataFrame] = None) -> str:
        """
        Generate an executive-friendly narrative: for one borrower if
        `borrower` is supplied (delegates to `explain_prediction`), or
        for the model overall if not (delegates to `explain_global_model`).

        Parameters
        ----------
        borrower : pd.DataFrame, optional
            Exactly one row of raw borrower features, or None for a
            global summary.

        Returns
        -------
        str
        """
        if borrower is not None:
            return self.explain_prediction(borrower).business_summary
        return self.explain_global_model().business_summary

    def export_borrower_explanation_report(self, borrower: pd.DataFrame) -> "interpretation_utils.ExportableReport":
        """
        Build an `ExportableReport` bundling a borrower's business
        summary and top risk/protective factors -- the "Borrower
        Explanation" exportable report required by Phase 4A.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw borrower features.

        Returns
        -------
        interpretation_utils.ExportableReport
        """
        local = self.explain_prediction(borrower)
        sections = {
            "Summary": local.business_summary,
            "Top Risk Factors": "\n".join(f"- {f}" for f in local.top_risk_factors) or "None identified.",
            "Top Protective Factors": "\n".join(f"- {f}" for f in local.top_protective_factors) or "None identified.",
            "Full Feature Contribution Table": interpretation_utils.dataframe_to_markdown_table(
                local.feature_contributions[["feature_label", "shap_value", "feature_value"]], max_rows=15,
            ),
        }
        return interpretation_utils.ExportableReport(title="Borrower Explanation Report", sections=sections)

    def export_global_explanation_report(self, X: Optional[pd.DataFrame] = None) -> "interpretation_utils.ExportableReport":
        """
        Build an `ExportableReport` bundling the model's global business
        summary and feature-importance table -- the "Feature Importance"
        / "Executive Report" exportable reports required by Phase 4A.

        Parameters
        ----------
        X : pd.DataFrame, optional

        Returns
        -------
        interpretation_utils.ExportableReport
        """
        global_explanation = self.explain_global_model(X)
        sections = {
            "Model": f"{self.model_display_name} (production scoring model)",
            "Executive Summary": global_explanation.business_summary,
            "Most Influential Variables": "\n".join(f"- {f}" for f in global_explanation.top_features),
            "Least Influential Variables": "\n".join(f"- {f}" for f in global_explanation.least_influential_features),
            "Positive Contributors to Default Risk": "\n".join(f"- {f}" for f in global_explanation.positive_contributors) or "None identified.",
            "Negative Contributors to Default Risk (protective)": "\n".join(f"- {f}" for f in global_explanation.negative_contributors) or "None identified.",
            "Feature Importance Table": interpretation_utils.dataframe_to_markdown_table(
                global_explanation.importance_table[["feature_label", "mean_abs_shap", "permutation_importance", "research_question"]],
                max_rows=15,
            ),
        }
        return interpretation_utils.ExportableReport(title="Global Model Explanation Report", sections=sections)

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def persist_explainability_artifacts(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None,
        fairness_group_columns: Sequence[str] = ("home_ownership", "purpose", "grade"),
    ) -> None:
        """
        Compute and serialize (via `joblib`) every reusable Phase 4A
        artifact a future Streamlit dashboard needs, without recomputing
        SHAP values from scratch each time the app starts:

            - `config.SHAP_IMPORTANCE_PATH` -- the SHAP/permutation/
              native importance comparison table (`summarize_feature_importance`).
            - `config.BUSINESS_SUMMARY_TEMPLATES_PATH` -- an example
              global and local business summary.
            - `config.MODEL_METADATA_PATH` -- model key, display name,
              feature names, decision threshold, explainer type.
            - `config.FAIRNESS_REPORT_PATH` -- per-group performance
              metrics (`interpretation_utils.fairness_report`).
            - `config.FEATURE_INTERACTION_SUMMARY_PATH` -- textual
              interpretations from `analyze_feature_interactions`
              (figures themselves are not persisted -- regenerate on
              demand, since a stale cached SHAP plot would be more
              confusing than a few seconds of recomputation).

        The risk-threshold configuration (`config.RISK_THRESHOLD_CONFIG_PATH`)
        is saved separately and automatically by
        `configurable_thresholds.RiskThresholdConfig.load` the first time
        it's requested -- not duplicated here.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Defaults to the Phase 1 test split.
        y : pd.Series, optional
            Required (alongside `X`) to compute the fairness report;
            defaults to the Phase 1 test split's target if `X` is also
            left as default.
        fairness_group_columns : sequence of str
        """
        if X is None or y is None:
            _, _, X, _, _, y = utils.load_splits()

        utils.ensure_directories()

        importance_table = self.summarize_feature_importance(X)
        utils.save_object(importance_table, config.SHAP_IMPORTANCE_PATH)

        global_explanation = self.explain_global_model(X)
        example_local = self.explain_prediction(X.iloc[[0]])
        utils.save_object(
            {
                "global_business_summary": global_explanation.business_summary,
                "example_local_business_summary": example_local.business_summary,
            },
            config.BUSINESS_SUMMARY_TEMPLATES_PATH,
        )

        utils.save_object(
            {
                "production_model_key": self.model_key,
                "model_display_name": self.model_display_name,
                "feature_names": self.feature_names,
                "n_features": len(self.feature_names),
                "decision_threshold": self.risk_scoring_engine._decision_threshold,
                "shap_explainer_type": type(self.explainer).__name__,
            },
            config.MODEL_METADATA_PATH,
        )

        proba = self.risk_scoring_engine.predict_probability(X)
        fairness_table = interpretation_utils.fairness_report(
            X, y, proba, group_columns=list(fairness_group_columns),
            threshold=self.risk_scoring_engine._decision_threshold,
        )
        utils.save_object(fairness_table, config.FAIRNESS_REPORT_PATH)

        interaction_results = self.analyze_feature_interactions(X)
        interaction_summary = {
            f"{primary}_x_{interacting}": result["interpretation"]
            for (primary, interacting), result in interaction_results.items()
        }
        utils.save_object(interaction_summary, config.FEATURE_INTERACTION_SUMMARY_PATH)

        logger.info("All Phase 4A explainability artifacts persisted to %s", config.EXPLAINABILITY_DIR)
