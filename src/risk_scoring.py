"""
risk_scoring.py
=================
Phase 4A reusable module: `RiskScoringEngine`, a single class that turns
a fitted Phase 3 model into borrower-facing lending decisions --
probability, risk tier, risk score, confidence, and recommended action /
interest rate / loan grade.

Design decision: `RiskScoringEngine` deliberately does NOT depend on
SHAP or `explainability.py`. It answers "what should we DO about this
borrower" (a scoring + business-rule question); `ExplainabilityEngine`
(see `explainability.py`) answers "WHY does the model think this"
(an attribution question) and internally uses `RiskScoringEngine` for
the "what" half of a full narrative explanation. Keeping this one-way
dependency (explainability -> risk_scoring, never the reverse) means
`RiskScoringEngine` alone -- the cheaper, SHAP-free path -- is all a
future Streamlit "quick score" view needs to import.

Everything a stakeholder might want to change (risk-tier boundaries,
lending actions, rate adjustments, loan-grade bands) is read from
`configurable_thresholds.RiskThresholdConfig`, never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config, interpretation_utils, model_utils, utils
from src.configurable_thresholds import RiskThresholdConfig, load_threshold_config

logger = utils.get_logger(__name__)

MODEL_PATHS: Dict[str, Path] = {
    "logistic_regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
    "random_forest": config.RANDOM_FOREST_MODEL_PATH,
    "xgboost": config.XGBOOST_MODEL_PATH,
}


@dataclass
class PredictionSummary:
    """
    Everything `RiskScoringEngine.generate_prediction_summary()` produces
    for one borrower -- the structured object a future Streamlit page
    would bind directly to its UI.
    """

    default_probability: float
    risk_score: float
    risk_tier: str
    risk_tier_description: str
    confidence_score: float
    recommended_action: str
    recommended_action_description: str
    recommended_interest_rate: Optional[float]
    recommended_interest_rate_adjustment_bps: float
    recommended_loan_grade: str

    def to_dict(self) -> dict:
        """Plain-dict representation (JSON-serializable, Streamlit-friendly)."""
        return {
            "default_probability": self.default_probability,
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier,
            "risk_tier_description": self.risk_tier_description,
            "confidence_score": self.confidence_score,
            "recommended_action": self.recommended_action,
            "recommended_action_description": self.recommended_action_description,
            "recommended_interest_rate": self.recommended_interest_rate,
            "recommended_interest_rate_adjustment_bps": self.recommended_interest_rate_adjustment_bps,
            "recommended_loan_grade": self.recommended_loan_grade,
        }


class RiskScoringEngine:
    """
    Loads one fitted Phase 3 model (preprocessor + classifier `Pipeline`)
    and turns raw borrower feature rows into risk scores and lending
    recommendations, using a `RiskThresholdConfig` for every business
    boundary.

    Parameters
    ----------
    model_key : str
        One of "logistic_regression", "random_forest", "xgboost".
        Defaults to `config.PRODUCTION_MODEL_KEY` (the model recommended
        by Phase 3's comparison table).
    threshold_config : RiskThresholdConfig, optional
        Business-policy thresholds. Defaults to
        `configurable_thresholds.load_threshold_config()` (loads
        `reports/risk_threshold_config.json`, bootstrapping it with
        built-in defaults on first run).
    pipeline : Pipeline, optional
        Inject an already-loaded, already-fitted `Pipeline` directly
        (e.g. from an in-memory Phase 3 `ModelResult.best_estimator`)
        instead of reading one from disk -- primarily for notebook and
        unit-test use where retraining a model just to test this class
        would be wasteful.

    Examples
    --------
    >>> engine = RiskScoringEngine()  # doctest: +SKIP
    >>> summary = engine.generate_prediction_summary(borrower_row)  # doctest: +SKIP
    >>> summary.risk_tier  # doctest: +SKIP
    'High Risk'
    """

    def __init__(
        self,
        model_key: str = config.PRODUCTION_MODEL_KEY,
        threshold_config: Optional[RiskThresholdConfig] = None,
        pipeline: Optional[Pipeline] = None,
        base_interest_rate: float = 10.0,
    ) -> None:
        if model_key not in MODEL_PATHS:
            raise ValueError(f"Unknown model_key '{model_key}'. Expected one of {list(MODEL_PATHS)}.")

        self.model_key = model_key
        self.model_display_name = model_utils.MODEL_DISPLAY_NAMES[model_key]
        self.threshold_config = threshold_config or load_threshold_config()
        # Base rate a borrower would be offered before any risk-based
        # adjustment -- a business input (this project's lending-policy
        # baseline), not something the model estimates. Exposed as a
        # constructor argument so Streamlit can make it user-adjustable.
        self.base_interest_rate = base_interest_rate

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            logger.info("Loading %s pipeline from %s", self.model_display_name, MODEL_PATHS[model_key])
            self.pipeline = utils.load_object(MODEL_PATHS[model_key])

        # The classification threshold Phase 3 determined minimizes
        # expected business cost for THIS model (see
        # reports/threshold_analysis.joblib / model_utils.recommend_threshold).
        # Used only for the binary predict() convenience method --
        # assign_risk_tier() operates on the full probability, not a
        # single cutoff, since tiers are a spectrum by design.
        self._decision_threshold = self._load_recommended_decision_threshold()

        logger.info(
            "RiskScoringEngine ready: model=%s, decision_threshold=%.2f",
            self.model_display_name, self._decision_threshold,
        )

    def _load_recommended_decision_threshold(self) -> float:
        """
        Read the Phase 3 cost-minimizing decision threshold for this
        model from `reports/threshold_analysis.joblib`. Falls back to
        0.50 with a warning if Phase 3 artifacts are not yet available
        (e.g. a fresh checkout before `run_phase3_pipeline()` has run).
        """
        try:
            threshold_tables = utils.load_object(config.THRESHOLD_ANALYSIS_PATH)
            table = threshold_tables[self.model_key]
            recommended = model_utils.recommend_threshold(table)
            return float(recommended["threshold"])
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                "Could not load Phase 3 recommended threshold for %s (%s) -- "
                "falling back to 0.50.", self.model_key, exc,
            )
            return 0.50

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probability for one or more borrowers.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows -- the fitted
            `Pipeline` handles preprocessing internally.

        Returns
        -------
        np.ndarray
            Predicted probability of default (class 1) per row. Returns
            an empty array (rather than raising sklearn's lower-level
            "minimum of 1 sample required" error) if `X` has zero rows --
            a zero-row batch is a valid, if uninteresting, input (e.g. a
            fully-filtered dashboard view) and should degrade gracefully.
        """
        if len(X) == 0:
            logger.info("predict_probability called with an empty DataFrame -- returning an empty array.")
            return np.array([])
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict a binary default outcome using either the Phase 3
        cost-minimizing threshold for this model (default) or an
        explicitly supplied threshold.

        Parameters
        ----------
        X : pd.DataFrame
        threshold : float, optional
            Overrides the Phase 3 recommended threshold if supplied.

        Returns
        -------
        np.ndarray
            0/1 predictions.
        """
        cutoff = threshold if threshold is not None else self._decision_threshold
        return (self.predict_probability(X) >= cutoff).astype(int)

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def calculate_risk_score(self, probability: float) -> float:
        """
        Convert a raw default probability into a 0-100 risk score
        (0 = safest, 100 = riskiest) for borrower-facing display. A
        simple linear rescaling is used deliberately -- unlike a
        traditional credit score, this project has no external
        benchmark population to calibrate a nonlinear scale against, so
        a transparent, auditable 1:1 mapping (probability x 100) is
        preferred over an opaque nonlinear transform that would imply
        false precision.

        Parameters
        ----------
        probability : float
            Predicted default probability in [0, 1].

        Returns
        -------
        float
            Risk score in [0, 100].
        """
        return round(float(probability) * 100, 1)

    def calculate_confidence_score(self, probability: float) -> float:
        """
        A 0-100 "how confident is the model in this classification"
        score, based on distance from the indifference point (0.50).
        A probability of exactly 0.50 (maximally ambiguous) scores 0;
        a probability of 0.0 or 1.0 (maximally decisive) scores 100.

        This is a CALIBRATION-INDEPENDENT confidence proxy (it does not
        assume the probability is perfectly calibrated) -- it measures
        how far the prediction sits from the model's own decision
        boundary, not literal certainty. This distinction is worth
        keeping in mind: a confidently-wrong model still scores high
        confidence here, which is why Phase 3's calibration-error metric
        (see `model_utils.expected_calibration_error`) remains the
        relevant check on whether probabilities can be trusted at face
        value.

        Parameters
        ----------
        probability : float

        Returns
        -------
        float
            Confidence score in [0, 100].
        """
        return round(abs(float(probability) - 0.5) * 2 * 100, 1)

    def assign_risk_tier(self, probability: float) -> str:
        """Return the risk-tier name for a predicted default probability (delegates to `RiskThresholdConfig`)."""
        return self.threshold_config.get_tier(probability)

    def recommend_lending_action(self, risk_tier: str) -> str:
        """Return the recommended lending action for a risk tier (delegates to `RiskThresholdConfig`)."""
        return self.threshold_config.get_action(risk_tier)

    def recommend_interest_rate(self, risk_tier: str, base_rate: Optional[float] = None) -> float:
        """
        Recommend an interest rate for a borrower by applying the
        configured basis-point adjustment for their risk tier to a base
        rate.

        Parameters
        ----------
        risk_tier : str
        base_rate : float, optional
            Overrides `self.base_interest_rate` if supplied (e.g. a
            product-specific base rate from a future Streamlit input).

        Returns
        -------
        float
            Recommended annual interest rate, in percent.
        """
        rate = base_rate if base_rate is not None else self.base_interest_rate
        adjustment_pct_points = self.threshold_config.get_rate_adjustment_bps(risk_tier) / 100.0
        return round(rate + adjustment_pct_points, 2)

    def recommend_loan_grade(self, probability: float) -> str:
        """Return the model-driven loan-grade letter for a predicted default probability (delegates to `RiskThresholdConfig`)."""
        return self.threshold_config.get_loan_grade(probability)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def generate_prediction_summary(
        self, borrower: pd.DataFrame, base_rate: Optional[float] = None,
    ) -> PredictionSummary:
        """
        Run the full risk-scoring workflow for exactly one borrower:
        probability -> risk score -> risk tier -> confidence ->
        recommended action / interest rate / loan grade.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw (pre-preprocessing) borrower features.
        base_rate : float, optional
            Passed through to `recommend_interest_rate`.

        Returns
        -------
        PredictionSummary

        Raises
        ------
        ValueError
            If `borrower` does not contain exactly one row.
        """
        if len(borrower) != 1:
            raise ValueError(
                f"generate_prediction_summary expects exactly one borrower row, got {len(borrower)}. "
                f"Use predict_probability()/predict() directly for batch scoring."
            )

        probability = float(self.predict_probability(borrower)[0])
        risk_score = self.calculate_risk_score(probability)
        risk_tier = self.assign_risk_tier(probability)
        confidence = self.calculate_confidence_score(probability)
        action = self.recommend_lending_action(risk_tier)
        rate = self.recommend_interest_rate(risk_tier, base_rate=base_rate)
        rate_adjustment = self.threshold_config.get_rate_adjustment_bps(risk_tier)
        grade = self.recommend_loan_grade(probability)

        return PredictionSummary(
            default_probability=probability,
            risk_score=risk_score,
            risk_tier=risk_tier,
            risk_tier_description=self.threshold_config.get_tier_description(risk_tier),
            confidence_score=confidence,
            recommended_action=action,
            recommended_action_description=self.threshold_config.get_action_description(risk_tier),
            recommended_interest_rate=rate,
            recommended_interest_rate_adjustment_bps=rate_adjustment,
            recommended_loan_grade=grade,
        )

    def generate_batch_summary(self, X: pd.DataFrame, base_rate: Optional[float] = None) -> pd.DataFrame:
        """
        Vectorized equivalent of `generate_prediction_summary` for many
        borrowers at once -- used for portfolio-level reporting (e.g.
        "what does the risk-tier distribution of this loan book look
        like") rather than a single application-time decision.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-preprocessing) borrower feature rows.
        base_rate : float, optional

        Returns
        -------
        pd.DataFrame
            One row per borrower with the same fields as
            `PredictionSummary.to_dict()`.
        """
        probabilities = self.predict_probability(X)
        rows = []
        for probability in probabilities:
            risk_tier = self.assign_risk_tier(probability)
            rows.append({
                "default_probability": probability,
                "risk_score": self.calculate_risk_score(probability),
                "risk_tier": risk_tier,
                "confidence_score": self.calculate_confidence_score(probability),
                "recommended_action": self.recommend_lending_action(risk_tier),
                "recommended_interest_rate": self.recommend_interest_rate(risk_tier, base_rate=base_rate),
                "recommended_loan_grade": self.recommend_loan_grade(probability),
            })
        return pd.DataFrame(rows, index=X.index)

    def export_prediction_report(self, borrower: pd.DataFrame, base_rate: Optional[float] = None) -> "interpretation_utils.ExportableReport":
        """
        Build an `ExportableReport` (Markdown/JSON-ready) summarizing one
        borrower's risk score and recommendation -- the "Prediction
        Summary" / "Risk Assessment" exportable report required by
        Phase 4A. Does not include SHAP-based explanation text; see
        `ExplainabilityEngine.generate_business_summary` for the fuller
        narrative that combines this with feature attribution.

        Parameters
        ----------
        borrower : pd.DataFrame
            Exactly one row of raw borrower features.
        base_rate : float, optional

        Returns
        -------
        interpretation_utils.ExportableReport
        """
        summary = self.generate_prediction_summary(borrower, base_rate=base_rate)
        sections = {
            "Model": f"{self.model_display_name} (production scoring model)",
            "Risk Assessment": (
                f"Default probability: {summary.default_probability:.1%}\n\n"
                f"Risk score: {summary.risk_score}/100\n\n"
                f"Risk tier: {summary.risk_tier} -- {summary.risk_tier_description}\n\n"
                f"Confidence: {summary.confidence_score}/100"
            ),
            "Recommendation": (
                f"Action: {summary.recommended_action} -- {summary.recommended_action_description}\n\n"
                f"Recommended interest rate: {summary.recommended_interest_rate}% "
                f"({summary.recommended_interest_rate_adjustment_bps:+.0f} bps vs. base rate)\n\n"
                f"Model-driven loan grade: {summary.recommended_loan_grade}"
            ),
        }
        return interpretation_utils.ExportableReport(title="Borrower Risk Assessment", sections=sections)


# ---------------------------------------------------------------------------
# Expanded threshold optimization (extends Phase 3's threshold_metrics_table)
# ---------------------------------------------------------------------------


def expand_threshold_analysis(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds=config.THRESHOLD_GRID,
) -> pd.DataFrame:
    """
    Extend Phase 3's `model_utils.threshold_metrics_table` with two
    additional lending-specific columns Phase 4A calls for: approval
    rate (share of loans the model would approve at this threshold) and
    false-positive/false-negative RATES (as opposed to Phase 3's raw
    counts folded into expected cost) -- both are business-facing
    figures a credit-policy stakeholder would ask for directly rather
    than infer from a cost number.

    Design decision: this REUSES `model_utils.threshold_metrics_table`
    (does not recompute precision/recall/F1/expected cost from scratch)
    and only adds the two new columns, consistent with the project rule
    against duplicating logic across phases.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_proba : array-like of predicted probabilities
    thresholds : sequence of float

    Returns
    -------
    pd.DataFrame
        Phase 3's threshold table plus `approval_rate`,
        `false_positive_rate`, `false_negative_rate` columns.
    """
    table = model_utils.threshold_metrics_table(y_true, y_proba, thresholds=thresholds)
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    n = len(y_true_arr)

    approval_rates, fpr_list, fnr_list = [], [], []
    for t in table["threshold"]:
        y_pred = (y_proba_arr >= t).astype(int)
        tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
        fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
        tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
        approval_rates.append((tn + fn) / n)  # predicted "not default" = approved
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else np.nan)
        fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else np.nan)

    table["approval_rate"] = approval_rates
    table["false_positive_rate"] = fpr_list
    table["false_negative_rate"] = fnr_list
    return table


def plot_expanded_threshold_analysis(
    threshold_table: pd.DataFrame, recommended_threshold: float, model_display_name: str,
) -> plt.Figure:
    """
    Six-panel visualization of the expanded threshold analysis:
    threshold vs. precision, recall, F1, approval rate, false-positive
    rate, and false-negative rate -- the full Phase 4A "Threshold
    Optimization" visualization requirement in one figure.

    Parameters
    ----------
    threshold_table : pd.DataFrame
        Output of `expand_threshold_analysis`.
    recommended_threshold : float
        Vertical reference line (e.g. Phase 3's cost-minimizing threshold).
    model_display_name : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = [
        ("precision", "Precision", "#2E86AB"),
        ("recall", "Recall", "#C0392B"),
        ("f1_score", "F1 Score", "#27AE60"),
        ("approval_rate", "Approval Rate", "#8E44AD"),
        ("false_positive_rate", "False Positive Rate", "#F39C12"),
        ("false_negative_rate", "False Negative Rate", "#E74C3C"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (column, label, color) in zip(axes.ravel(), panels):
        ax.plot(threshold_table["threshold"], threshold_table[column], color=color, linewidth=2.2)
        ax.axvline(recommended_threshold, color="black", linestyle=":", linewidth=1.3)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        f"{model_display_name}: Threshold Optimization "
        f"(recommended = {recommended_threshold:.2f}, dotted line)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    return fig
