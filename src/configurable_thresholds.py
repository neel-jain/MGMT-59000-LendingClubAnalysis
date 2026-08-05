"""
configurable_thresholds.py
============================
Centralized, hot-reloadable BUSINESS POLICY configuration for risk
tiers, lending actions, interest-rate adjustments, and loan-grade
assignment.

Design decision: this is deliberately a SEPARATE module from
`src/config.py`. `config.py` holds engineering constants (file paths, CV
folds, hyperparameter search spaces) that a developer changes when the
pipeline itself changes. This module holds BUSINESS POLICY constants
(where the "Low Risk" cutoff sits, how many basis points a "High Risk"
borrower's rate is adjusted) that a credit-risk or lending-operations
stakeholder should be able to change WITHOUT touching source code or
redeploying the application -- hence every value here is wrapped in a
dataclass with a JSON load/save API (`RiskThresholdConfig.load` /
`.save`), not just a bare module-level constant.

Usage
-----
    from src.configurable_thresholds import load_threshold_config

    thresholds = load_threshold_config()   # reads reports/risk_threshold_config.json
                                            # if present, else uses built-in
                                            # defaults AND writes them to disk
                                            # so they become editable going forward.
    thresholds.get_tier(0.42)              # -> "High Risk"
    thresholds.get_action("High Risk")     # -> "Manual Review"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src import config, utils

logger = utils.get_logger(__name__)


@dataclass
class RiskTierBoundary:
    """One risk tier's probability range and borrower-facing description."""

    name: str
    min_probability: float
    max_probability: float
    description: str


@dataclass
class LendingActionRule:
    """The recommended lending action and rationale for one risk tier."""

    tier: str
    action: str
    description: str


@dataclass
class LoanGradeBand:
    """One LendingClub-style letter-grade band, keyed on default probability."""

    grade: str
    min_probability: float
    max_probability: float


@dataclass
class RiskThresholdConfig:
    """
    The complete set of business-policy thresholds used by
    `RiskScoringEngine` (see `src/risk_scoring.py`).

    All boundaries are expressed as PREDICTED DEFAULT PROBABILITY in
    [0, 1] (the model's `predict_proba(...)[:, 1]` output) so every rule
    in this file operates on the same scale regardless of which of the
    three Phase 3 models is in use.

    Defaults below are informed by, but distinct from, Phase 3's
    statistically-optimal (cost-minimizing) decision threshold per model
    (`reports/threshold_analysis.joblib`): that threshold answers "where
    do we draw the accept/decline line to minimize expected cost", while
    the tiers here answer the broader business question "how do we
    communicate and act on a *spectrum* of risk, including partial
    actions like manual review or a rate markup, not just a binary
    accept/decline." A stakeholder is free to override these without
    touching the Phase 3 model or its cost-minimizing threshold.
    """

    risk_tiers: List[RiskTierBoundary] = field(default_factory=lambda: [
        RiskTierBoundary(
            "Low Risk", 0.00, 0.15,
            "Strong repayment profile; default probability is low relative to the portfolio.",
        ),
        RiskTierBoundary(
            "Moderate Risk", 0.15, 0.35,
            "Acceptable repayment profile with some risk factors present.",
        ),
        RiskTierBoundary(
            "High Risk", 0.35, 0.60,
            "Elevated default probability; multiple risk factors present.",
        ),
        RiskTierBoundary(
            "Very High Risk", 0.60, 1.01,  # 1.01 so probability == 1.0 is inclusive
            "Default probability is high; approval is not recommended without mitigation.",
        ),
    ])

    lending_actions: List[LendingActionRule] = field(default_factory=lambda: [
        LendingActionRule(
            "Low Risk", "Approve",
            "Proceed with standard underwriting terms.",
        ),
        LendingActionRule(
            "Moderate Risk", "Approve with Conditions",
            "Approve with a rate adjustment and/or reduced principal to price in the added risk.",
        ),
        LendingActionRule(
            "High Risk", "Manual Review",
            "Route to a credit analyst for manual review before a decision is made.",
        ),
        LendingActionRule(
            "Very High Risk", "Decline",
            "Decline under standard policy; only proceed via an exception process.",
        ),
    ])

    # Interest-rate adjustment, in basis points (1 bps = 0.01 percentage
    # point), applied on top of a borrower's base approved rate. Positive
    # = rate increase (compensates for added risk); negative = rate
    # discount (rewards especially low risk).
    interest_rate_adjustment_bps: Dict[str, float] = field(default_factory=lambda: {
        "Low Risk": -50.0,
        "Moderate Risk": 0.0,
        "High Risk": 150.0,
        "Very High Risk": 400.0,
    })

    # LendingClub-style letter grade purely as a function of predicted
    # default probability (distinct from the historical `grade` FEATURE
    # in the training data, which was assigned by LendingClub's own
    # underwriting at origination -- this is this project's model-driven
    # analogue, used for borrower-facing communication).
    loan_grade_bands: List[LoanGradeBand] = field(default_factory=lambda: [
        LoanGradeBand("A", 0.00, 0.08),
        LoanGradeBand("B", 0.08, 0.15),
        LoanGradeBand("C", 0.15, 0.25),
        LoanGradeBand("D", 0.25, 0.40),
        LoanGradeBand("E", 0.40, 0.55),
        LoanGradeBand("F", 0.55, 0.70),
        LoanGradeBand("G", 0.70, 1.01),
    ])

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def get_tier(self, probability: float) -> str:
        """
        Return the risk-tier name whose [min_probability, max_probability)
        range contains `probability`.

        Parameters
        ----------
        probability : float
            Predicted default probability in [0, 1].

        Returns
        -------
        str
            Risk tier name (e.g. "High Risk"). Falls back to the last
            (highest) tier if `probability` exceeds every configured
            upper bound, so this never raises for a valid probability.
        """
        for tier in self.risk_tiers:
            if tier.min_probability <= probability < tier.max_probability:
                return tier.name
        return self.risk_tiers[-1].name

    def get_tier_description(self, tier_name: str) -> str:
        """Return the borrower-facing description for a risk-tier name."""
        for tier in self.risk_tiers:
            if tier.name == tier_name:
                return tier.description
        raise ValueError(f"Unknown risk tier '{tier_name}'.")

    def get_action(self, tier_name: str) -> str:
        """Return the recommended lending action for a risk-tier name."""
        for rule in self.lending_actions:
            if rule.tier == tier_name:
                return rule.action
        raise ValueError(f"No lending-action rule configured for tier '{tier_name}'.")

    def get_action_description(self, tier_name: str) -> str:
        """Return the rationale text for the lending action assigned to a tier."""
        for rule in self.lending_actions:
            if rule.tier == tier_name:
                return rule.description
        raise ValueError(f"No lending-action rule configured for tier '{tier_name}'.")

    def get_rate_adjustment_bps(self, tier_name: str) -> float:
        """Return the interest-rate adjustment (basis points) for a risk-tier name."""
        if tier_name not in self.interest_rate_adjustment_bps:
            raise ValueError(f"No rate adjustment configured for tier '{tier_name}'.")
        return self.interest_rate_adjustment_bps[tier_name]

    def get_loan_grade(self, probability: float) -> str:
        """Return the model-driven loan-grade letter for a predicted default probability."""
        for band in self.loan_grade_bands:
            if band.min_probability <= probability < band.max_probability:
                return band.grade
        return self.loan_grade_bands[-1].grade

    # ------------------------------------------------------------------
    # Serialization (the "change without modifying source code" API)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RiskThresholdConfig":
        """Deserialize from a plain dict (inverse of `to_dict`)."""
        return cls(
            risk_tiers=[RiskTierBoundary(**t) for t in data["risk_tiers"]],
            lending_actions=[LendingActionRule(**a) for a in data["lending_actions"]],
            interest_rate_adjustment_bps=dict(data["interest_rate_adjustment_bps"]),
            loan_grade_bands=[LoanGradeBand(**g) for g in data["loan_grade_bands"]],
        )

    def save(self, path: Path = config.RISK_THRESHOLD_CONFIG_PATH) -> None:
        """
        Persist this configuration as human-editable JSON. A lending-
        operations stakeholder can open this file directly and change
        any boundary, action, or adjustment without touching Python.

        Parameters
        ----------
        path : Path
            Destination JSON file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved risk threshold configuration to %s", path)

    @classmethod
    def load(cls, path: Path = config.RISK_THRESHOLD_CONFIG_PATH) -> "RiskThresholdConfig":
        """
        Load configuration from JSON if it exists; otherwise return
        built-in defaults AND write them to `path` so the file exists
        for future editing (self-bootstrapping on first run).

        Parameters
        ----------
        path : Path
            Source/destination JSON file path.

        Returns
        -------
        RiskThresholdConfig
        """
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded risk threshold configuration from %s", path)
            return cls.from_dict(data)

        logger.info(
            "No risk threshold configuration found at %s -- using built-in "
            "defaults and writing them to disk for future editing.", path,
        )
        default_config = cls()
        default_config.save(path)
        return default_config

    def validate(self) -> None:
        """
        Sanity-check that risk tiers form a contiguous, non-overlapping
        partition of [0, 1] and that every tier has a matching lending-
        action rule and rate adjustment. Raises `ValueError` on the
        first problem found -- intended to be called after loading a
        stakeholder-edited JSON file, before the configuration is used
        for any actual scoring.
        """
        sorted_tiers = sorted(self.risk_tiers, key=lambda t: t.min_probability)
        if abs(sorted_tiers[0].min_probability - 0.0) > 1e-9:
            raise ValueError("Risk tiers must start at probability 0.0.")
        if sorted_tiers[-1].max_probability < 1.0:
            raise ValueError("Risk tiers must cover up to probability 1.0.")
        for prev, curr in zip(sorted_tiers, sorted_tiers[1:]):
            if abs(prev.max_probability - curr.min_probability) > 1e-9:
                raise ValueError(
                    f"Gap or overlap between tiers '{prev.name}' and '{curr.name}'."
                )

        tier_names = {t.name for t in self.risk_tiers}
        action_tiers = {a.tier for a in self.lending_actions}
        rate_tiers = set(self.interest_rate_adjustment_bps.keys())
        if tier_names != action_tiers:
            raise ValueError(
                f"Lending-action rules must cover exactly the configured tiers. "
                f"Tiers={tier_names}, action rule tiers={action_tiers}."
            )
        if tier_names != rate_tiers:
            raise ValueError(
                f"Interest-rate adjustments must cover exactly the configured tiers. "
                f"Tiers={tier_names}, rate adjustment tiers={rate_tiers}."
            )
        logger.info("Risk threshold configuration validated successfully.")


def load_threshold_config(path: Optional[Path] = None) -> RiskThresholdConfig:
    """
    Convenience module-level function: load (or bootstrap) the risk
    threshold configuration and validate it before returning.

    Parameters
    ----------
    path : Path, optional
        Defaults to `config.RISK_THRESHOLD_CONFIG_PATH`.

    Returns
    -------
    RiskThresholdConfig
    """
    resolved_path = path if path is not None else config.RISK_THRESHOLD_CONFIG_PATH
    threshold_config = RiskThresholdConfig.load(resolved_path)
    threshold_config.validate()
    return threshold_config
