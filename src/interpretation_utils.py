"""
interpretation_utils.py
=========================
Reusable functions that translate technical model output (feature
names, SHAP values, per-group metrics) into executive/business-friendly
language and exportable report formats.

This module is intentionally free of any modeling or SHAP-computation
logic (that lives in `explainability.py` / `model_utils.py`) so it can
be imported by both `explainability.py` and `risk_scoring.py` without
either depending on the other's internals -- a shared "how do we talk
about this" layer rather than a shared "how do we compute this" layer.

Organized into sections:
    1. Feature-name humanization
    2. Research-question linkage
    3. Business-summary text generation
    4. Fairness reporting
    5. Exportable report formatting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import config, model_utils, utils

logger = utils.get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. FEATURE-NAME HUMANIZATION
# ---------------------------------------------------------------------------

# Business-friendly labels for every RAW column that can reach the
# preprocessing pipeline (see config.NUMERIC_FEATURES /
# ONEHOT_CATEGORICAL_FEATURES / ORDINAL_CATEGORICAL_FEATURES). Centralized
# here rather than scattered across plotting code so every report,
# chart, and generated sentence in Phase 4A uses identical wording.
FEATURE_LABELS: Dict[str, str] = {
    "loan_amnt": "Loan amount",
    "int_rate": "Interest rate",
    "installment": "Monthly installment",
    "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio",
    "delinq_2yrs": "Delinquencies (past 2 years)",
    "open_acc": "Open credit accounts",
    "pub_rec": "Public derogatory records",
    "revol_bal": "Revolving credit balance",
    "revol_util": "Revolving credit utilization",
    "total_acc": "Total credit accounts",
    "mort_acc": "Mortgage accounts",
    "pub_rec_bankruptcies": "Bankruptcies on record",
    "emp_length_years": "Employment length (years)",
    "term": "Loan term",
    "home_ownership": "Home ownership status",
    "verification_status": "Income verification status",
    "purpose": "Loan purpose",
    "initial_list_status": "Initial listing status",
    "application_type": "Application type",
    "grade": "LendingClub credit grade",
}

# Preprocessing ColumnTransformer prefixes (see
# utils.build_preprocessing_pipeline), longest-first so "onehot_categorical"
# is checked before any shorter accidental substring match.
_TRANSFORMER_PREFIXES: Tuple[str, ...] = (
    "onehot_categorical__",
    "ordinal_categorical__",
    "numeric__",
)

# All raw column names that can appear inside a one-hot feature name,
# sorted longest-first so e.g. "home_ownership" is matched before a
# shorter false-positive prefix could be.
_ONEHOT_BASE_COLUMNS: Tuple[str, ...] = tuple(
    sorted(config.ONEHOT_CATEGORICAL_FEATURES, key=len, reverse=True)
)


def humanize_feature_name(technical_name: str) -> str:
    """
    Convert a preprocessed feature name (as produced by
    `preprocessor.get_feature_names_out()`, e.g.
    `"onehot_categorical__purpose_debt_consolidation"` or
    `"numeric__dti"`) into a business-friendly label (e.g.
    `"Loan purpose: Debt Consolidation"` or `"Debt-to-income ratio"`).

    Parameters
    ----------
    technical_name : str
        A single output feature name from the fitted preprocessor.

    Returns
    -------
    str
        Human-readable label. Falls back to a lightly cleaned version of
        the original name (prefix stripped, underscores replaced with
        spaces) if no mapping is found, so this never raises and never
        silently drops information.
    """
    remainder = technical_name
    is_onehot = False
    for prefix in _TRANSFORMER_PREFIXES:
        if technical_name.startswith(prefix):
            remainder = technical_name[len(prefix):]
            is_onehot = prefix == "onehot_categorical__"
            break

    if is_onehot:
        for base_col in _ONEHOT_BASE_COLUMNS:
            marker = base_col + "_"
            if remainder.startswith(marker):
                category_value = remainder[len(marker):]
                base_label = FEATURE_LABELS.get(base_col, base_col.replace("_", " ").title())
                category_label = category_value.replace("_", " ").strip().title()
                return f"{base_label}: {category_label}"
        # Unrecognized one-hot base column -- fall through to generic cleanup.

    if remainder in FEATURE_LABELS:
        return FEATURE_LABELS[remainder]

    return remainder.replace("_", " ").strip().capitalize()


def humanize_feature_table(df: pd.DataFrame, feature_column: str = "feature") -> pd.DataFrame:
    """
    Add a `feature_label` column to any feature-importance-style table
    (from `model_utils.py` or `explainability.py`) by applying
    `humanize_feature_name` to every value in `feature_column`.

    Parameters
    ----------
    df : pd.DataFrame
    feature_column : str
        Column containing raw/technical feature names.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with an added `feature_label` column, inserted
        immediately after `feature_column`.
    """
    result = df.copy()
    labels = result[feature_column].apply(humanize_feature_name)
    insert_at = result.columns.get_loc(feature_column) + 1
    result.insert(insert_at, "feature_label", labels)
    return result


# ---------------------------------------------------------------------------
# 2. RESEARCH-QUESTION LINKAGE
# ---------------------------------------------------------------------------

# Maps a RAW feature name to the project's Phase 2 research questions it
# most directly informs. Used so every generated explanation can cite
# which research question its evidence supports, per the Phase 4A
# "Research Question Support" requirement.
RESEARCH_QUESTION_MAP: Dict[str, str] = {
    "grade": "RQ2 (are LendingClub grades predictive of default?)",
    "int_rate": "RQ3 (which variables relate to higher interest rates?)",
    "annual_inc": "RQ4 (does income relate to repayment success?)",
    "dti": "RQ5 (does DTI influence default?)",
    "emp_length_years": "RQ6 (does employment length matter?)",
    "loan_amnt": "RQ1 (which borrower characteristics associate with default?)",
    "installment": "RQ1 (which borrower characteristics associate with default?)",
    "home_ownership": "RQ1 (which borrower characteristics associate with default?)",
    "purpose": "RQ1 (which borrower characteristics associate with default?)",
    "revol_util": "RQ1 (which borrower characteristics associate with default?)",
    "delinq_2yrs": "RQ1 (which borrower characteristics associate with default?)",
    "pub_rec": "RQ1 (which borrower characteristics associate with default?)",
    "pub_rec_bankruptcies": "RQ1 (which borrower characteristics associate with default?)",
    "mort_acc": "RQ1 (which borrower characteristics associate with default?)",
    "revol_bal": "RQ1 (which borrower characteristics associate with default?)",
    "total_acc": "RQ1 (which borrower characteristics associate with default?)",
    "open_acc": "RQ1 (which borrower characteristics associate with default?)",
    "verification_status": "RQ1 (which borrower characteristics associate with default?)",
    "term": "RQ1 (which borrower characteristics associate with default?)",
    "application_type": "RQ1 (which borrower characteristics associate with default?)",
    "initial_list_status": "RQ1 (which borrower characteristics associate with default?)",
}


def link_feature_to_research_question(technical_name: str) -> Optional[str]:
    """
    Identify which project research question a (technical or raw)
    feature name most directly supports.

    Parameters
    ----------
    technical_name : str
        Either a raw column name (e.g. "dti") or a preprocessed feature
        name (e.g. "numeric__dti", "onehot_categorical__purpose_other").

    Returns
    -------
    str or None
        The matching research-question text, or None if no raw feature
        name in `RESEARCH_QUESTION_MAP` is a match.
    """
    remainder = technical_name
    for prefix in _TRANSFORMER_PREFIXES:
        if technical_name.startswith(prefix):
            remainder = technical_name[len(prefix):]
            break

    for raw_name, research_question in RESEARCH_QUESTION_MAP.items():
        if remainder == raw_name or remainder.startswith(raw_name + "_"):
            return research_question
    return None


# ---------------------------------------------------------------------------
# 3. BUSINESS-SUMMARY TEXT GENERATION
# ---------------------------------------------------------------------------


def _format_factor_phrase(feature_label: str) -> str:
    """Lower-case the first letter of a feature label for mid-sentence use, preserving acronyms/brand names like 'LendingClub'."""
    if not feature_label:
        return feature_label
    if feature_label[:2].isupper() or feature_label.startswith("LendingClub"):
        return feature_label
    return feature_label[0].lower() + feature_label[1:]


def _join_with_and(items: Sequence[str]) -> str:
    """Join a list of phrases into natural English ('a, b, and c')."""
    items = list(items)
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def generate_borrower_business_summary(
    risk_tier: str,
    default_probability: float,
    top_risk_factors: Sequence[str],
    top_protective_factors: Sequence[str],
    recommended_action: str,
) -> str:
    """
    Generate a credit-analyst-style plain-language paragraph for one
    borrower, in the style of:

        "This borrower is classified as High Risk primarily due to a
        high debt-to-income ratio, elevated interest rate, and lower
        Lending Club credit grade."

    Deliberately avoids ML terminology (no "SHAP value", "feature
    importance", "model", "probability distribution", etc.) -- states
    the risk tier and the plain-English factors driving it only.

    Parameters
    ----------
    risk_tier : str
        e.g. "High Risk" (from `RiskThresholdConfig.get_tier`).
    default_probability : float
        Predicted default probability in [0, 1] (reported as a
        round-number percentage risk, not the raw model score).
    top_risk_factors : sequence of str
        Human-readable feature labels driving default risk UP,
        strongest first (already humanized -- see `humanize_feature_name`).
    top_protective_factors : sequence of str
        Human-readable feature labels driving default risk DOWN,
        strongest first.
    recommended_action : str
        e.g. "Manual Review" (from `RiskThresholdConfig.get_action`).

    Returns
    -------
    str
        A short (2-4 sentence) executive-friendly paragraph.
    """
    risk_pct = round(default_probability * 100)
    sentences = [
        f"This borrower is classified as {risk_tier} (estimated {risk_pct}% "
        f"likelihood of default)."
    ]

    if top_risk_factors:
        phrases = [_format_factor_phrase(f) for f in top_risk_factors[:3]]
        factor_text = _join_with_and(phrases)
        sentences.append(f"This is primarily driven by {factor_text}.")

    if top_protective_factors:
        phrases = [_format_factor_phrase(f) for f in top_protective_factors[:2]]
        factor_text = _join_with_and(phrases)
        sentences.append(f"On the positive side, {factor_text} help offset some of this risk.")

    sentences.append(f"Recommended action: {recommended_action}.")
    return " ".join(sentences)


def generate_global_business_summary(
    model_display_name: str,
    top_features: Sequence[str],
    least_influential_features: Sequence[str],
) -> str:
    """
    Generate an executive-friendly paragraph summarizing what drives the
    model's predictions in general (not for one borrower).

    Parameters
    ----------
    model_display_name : str
        e.g. "XGBoost".
    top_features : sequence of str
        Human-readable labels of the most influential features overall,
        strongest first.
    least_influential_features : sequence of str
        Human-readable labels of the least influential features
        considered by the model.

    Returns
    -------
    str
    """
    top_text = _join_with_and([_format_factor_phrase(f) for f in top_features[:5]])
    sentences = [
        f"Across the full borrower population, the {model_display_name} model's "
        f"predictions are most strongly driven by {top_text}."
    ]
    if least_influential_features:
        least_text = _join_with_and([_format_factor_phrase(f) for f in least_influential_features[:3]])
        sentences.append(
            f"By comparison, {least_text} contribute comparatively little to the "
            f"model's predictions and are lower-priority for data-collection efforts."
        )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# 4. FAIRNESS REPORTING
# ---------------------------------------------------------------------------


def fairness_report(
    X: pd.DataFrame, y_true: pd.Series, y_proba: np.ndarray,
    group_columns: Sequence[str], threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute the standard classification-metric suite (reusing
    `model_utils.compute_classification_metrics`) separately WITHIN each
    category of each specified grouping column, to surface whether model
    performance is meaningfully uneven across borrower subgroups.

    Design decision: this reuses `model_utils.compute_classification_metrics`
    rather than duplicating metric logic, and operates on RAW (pre-
    preprocessing) columns so results are reported in business terms
    ("MORTGAGE homeowners", not a one-hot column index).

    Important limitation (reported alongside every fairness table, per
    the "avoid unsupported fairness claims" requirement): this dataset
    contains no legally protected class attributes (race, gender, age,
    religion, national origin, etc.) -- it was never collected with
    them. This report can only speak to performance parity across the
    BUSINESS/FINANCIAL attributes actually present (income level,
    employment tenure, homeownership, loan purpose, credit grade). It
    cannot support or refute any claim about fairness with respect to
    legally protected classes, and should not be represented as doing so.

    Parameters
    ----------
    X : pd.DataFrame
        Raw (pre-preprocessing) feature frame, aligned index-for-index
        with `y_true` and `y_proba`.
    y_true : pd.Series
    y_proba : np.ndarray
        Predicted default probabilities.
    group_columns : sequence of str
        Raw column names to group by (e.g. ["home_ownership", "purpose"]).
        For continuous columns (e.g. "annual_inc"), pass a pre-binned
        column name instead (see `bin_column_for_fairness`).
    threshold : float
        Decision threshold used to compute threshold-dependent metrics.

    Returns
    -------
    pd.DataFrame
        One row per (group_column, group_value) pair, with columns:
        group_column, group_value, n_loans, actual_default_rate, plus
        every metric from `model_utils.compute_classification_metrics`.
    """
    rows = []
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    y_pred_arr = (y_proba_arr >= threshold).astype(int)

    for group_column in group_columns:
        if group_column not in X.columns:
            logger.warning("Fairness group column '%s' not found in X -- skipping.", group_column)
            continue
        for group_value in X[group_column].dropna().unique():
            mask = (X[group_column] == group_value).to_numpy()
            n = int(mask.sum())
            if n < 10:
                # Too few observations for a stable metric estimate --
                # reported as excluded rather than silently shown with
                # misleadingly precise (noisy) numbers.
                logger.info(
                    "Skipping fairness row for %s=%s (n=%d < 10).", group_column, group_value, n,
                )
                continue
            metrics = model_utils.compute_classification_metrics(
                y_true_arr[mask], y_pred_arr[mask], y_proba_arr[mask],
            )
            rows.append({
                "group_column": group_column,
                "group_value": str(group_value),
                "n_loans": n,
                "actual_default_rate": float(y_true_arr[mask].mean()),
                **metrics,
            })

    return pd.DataFrame(rows)


def bin_column_for_fairness(series: pd.Series, n_bins: int = 4, label_prefix: str = "Q") -> pd.Series:
    """
    Quantile-bin a continuous column (e.g. `annual_inc`) into labeled
    groups (Q1..Qn) suitable for `fairness_report`'s `group_columns`.

    Parameters
    ----------
    series : pd.Series
    n_bins : int
    label_prefix : str

    Returns
    -------
    pd.Series
        Categorical series with labels like "Q1", "Q2", ... "Qn".
    """
    labels = [f"{label_prefix}{i + 1}" for i in range(n_bins)]
    return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")


def summarize_fairness_disparities(fairness_table: pd.DataFrame, metric: str = "recall") -> pd.DataFrame:
    """
    For each grouping column in a fairness table, compute the spread
    (max - min) of a chosen metric across that column's groups -- a
    quick way to flag which grouping shows the largest performance gap
    without eyeballing the full table.

    Parameters
    ----------
    fairness_table : pd.DataFrame
        Output of `fairness_report`.
    metric : str
        Column name to compare (e.g. "recall", "roc_auc").

    Returns
    -------
    pd.DataFrame
        One row per group_column: min, max, and spread of `metric`.
    """
    if fairness_table.empty:
        return pd.DataFrame(columns=["group_column", f"{metric}_min", f"{metric}_max", f"{metric}_spread"])

    summary = (
        fairness_table.groupby("group_column")[metric]
        .agg(["min", "max"])
        .reset_index()
    )
    summary["spread"] = summary["max"] - summary["min"]
    summary.columns = ["group_column", f"{metric}_min", f"{metric}_max", f"{metric}_spread"]
    return summary.sort_values(f"{metric}_spread", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. EXPORTABLE REPORT FORMATTING
# ---------------------------------------------------------------------------


@dataclass
class ExportableReport:
    """
    A named bundle of report sections ready to be rendered as
    Markdown/JSON and written to disk -- the shared building block both
    `ExplainabilityEngine` and `RiskScoringEngine` use for every
    "exportable report" required by Phase 4A (Prediction Summary,
    Feature Importance, Executive Report, Borrower Explanation, Risk
    Assessment). Designed to be immediately reusable by a future
    Streamlit "Download report" button (`to_markdown()` / `to_json()`
    output can be handed directly to `st.download_button`).
    """

    title: str
    sections: Dict[str, str]

    def to_markdown(self) -> str:
        """Render as a Markdown document (one `##` heading per section)."""
        lines = [f"# {self.title}", ""]
        for heading, body in self.sections.items():
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(body)
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Render as a plain dict (JSON-serializable)."""
        return {"title": self.title, "sections": self.sections}

    def to_json(self) -> str:
        """Render as a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path, fmt: str = "markdown") -> None:
        """
        Write this report to disk.

        Parameters
        ----------
        path : Path
            Destination file path.
        fmt : str
            One of "markdown", "json", "text". "text" writes the
            Markdown rendering -- useful for plain `.txt` downloads.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            content = self.to_json()
        elif fmt in ("markdown", "text"):
            content = self.to_markdown()
        else:
            raise ValueError(f"Unknown export format '{fmt}'. Expected markdown, json, or text.")
        path.write_text(content, encoding="utf-8")
        logger.info("Saved %s report ('%s') to %s", fmt, self.title, path)


def _format_cell(value) -> str:
    """Format a single DataFrame cell for Markdown-table display."""
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Render a DataFrame as a Markdown table (used by `ExportableReport`
    sections that embed tabular data, e.g. a feature-importance table,
    without pulling in an extra dependency for Markdown table rendering).

    Parameters
    ----------
    df : pd.DataFrame
    max_rows : int
        Row cap to keep exported reports readable; a note is appended
        if rows were truncated.

    Returns
    -------
    str
    """
    truncated = len(df) > max_rows
    display_df = df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in display_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(_format_cell(v) for v in row) + " |"
        for row in display_df.itertuples(index=False)
    ]
    table = "\n".join([header, separator] + rows)
    if truncated:
        table += f"\n\n*(showing first {max_rows} of {len(df)} rows)*"
    return table
