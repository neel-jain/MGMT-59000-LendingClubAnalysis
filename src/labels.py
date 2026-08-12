"""
labels.py
==========
Chart-facing display labels for the Streamlit app and the notebooks.

This module is the SINGLE SOURCE OF TRUTH for the title-case,
business-friendly labels shown on chart axes, tick labels, and
form/filter dropdowns -- e.g. "Debt-to-Income Ratio" for the `dti`
column and "Debt Consolidation" for the `debt_consolidation` loan
purpose -- so every visualization reads consistently without each
plotting function re-deriving a label.

Design notes
------------
* Deliberately dependency-light (imports only `config`) so it can be
  imported by `eda_utils`, `cluster_visualization`, `model_utils`, and
  the Streamlit app without introducing a circular import
  (`interpretation_utils` -> `model_utils` -> `eda_utils`).
* This title-case layer is SEPARATE from
  `interpretation_utils.FEATURE_LABELS`, which holds sentence-case
  labels ("Debt-to-income ratio") used in generated prose and pinned by
  tests. Do NOT merge the two maps -- they serve different contexts
  (charts vs. sentences) and changing `FEATURE_LABELS` would silently
  alter model-explanation wording.
* Every helper falls back to a cleaned title-case string rather than
  raising, so an unknown column or category value never crashes a chart.
"""

from __future__ import annotations

from typing import Dict

from src import config

# ---------------------------------------------------------------------------
# Column-level display labels (title case, chart-facing)
# ---------------------------------------------------------------------------
# Covers every feature that reaches the preprocessing pipeline plus the
# binary target. Curated here (rather than derived) so abbreviations and
# conventional finance phrasing ("Debt-to-Income Ratio", "Credit
# Utilization") read naturally on an axis.
COLUMN_LABELS: Dict[str, str] = {
    "loan_amnt": "Loan Amount",
    "int_rate": "Interest Rate",
    "installment": "Monthly Installment",
    "annual_inc": "Annual Income",
    "dti": "Debt-to-Income Ratio",
    "delinq_2yrs": "Delinquencies (Past 2 Years)",
    "open_acc": "Open Credit Accounts",
    "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance",
    "revol_util": "Credit Utilization",
    "total_acc": "Total Accounts",
    "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Bankruptcies",
    "emp_length_years": "Employment Length",
    "term": "Loan Term",
    "home_ownership": "Home Ownership",
    "verification_status": "Verification Status",
    "purpose": "Loan Purpose",
    "initial_list_status": "Listing Status",
    "application_type": "Application Type",
    "grade": "Credit Grade",
    config.TARGET_COLUMN: "Default",
}

# ---------------------------------------------------------------------------
# Category-value display labels (per categorical column)
# ---------------------------------------------------------------------------
# Every observed value in the cleaned dataset (verified against
# data/processed/lendingclub_indiana_cleaned.csv) plus a few values the
# prediction form exposes. Keys are stored WITHOUT leading/trailing
# whitespace; `category_label` strips input before lookup.
CATEGORY_LABELS: Dict[str, Dict[str, str]] = {
    "purpose": {
        "car": "Car",
        "credit_card": "Credit Card",
        "debt_consolidation": "Debt Consolidation",
        "home_improvement": "Home Improvement",
        "house": "House",
        "major_purchase": "Major Purchase",
        "medical": "Medical",
        "moving": "Moving",
        "other": "Other",
        "renewable_energy": "Renewable Energy",
        "small_business": "Small Business",
        "vacation": "Vacation",
        "wedding": "Wedding",
    },
    "home_ownership": {
        "ANY": "Any",
        "MORTGAGE": "Mortgage",
        "OWN": "Own",
        "RENT": "Rent",
        "OTHER": "Other",
        "NONE": "None",
    },
    "verification_status": {
        "Not Verified": "Not Verified",
        "Source Verified": "Source Verified",
        "Verified": "Verified",
    },
    "term": {
        "36 months": "36 Months",
        "60 months": "60 Months",
    },
    "application_type": {
        "Individual": "Individual",
        "Joint App": "Joint Application",
    },
    "initial_list_status": {
        "f": "Fractional",
        "w": "Whole",
    },
}

# Small connector words kept lowercase by `smart_title_case` (except as
# the first word), so a fallback label reads "Debt-to-Income" rather than
# "Debt-To-Income".
_LITTLE_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "but", "or", "for", "nor", "of", "on",
    "to", "in", "at", "by", "with",
})


def smart_title_case(text: str) -> str:
    """
    Title-case a snake_case or mixed-case string for a chart label.

    Capitalizes the first letter of each word, keeps small connector
    words ("to", "of", "the", ...) lowercase except as the first word,
    and leaves single-letter codes / numbers intact. Never raises.
    """
    words = str(text).strip().replace("_", " ").split()
    if not words:
        return ""
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        if i > 0 and lower in _LITTLE_WORDS:
            out.append(lower)
        else:
            out.append(word[0].upper() + word[1:].lower())
    return " ".join(out)


def column_label(col: str) -> str:
    """Return the title-case display label for a raw column name."""
    if col in COLUMN_LABELS:
        return COLUMN_LABELS[col]
    return smart_title_case(col)


def category_label(col: str, value) -> str:
    """Return the friendly display label for one category value of a column."""
    key = str(value).strip()
    mapping = CATEGORY_LABELS.get(col)
    if mapping and key in mapping:
        return mapping[key]
    return smart_title_case(key)
