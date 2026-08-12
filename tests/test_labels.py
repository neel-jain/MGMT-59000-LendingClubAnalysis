"""
test_labels.py
================
Unit tests for src/labels.py (chart-facing display labels).

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import labels  # noqa: E402


# ---------------------------------------------------------------------------
# Column-level labels
# ---------------------------------------------------------------------------


def test_column_label_known_columns():
    assert labels.column_label("dti") == "Debt-to-Income Ratio"
    assert labels.column_label("int_rate") == "Interest Rate"
    assert labels.column_label("annual_inc") == "Annual Income"
    assert labels.column_label("loan_amnt") == "Loan Amount"


def test_column_label_table_columns():
    assert labels.column_label("roc_auc") == "ROC-AUC"
    assert labels.column_label("f1_score") == "F1 Score"
    assert labels.column_label("mean_abs_shap") == "Mean |SHAP|"
    assert labels.column_label("segment_name") == "Segment"
    assert labels.column_label("n_borrowers") == "Borrowers"
    assert labels.column_label("typical_income") == "Typical Income"
    assert labels.column_label("average_default_rate") == "Default Rate"


def test_column_label_unknown_falls_back_gracefully():
    result = labels.column_label("some_unknown_column")
    assert result == "Some Unknown Column"


# ---------------------------------------------------------------------------
# Category-value labels
# ---------------------------------------------------------------------------


def test_category_label_purpose_values():
    assert labels.category_label("purpose", "debt_consolidation") == "Debt Consolidation"
    assert labels.category_label("purpose", "credit_card") == "Credit Card"


def test_category_label_home_ownership_uppercase():
    assert labels.category_label("home_ownership", "RENT") == "Rent"
    assert labels.category_label("home_ownership", "MORTGAGE") == "Mortgage"


def test_category_label_initial_list_status_codes():
    assert labels.category_label("initial_list_status", "f") == "Fractional"
    assert labels.category_label("initial_list_status", "w") == "Whole"


def test_category_label_term_strips_leading_space():
    # The cleaned dataset stores term with a leading space (" 36 months").
    assert labels.category_label("term", " 36 months") == "36 Months"
    assert labels.category_label("term", "60 months") == "60 Months"


def test_category_label_unknown_value_falls_back_gracefully():
    result = labels.category_label("purpose", "some_unknown_purpose")
    assert result == "Some Unknown Purpose"
