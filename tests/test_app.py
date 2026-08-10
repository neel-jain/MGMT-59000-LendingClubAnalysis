"""
test_app.py
============
Phase 5 test suite for the Streamlit application, using Streamlit's
`AppTest` headless testing API (no browser required).

Verifies:
    - Every page loads without raising an exception.
    - Navigation between pages works.
    - The Borrower Risk Prediction form can be submitted end-to-end.
    - Sidebar controls (model selector, filters) render without error.

Run with:
    pytest tests/test_app.py -v

Note: these tests require the Phase 1/3/4A/4B artifacts (data splits,
trained models, explainability/segmentation artifacts) to already exist
on disk -- run `python -m src.train_models` and persist the Phase 4A/4B
artifacts first if running this suite from a fresh checkout (see
`tests/generate_synthetic_fixture.py` for a quick synthetic stand-in).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

APP_ENTRY = str(PROJECT_ROOT / "app" / "app.py")

PAGE_FILES = [
    "app_pages/executive_dashboard.py",
    "app_pages/exploratory_analysis.py",
    "app_pages/borrower_risk_prediction.py",
    "app_pages/borrower_segmentation.py",
    "app_pages/business_insights.py",
    "app_pages/model_explainability.py",
    "app_pages/about_project.py",
]


def _fresh_app() -> AppTest:
    """Return a freshly-run AppTest instance (default page: Executive Dashboard)."""
    at = AppTest.from_file(APP_ENTRY, default_timeout=180)
    at.run()
    return at


@pytest.fixture(scope="module")
def app():
    return _fresh_app()


# ---------------------------------------------------------------------------
# Navigation / no-broken-pages
# ---------------------------------------------------------------------------


def test_default_page_loads_without_exception(app):
    assert not app.exception


def test_default_page_shows_kpi_metrics(app):
    assert len(app.metric) > 0


@pytest.mark.parametrize("page_file", PAGE_FILES)
def test_every_page_loads_without_exception(page_file):
    """Every visible page in the Phase 5 brief must load cleanly -- no broken navigation, no import errors."""
    at = _fresh_app()
    at.switch_page(page_file)
    at.run()
    assert not at.exception, f"Page {page_file} raised: {at.exception}"


def test_all_pages_are_present():
    """Sanity check that no visible page from the brief is missing."""
    assert len(PAGE_FILES) == 7
    for page_file in PAGE_FILES:
        assert (PROJECT_ROOT / "app" / page_file).exists(), f"Missing page file: {page_file}"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def test_sidebar_advanced_toggle_present(app):
    assert len(app.sidebar.checkbox) >= 1


def test_sidebar_toggling_advanced_reruns_without_error(app):
    checkbox = app.sidebar.checkbox[0]
    checkbox.set_value(not checkbox.value).run()
    assert not app.exception


# ---------------------------------------------------------------------------
# Borrower Risk Prediction: full form submission
# ---------------------------------------------------------------------------


def test_prediction_form_submits_and_shows_results():
    at = _fresh_app()
    at.switch_page("app_pages/borrower_risk_prediction.py")
    at.run()

    predict_buttons = [b for b in at.button if "Predict Risk" in b.label]
    assert len(predict_buttons) == 1

    predict_buttons[0].click().run()
    assert not at.exception
    # Risk tier, probability, confidence, action, rate, grade = 6 metrics.
    assert len(at.metric) == 6


def test_prediction_form_shows_placeholder_before_submission():
    at = _fresh_app()
    at.switch_page("app_pages/borrower_risk_prediction.py")
    at.run()
    assert not at.exception
    assert len(at.metric) == 0  # no prediction metrics before the form is submitted


# ---------------------------------------------------------------------------
# Business Insights: research-question coverage
# ---------------------------------------------------------------------------


def test_business_insights_covers_rq1_and_advanced_toggle():
    at = _fresh_app()
    at.switch_page("app_pages/business_insights.py")
    at.run()
    assert not at.exception
    headers_text = " ".join(m.value for m in at.markdown if m.value.startswith("####"))
    assert "RQ1:" in headers_text
    assert "RQ2:" not in headers_text
    assert "RQ3:" not in headers_text
    assert "RQ4:" not in headers_text
    assert "RQ5:" not in headers_text
    assert "RQ6:" not in headers_text
    assert "RQ7:" not in headers_text
    assert any("Show advanced statistics" in element.label for element in at.sidebar.checkbox)


# ---------------------------------------------------------------------------
# About Project: static content sanity check
# ---------------------------------------------------------------------------


def test_about_project_covers_required_sections():
    at = _fresh_app()
    at.switch_page("app_pages/about_project.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    for required_text in [
        "Business Problem", "Research Questions", "PDID Framework", "Project Methodology",
        "Machine Learning Workflow", "Technology Stack", "Project Limitations",
        "Future Improvements", "4A", "4B",
    ]:
        assert required_text in full_text
