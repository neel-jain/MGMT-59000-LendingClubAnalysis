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
    "app_pages/model_comparison.py",
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
    """Every page in the Phase 5 brief (8 pages) must load cleanly -- no broken navigation, no import errors."""
    at = _fresh_app()
    at.switch_page(page_file)
    at.run()
    assert not at.exception, f"Page {page_file} raised: {at.exception}"


def test_all_eight_pages_are_present():
    """Sanity check that no page from the Phase 5 brief is missing."""
    assert len(PAGE_FILES) == 8
    for page_file in PAGE_FILES:
        assert (PROJECT_ROOT / "app" / page_file).exists(), f"Missing page file: {page_file}"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def test_sidebar_model_selector_present(app):
    assert len(app.sidebar.selectbox) >= 1


def test_sidebar_theme_toggle_present(app):
    assert len(app.sidebar.radio) >= 1


def test_sidebar_selecting_different_model_reruns_without_error(app):
    selectbox = app.sidebar.selectbox[0]
    original = selectbox.value
    other_options = [o for o in selectbox.options if o != original]
    selectbox.set_value(other_options[0]).run()
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


def test_business_insights_covers_all_seven_research_questions():
    at = _fresh_app()
    at.switch_page("app_pages/business_insights.py")
    at.run()
    assert not at.exception
    headers_text = " ".join(m.value for m in at.markdown if m.value.startswith("####"))
    for i in range(1, 8):
        assert f"RQ{i}:" in headers_text


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
