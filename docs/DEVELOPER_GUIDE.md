# Developer Guide

**MGMT 590 LendingClub Loan Default Risk Capstone**

For engineers extending, maintaining, or grading this codebase. Read
`docs/TECHNICAL_DOCUMENTATION.md` first for the architecture and API
surface; this guide is task-oriented ("how do I...").

---

## Setting Up for Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python tests/generate_synthetic_fixture.py
python -m src.train_models
pytest tests/ -q
```

If all 265+ tests pass, your environment is correctly configured.

## Project Conventions

- **`src/` never imports Streamlit.** All business/ML logic is
  UI-framework-agnostic; `app/` is the only place `import streamlit`
  appears.
- **Every engine loads serialized artifacts; none trains inline in the
  dashboard.** If you find yourself calling `.fit()` on a supervised
  model inside `app/`, stop -- that belongs in `src/train_models.py`.
- **Business-policy constants live in `configurable_thresholds.py`
  (JSON-backed); engineering constants live in `config.py` (Python).**
  Don't add a new hardcoded threshold anywhere else.
- **Every new plotting function returns a `matplotlib.figure.Figure`
  and never calls `plt.show()`** -- the caller (a notebook cell or a
  Streamlit page) decides how to render it.
- **Logging:** always `from src import utils; logger = utils.get_logger(__name__)`
  -- never `print()` for anything beyond a notebook's own scratch cells.

## How to Add a New Dashboard Page

1. Create `app/app_pages/your_page.py`, following the structure of an
   existing page: `sys.path` bootstrap -> imports from `app.common` ->
   `apply_global_style()` -> `render_page_header(...)` -> engine calls ->
   `st.pyplot`/`st.dataframe`/`st.metric` display.
2. Register it in `app/app.py`'s `pages` dict via `st.Page(...)`.
3. Add a corresponding test to `tests/test_app.py`'s `PAGE_FILES` list
   (the parametrized `test_every_page_loads_without_exception` will pick
   it up automatically).
4. If the page needs an expensive, rerun-repeated computation, add a
   cached wrapper to `app/common.py` (follow the `get_global_explanation`/
   `get_cluster_visualization` pattern: `st.cache_data` with a
   leading-underscore parameter for the unhashable engine).

## How to Add a New Model

The project intentionally has exactly three supervised models
(Logistic Regression, Random Forest, XGBoost) -- Phase 6/7 explicitly
prohibit adding new ones. If a future phase legitimately calls for a
fourth model:

1. Add a `build_<name>_pipeline()` function to `src/model_utils.py`,
   following the existing three (preprocessing + classifier bundled in
   one `Pipeline`).
2. Add its hyperparameter search space to `src/config.py`.
3. Register it in `model_utils.PIPELINE_BUILDERS` and
   `model_utils.MODEL_DISPLAY_NAMES`.
4. `RiskScoringEngine`, `ExplainabilityEngine`, and every dashboard page
   that iterates over `MODEL_KEYS`/`MODEL_LABELS` will pick it up
   automatically without further code changes -- that generality was a
   deliberate design goal from Phase 3 onward.

## How to Run the Test Suite

```bash
pytest tests/ -q                                   # everything (265+ tests)
pytest tests/test_integration.py -v                 # cross-component seams only
pytest tests/test_edge_cases.py -v                  # missing/invalid input handling only
pytest tests/test_app.py -v                          # dashboard (headless, via AppTest)
python -m pyflakes src/ app/                         # static unused-import/dead-code check
```

Add new tests in the same style: real artifacts on disk for integration
tests, deliberately broken/extreme inputs for edge-case tests, `AppTest`
for anything dashboard-related.

## How to Regenerate a Notebook

Each notebook has a matching `tests/build_notebook_phaseN.py` script
that generates it programmatically (so the notebook can be regenerated
deterministically rather than hand-edited cell by cell):

```bash
python tests/build_notebook_phase3.py   # writes notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb
```

Re-execute it afterward (e.g. via `jupyter nbconvert --execute` or
`nbclient`) to populate real output cells before committing.

## Debugging Checklist

1. **A page shows a warning instead of data** -- the underlying artifact
   is missing; run the command the warning message prints.
2. **An engine raises `FileNotFoundError`** -- same cause; check
   `models/`, `pipelines/`, `reports/`, `reports/explainability/`,
   `reports/segmentation/` for the expected `.joblib` file.
3. **A test fails after a `src/` change** -- check
   `tests/test_integration.py` first; it's the fastest way to confirm
   whether you broke a cross-component contract (e.g. changed a
   `RiskScoringEngine` method's return shape that `ExplainabilityEngine`
   depends on).
4. **Slow dashboard interaction** -- check whether the operation is
   already wrapped by one of `app/common.py`'s `st.cache_data`/
   `st.cache_resource` functions; if not, and it's expensive, it
   probably should be (see `PERFORMANCE_REPORT.md` for what's already
   covered).

## Code Review Checklist (apply to every PR/commit)

- [ ] No `import streamlit` anywhere under `src/`.
- [ ] No hardcoded threshold/business constant outside `configurable_thresholds.py`.
- [ ] No hardcoded absolute file path outside `config.py`.
- [ ] Every new public function/class has a docstring explaining *why*,
      not just *what*.
- [ ] `pytest tests/ -q` and `python -m pyflakes src/ app/` both pass
      clean before committing.
