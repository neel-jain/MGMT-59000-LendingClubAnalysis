# Submission Checklist

**MGMT 590 LendingClub Loan Default Risk Capstone — Final Submission**

---

## ✔ Code

- [x] All `src/` modules present, PEP 8-formatted, type-hinted, docstringed
- [x] `python -m pyflakes src/ app/` reports zero unused imports/dead code
- [x] No hardcoded business thresholds outside `configurable_thresholds.py`
- [x] No hardcoded absolute file paths outside `config.py`
- [x] `app/` contains zero machine-learning logic (orchestration only)

## ✔ Notebooks

- [x] `MGMT590_LendingClub_Analysis.ipynb` (Phase 1) — executed, zero errors
- [x] `MGMT590_LendingClub_EDA_Phase2.ipynb` (Phase 2) — executed, zero errors
- [x] `MGMT590_LendingClub_Modeling_Phase3.ipynb` (Phase 3) — executed, zero errors
- [x] `MGMT590_LendingClub_Explainability_Phase4A.ipynb` (Phase 4A) — executed, zero errors
- [x] `MGMT590_LendingClub_Segmentation_Phase4B.ipynb` (Phase 4B) — executed, zero errors

## ✔ Dashboard

- [x] All 8 pages present and load without exception
- [x] Navigation works between all pages
- [x] Borrower Risk Prediction form submits end-to-end (score + SHAP)
- [x] Borrower Segmentation page shows segment lookup + visualizations
- [x] Downloads (CSV, Markdown, JSON) work from at least one page each
- [x] Sidebar controls (model selector, filters, theme, downloads, about) all render

## ✔ Documentation

- [x] `README.md` — full outline (overview through license)
- [x] `CHANGELOG.md` — phase-by-phase build history
- [x] `docs/TECHNICAL_DOCUMENTATION.md` — architecture, modules, API, config
- [x] `docs/USER_GUIDE.md`
- [x] `docs/DEVELOPER_GUIDE.md`
- [x] `docs/TECHNICAL_REPORT.md`
- [x] `DEPLOYMENT.md`, `PERFORMANCE_REPORT.md`, `QA_CHECKLIST.md` (Phase 6, referenced not duplicated)

## ✔ Serialized Models

- [x] `models/logistic_regression_model.joblib`
- [x] `models/random_forest_model.joblib`
- [x] `models/xgboost_model.joblib`
- [x] `models/clustering_model.joblib`
- [x] `pipelines/preprocessing_pipeline.joblib`

> **Before final submission:** regenerate all of the above against the
> real ~37,515-row Indiana extract (`python -m src.train_models` +
> the explainability/segmentation persistence commands in `README.md`),
> not the synthetic fixture used during development.

## ✔ Testing

- [x] 265+ automated tests, all passing (`pytest tests/ -q`)
- [x] Unit tests (Phases 1-4B), integration tests, edge-case tests,
      dashboard (`AppTest`) tests all present

## ✔ Presentation

- [x] `docs/MGMT590_Capstone_Presentation.pptx` — 15 slides, speaker notes, validated
- [x] `docs/PRESENTATION_SCRIPT.md` — timing + consolidated script
- [x] `docs/DEMO_SCRIPT.md` — live demo walkthrough
- [x] `docs/FACULTY_QA.md` — anticipated questions + answers

## ✔ Report

- [x] `docs/TECHNICAL_REPORT.md` — Executive Summary through Future Work

## ✔ Requirements

- [x] `requirements.txt` (full dev environment)
- [x] `requirements-app.txt` (lean deployment-only)
- [x] `runtime.txt` (Python version pin)
- [x] No unused dependencies (verified via codebase-wide import grep)

## ✔ Deployment Guide

- [x] `DEPLOYMENT.md` — environment setup, launch instructions, folder/
      dependency verification, 4 platform-specific guides

## ✔ Screenshots

- [ ] **Action required before final submission:** capture screenshots
      per `README.md`'s suggested set and `docs/DEMO_SCRIPT.md`'s
      walkthrough, save under `docs/screenshots/`, and embed them in
      `README.md`'s Screenshots section.

## Final Pre-Submission Steps

1. Replace the synthetic test fixture with the real Indiana LendingClub
   extract and re-run the full pipeline (`README.md`'s "Running the
   Project" section).
2. Re-run `pytest tests/ -q` against the real data to confirm nothing
   regressed.
3. Capture the screenshots noted above.
4. Rehearse the presentation once using `docs/PRESENTATION_SCRIPT.md`
   and `docs/DEMO_SCRIPT.md` together, live.
5. Zip or push the final repository per `docs/GITHUB_PREPARATION.md`.
