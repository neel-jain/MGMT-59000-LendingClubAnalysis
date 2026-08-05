# GitHub Preparation

Recommendations for publishing this repository publicly.

## Repository Description

> A full-stack lending risk analytics platform for LendingClub Indiana
> borrowers — supervised ML (Logistic Regression, Random Forest,
> XGBoost), SHAP explainability, unsupervised borrower segmentation,
> and an 8-page Streamlit executive dashboard. 265+ automated tests.

## Suggested Repository Name

`lendingclub-indiana-risk-platform` or `mgmt590-lendingclub-capstone`
(the latter if keeping the course/academic framing explicit; the former
if optimizing for a professional portfolio audience unfamiliar with the
course code).

## Suggested Topics/Tags

`machine-learning` `streamlit` `shap` `explainable-ai` `xgboost`
`scikit-learn` `clustering` `credit-risk` `fintech` `python` `pandas`
`data-science` `capstone-project` `risk-scoring` `dashboard`

## Final Folder Organization Review

The current structure (documented in full in `README.md`'s "Folder
Structure" section and `docs/TECHNICAL_DOCUMENTATION.md`) is
recommended as-is for publication:

- `src/` — all business/ML logic (no Streamlit imports)
- `app/` — the dashboard (orchestration only)
- `notebooks/` — one notebook per phase
- `docs/` — Phase 7 documentation, report, and presentation
- `tests/` — the full automated test suite
- `data/`, `models/`, `pipelines/`, `reports/`, `logs/` — generated
  artifacts, correctly excluded from version control by `.gitignore`

One recommendation before publishing: **do not commit the real
LendingClub data file** (`data/raw/lendingclub_indiana_raw.csv`) if it
is not already public-domain-clear for redistribution — confirm
LendingClub's data license terms, or keep the raw data out of the
repository entirely and rely on the documented download/setup
instructions instead. The synthetic fixture generator
(`tests/generate_synthetic_fixture.py`) is safe to keep, since it
generates no real borrower data.

## .gitignore Review

The current `.gitignore` correctly excludes all generated artifacts
(`data/raw/*`, `data/processed/*`, `data/splits/*`, `models/*`,
`pipelines/*`, `reports/*` and its `explainability/`/`segmentation/`
subdirectories, `logs/*`) while preserving each directory's structure
via `.gitkeep` placeholders — this is the correct pattern for a
repository that should be cloneable and immediately runnable (via the
documented setup commands) without shipping potentially large or
sensitive generated files. No changes recommended.

## Release Version: v1.0

Recommended tag/release notes for the first public release:

```
v1.0 — Initial Public Release

Complete seven-phase capstone: data pipeline, EDA, three supervised
models (Logistic Regression, Random Forest, XGBoost), SHAP-based
explainability, unsupervised borrower segmentation, an 8-page Streamlit
dashboard, and 265+ automated tests (unit, integration, edge-case, and
dashboard-level). See CHANGELOG.md for the full phase-by-phase history
and docs/TECHNICAL_REPORT.md for the complete write-up.
```

## README Badges (optional polish)

If desired, add badges near the top of `README.md` for: Python version
(from `runtime.txt`), test count/status (if wiring up CI), and license
(once one is selected — see `README.md`'s License section).

## Suggested `.github/` additions (not created in this phase)

If continuing to maintain this repository, consider adding a GitHub
Actions workflow that runs `pytest tests/ -q` and
`python -m pyflakes src/ app/` on every push — the test suite is already
structured to support this with no changes.
