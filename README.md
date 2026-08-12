# LendingClub Loan Default Risk — Indiana Borrowers

**A full-stack analytics platform for lending risk: predictive modeling, explainable AI, borrower segmentation, and an executive Streamlit dashboard.**

[**▶ Launch the live Streamlit dashboard**](https://mgmt-59000-lendingclubanalysis.streamlit.app/)

Graduate Business Analytics capstone — MGMT 59000, Purdue University
System, Summer 2026. Predicts loan default risk for Indiana LendingClub
borrowers and turns that prediction into an explainable, actionable
lending decision through four reusable engines and an eight-page
executive dashboard.

> **Project status:** all seven phases complete. 265+ automated tests
> passing. See `CHANGELOG.md` for the detailed phase-by-phase build
> history.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Research Questions](#research-questions)
4. [Dataset Description](#dataset-description)
5. [Technology Stack](#technology-stack)
6. [Project Architecture](#project-architecture)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [Explainability Engine](#explainability-engine)
9. [Risk Scoring Engine](#risk-scoring-engine)
10. [Segmentation Engine](#segmentation-engine)
11. [Dashboard Overview](#dashboard-overview)
12. [Installation Instructions](#installation-instructions)
13. [Running the Project](#running-the-project)
14. [Folder Structure](#folder-structure)
15. [Screenshots](#screenshots)
16. [Known Limitations](#known-limitations)
17. [Future Enhancements](#future-enhancements)
18. [Documentation Index](#documentation-index)
19. [Acknowledgments](#acknowledgments)
20. [License](#license)

---

## Project Overview

This project answers a simple-sounding question with a genuinely
multi-layered system: **should LendingClub lend to this borrower, and
why?** Rather than stopping at a single predictive model, the project
builds four reusable, independently-tested engines and wires them into
a professional Streamlit dashboard:

- **`RiskScoringEngine`** — turns a borrower's application into a
  probability of default, a risk tier, a confidence score, and a
  concrete lending recommendation (approve / manual review / decline,
  suggested rate, model-driven grade).
- **`ExplainabilityEngine`** — answers *why* using SHAP, at both the
  population level (which variables matter most overall) and the
  individual-borrower level (why THIS borrower scored the way they did).
- **`SegmentationEngine`** — identifies natural borrower groups via
  unsupervised clustering, complementing the supervised risk score with
  a portfolio-level view.
- **A Streamlit dashboard** — an eight-page executive application that
  orchestrates the three engines above with zero embedded modeling
  logic of its own.

The result reads less like a class assignment and more like the
explainability/risk layer of a production lending platform — which was
the explicit goal from Phase 4A onward.

## Business Problem

LendingClub, a peer-to-peer lending platform, must assess borrower
default risk accurately enough to price loans fairly, protect portfolio
quality, and treat applicants consistently. Getting this wrong in either
direction is costly: under-pricing risk increases losses; over-declining
good borrowers loses revenue and trust. This project scopes that broad
problem to Indiana borrowers specifically and builds a decision-support
tool — not a black box — for lending executives, underwriters, and
portfolio managers.

## Research Questions

1. Which borrower characteristics appear associated with default?
2. Do LendingClub grades appear predictive of default?
3. Which variables are related to higher interest rates?
4. Does income relate to repayment success?
5. Does DTI (debt-to-income ratio) influence default?
6. Does employment length matter?
7. Which borrower segments represent the highest lending risk, and can
   natural groups be observed before clustering?

Every phase of this project — EDA, the three supervised models, SHAP
explanations, and segmentation — traces back to one or more of these
questions; see the dashboard's **Business Insights** page for how each
question is answered with evidence, a visualization, and a business
recommendation.

## Dataset Description

- **Source:** LendingClub's historical loan-performance data, filtered
  to Indiana (`addr_state == "IN"`) borrowers.
- **Scope:** ~37,515 resolved loans (`Fully Paid`, `Charged Off`, or
  `Default` — loans still `Current`, `Late`, or `In Grace Period` are
  excluded to avoid mislabeling unresolved outcomes).
- **Target:** `default_flag` — binary, 1 for `Charged Off`/`Default`,
  0 for `Fully Paid`.
- **Features (22 raw columns):** loan amount, term, interest rate,
  installment, grade, home ownership, annual income, verification
  status, purpose, DTI, delinquencies, open/total credit accounts,
  public records, revolving balance/utilization, mortgage accounts,
  bankruptcies, employment length, initial listing status, application
  type.
- **A note on the data in this repository:** a synthetic test fixture
  (`tests/generate_synthetic_fixture.py`) is provided so the full
  pipeline can be verified end-to-end without the genuine LendingClub
  extract. **Replace it with the real data** (`data/raw/lendingclub_indiana_raw.csv`)
  before treating any number in this repository's generated reports,
  notebooks, or dashboard as a real finding.

## Technology Stack

| Layer | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn, XGBoost |
| Statistics | scipy, statsmodels |
| Explainability | SHAP |
| Segmentation | scikit-learn (K-Means, Agglomerative, GMM), UMAP |
| Visualization | matplotlib, seaborn |
| Application | Streamlit (`st.navigation`/`st.Page` multipage) |
| Serialization | joblib |
| Testing | pytest, Streamlit's `AppTest` |
| Environment | Python 3.12 |

## Project Architecture

```
Raw Data -> Preprocessing Pipeline -> Trained Models (LR / RF / XGBoost)
                                          |
                     +--------------------+--------------------+
                     v                    v                    v
            RiskScoringEngine   ExplainabilityEngine   SegmentationEngine
                     |                    |                    |
                     +--------------------+--------------------+
                                          v
                              Streamlit Dashboard (8 pages)
                       (orchestration only -- no ML logic in app/)
```

The dividing line between `src/` (all business/ML logic) and `app/`
(pure orchestration) is a deliberate, enforced architectural boundary —
every Streamlit page calls into one of the three engines and displays
whatever comes back; none re-implements scoring, explanation, or
clustering logic. See `docs/TECHNICAL_DOCUMENTATION.md` for the full
architecture diagram and module-by-module description.

## Machine Learning Pipeline

Three supervised models are trained, tuned, and compared:

| Model | Role | Why |
|---|---|---|
| **Logistic Regression** | Interpretable baseline | Coefficients convert directly to odds ratios a credit committee can defend |
| **Random Forest** | Nonlinear workhorse | Captures feature interactions with minimal tuning |
| **XGBoost** | Production model | Strongest raw discrimination via sequential boosting |

Pipeline: ingest -> validate -> clean -> leakage-safe stratified
train/validation/test split -> preprocessing (median imputation,
standardization, one-hot/ordinal encoding) embedded inside a
scikit-learn `Pipeline` alongside each classifier -> hyperparameter
tuning (`GridSearchCV` for Logistic Regression, `RandomizedSearchCV` for
Random Forest/XGBoost) with Stratified 5-fold cross-validation,
optimizing ROC-AUC -> full evaluation-metric suite on a held-out test
set -> the best model by test ROC-AUC becomes `config.PRODUCTION_MODEL_KEY`.

Full methodology, metric definitions, and business rationale for every
choice live in `notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb` and
`src/model_utils.py`.

## Explainability Engine

`src/explainability.py`'s `ExplainabilityEngine` wraps the production
model with SHAP (`TreeExplainer` for tree-based models, `LinearExplainer`
for Logistic Regression) and exposes:

- **Global explanations:** `explain_global_model()`, `generate_shap_summary()`
  (beeswarm/bar), `generate_dependence_plot()`, `generate_decision_plot()`,
  feature-interaction analysis, partial dependence/ICE plots.
- **Local (per-borrower) explanations:** `explain_prediction()`,
  `generate_waterfall_plot()`, `generate_force_plot()`, top risk/
  protective factors, and a plain-language business summary.
- **Exports:** `export_borrower_explanation_report()` and
  `export_global_explanation_report()`, both Markdown/JSON-ready.

## Risk Scoring Engine

`src/risk_scoring.py`'s `RiskScoringEngine` turns a probability into a
business decision:

- `predict_probability()` / `predict()` — raw and thresholded predictions.
- `generate_prediction_summary()` — probability, 0-100 risk score, risk
  tier, confidence score, recommended action, suggested interest rate,
  model-driven loan grade — all in one call.
- Every threshold, tier boundary, and rate adjustment comes from
  `src/configurable_thresholds.py`'s JSON-backed `RiskThresholdConfig` —
  editable by a business stakeholder without touching code.

## Segmentation Engine

`src/segmentation_engine.py`'s `SegmentationEngine` answers a question
supervised learning can't: *what natural borrower groups exist?*

- `fit()` on the training population; K-Means by default, with
  Agglomerative Clustering and Gaussian Mixture Models available and
  compared.
- Optimal cluster count chosen by combining silhouette score,
  Calinski-Harabasz, and Davies-Bouldin rankings — not a single metric.
- Segment names are **assigned from the data** (relative income/DTI/
  rate/default-rate patterns), never fixed in advance.
- `compare_with_supervised_models()` cross-references each segment
  against `RiskScoringEngine`'s predicted probability — validating that
  the two independent analytical lenses agree.

## Dashboard Overview

An eight-page Streamlit application (`app/app.py` + `app/app_pages/`):

| # | Page | Purpose |
|---|---|---|
| 1 | Executive Dashboard | Portfolio KPIs, grade distribution, top risk factors |
| 2 | Exploratory Analysis | Interactive, filterable EDA visualizations |
| 3 | Model Comparison | ROC/PR/confusion/calibration/learning curves, ranking |
| 4 | Borrower Risk Prediction | Live single-borrower scoring + SHAP explanation |
| 5 | Borrower Segmentation | Segment lookup, PCA/t-SNE, recommendations |
| 6 | Business Insights | All 7 research questions as an executive report |
| 7 | Model Explainability | Global SHAP summary/dependence/decision plots |
| 8 | About Project | Methodology, PDID framework, limitations |

Every page is orchestration-only; every caching, error-handling, and
styling concern is centralized in `app/common.py`.

## Installation Instructions

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # full dev environment
# or: pip install -r requirements-app.txt   # lean, deployment-only
```

Place the real Indiana LendingClub extract at
`data/raw/lendingclub_indiana_raw.csv`, or generate a synthetic
stand-in to verify the pipeline runs:

```bash
python tests/generate_synthetic_fixture.py
```

## Running the Project

**1. Run the data + modeling pipeline (Phases 1 + 3):**

```bash
python -m src.train_models
```

This now applies IQR-based winsorization to `annual_inc` and `dti` before splitting and training.

**2. Persist the explainability and segmentation artifacts (Phases 4A/4B):**

```bash
python -c "
from src.explainability import ExplainabilityEngine
ExplainabilityEngine().persist_explainability_artifacts()
"
python -c "
from src import utils
from src.segmentation_engine import SegmentationEngine
X_train, _, _, y_train, _, _ = utils.load_splits()
engine = SegmentationEngine()
engine.fit(X_train, default_flags=y_train)
engine.persist_segmentation_artifacts()
"
```

**3. Launch the dashboard (from the project root):**

```bash
streamlit run app/app.py
```

The dashboard now uses the winsorized borrower dataset by default, with no non-winsorized option.

**4. Run the test suite:**

```bash
pytest tests/ -q          # full suite (265+ tests)
pytest tests/test_app.py tests/test_integration.py tests/test_edge_cases.py -v   # Phase 5/6 suites only
```

See `DEPLOYMENT.md` for environment variables, folder verification, and
platform-specific deployment guidance (Streamlit Community Cloud,
Render, Railway, Hugging Face Spaces).

## Folder Structure

```
mgmt590_capstone/
├── README.md, CHANGELOG.md
├── requirements.txt, requirements-app.txt, runtime.txt
├── PERFORMANCE_REPORT.md, QA_CHECKLIST.md, DEPLOYMENT.md
├── docs/                     # Phase 7: technical docs, guides, report, presentation
├── .streamlit/config.toml    # dashboard theme
├── src/                      # all business/ML logic (no Streamlit imports)
│   ├── config.py, utils.py, eda_utils.py, model_utils.py, train_models.py
│   ├── configurable_thresholds.py, interpretation_utils.py
│   ├── risk_scoring.py, explainability.py
│   └── cluster_analysis.py, cluster_visualization.py, segment_profiles.py, segmentation_engine.py
├── app/                      # Phase 5: Streamlit dashboard (orchestration only)
│   ├── app.py, common.py
│   └── app_pages/            # 8 pages
├── notebooks/                 # one notebook per phase (1, 2, 3, 4A, 4B)
├── data/{raw,processed,splits}/
├── models/, pipelines/, reports/{explainability,segmentation}/, logs/
└── tests/                     # 20 test files, 265+ tests
```

Full annotated tree (every file with its one-line purpose) is in
`docs/TECHNICAL_DOCUMENTATION.md`.

## Screenshots

*Screenshots of the running dashboard are not yet embedded in this
repository — add them here before final submission/portfolio use.*
Suggested set (see `docs/DEMO_SCRIPT.md` for the exact walkthrough to
capture them from):

- `docs/screenshots/01_executive_dashboard.png`
- `docs/screenshots/02_borrower_risk_prediction.png`
- `docs/screenshots/03_shap_waterfall.png`
- `docs/screenshots/04_borrower_segmentation.png`
- `docs/screenshots/05_model_comparison.png`

## Known Limitations

- No legally protected class attributes exist in the data — the
  fairness assessment can only speak to business/financial attribute
  parity (home ownership, loan purpose, grade), not protected classes.
- Indiana-only scope — findings may not generalize elsewhere without
  re-validation.
- Historical data reflects past origination patterns, not subsequent
  economic conditions.
- Thresholds are configurable but require periodic human review, not an
  automated feedback loop.
- No production-grade authentication — current deployment guidance
  targets demo/portfolio use, not a public production service.

Full details: `QA_CHECKLIST.md`'s "Known Limitations" section and
`docs/FINAL_QUALITY_REVIEW.md`.

## Future Enhancements

- Incorporate macroeconomic features (unemployment rate, rate
  environment at origination).
- Automated model-drift monitoring and retraining cadence.
- Ensemble the three supervised models rather than selecting one
  production scorer.
- Persist `SegmentationEngine`'s optimal-k evaluation across cold starts.
- PNG chart export wired into every dashboard page (helper already
  exists in `app/common.py`).

## Documentation Index

| Document | Contents |
|---|---|
| `CHANGELOG.md` | Detailed phase-by-phase build history |
| `docs/TECHNICAL_DOCUMENTATION.md` | Architecture, modules, class/API docs, configuration guide |
| `docs/USER_GUIDE.md` | Non-technical guide to using the dashboard |
| `docs/DEVELOPER_GUIDE.md` | Guide for extending/maintaining the codebase |
| `docs/TECHNICAL_REPORT.md` | Full capstone technical report |
| `docs/MGMT590_Capstone_Presentation.pptx` | 15-slide presentation with speaker notes |
| `docs/PRESENTATION_SCRIPT.md` | Script, timing, and consolidated speaker notes |
| `docs/DEMO_SCRIPT.md` | Click-by-click live demonstration guide |
| `docs/FACULTY_QA.md` | Anticipated faculty questions with prepared answers |
| `docs/SUBMISSION_CHECKLIST.md` | Comprehensive submission checklist |
| `docs/GITHUB_PREPARATION.md` | Repository description, topics, release notes |
| `docs/PORTFOLIO_MATERIALS.md` | Resume/LinkedIn/portfolio-site descriptions |
| `docs/FINAL_QUALITY_REVIEW.md` | Four-perspective final review + recommendations |
| `PERFORMANCE_REPORT.md` | Measured startup/latency/memory + bottlenecks |
| `QA_CHECKLIST.md` | Test-referenced QA checklist |
| `DEPLOYMENT.md` | Environment setup + platform deployment guidance |

## Acknowledgments

- **Data:** LendingClub's publicly released historical loan-performance
  data.
- **Course:** MGMT 59000, Purdue University System.
- **Libraries:** the maintainers of pandas, scikit-learn, XGBoost, SHAP,
  UMAP, and Streamlit, whose open-source work makes a project like this
  possible for a single-developer capstone.

## License

No license has been formally selected for this repository. If
publishing publicly, an MIT License is a reasonable default for a
portfolio/academic project (permissive, widely recognized, and
compatible with resume/portfolio use) — see `docs/GITHUB_PREPARATION.md`
for the specific recommendation and rationale.
