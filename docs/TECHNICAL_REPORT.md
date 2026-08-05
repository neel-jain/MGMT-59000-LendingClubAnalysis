# Technical Report

**LendingClub Loan Default Risk — Indiana Borrowers**
**MGMT 59000 Business Analytics Capstone, Purdue University System**

---

## Executive Summary

This report documents a seven-phase capstone project that builds a
complete lending-risk analytics platform: a validated data pipeline,
three tuned supervised classifiers, a SHAP-based explainability layer,
an unsupervised borrower-segmentation layer, and an eight-page Streamlit
executive dashboard — all backed by 265+ automated tests and
deployment-ready configuration. The production model (XGBoost) is
wrapped by a `RiskScoringEngine` that converts a raw prediction into a
risk tier, confidence score, and concrete lending recommendation; an
`ExplainabilityEngine` explains every prediction in plain business
language; and a `SegmentationEngine` identifies natural borrower groups
that complement (never replace) the individual-borrower prediction.

## Business Objective

LendingClub needs to price and approve loans in a way that balances
growth against portfolio risk, for Indiana borrowers specifically. The
objective was not merely to build "a model" but a decision-support
system: something a lending executive, underwriter, or portfolio
manager could open, use, question, and trust — with every number
traceable back to a specific, explainable cause.

## PDID Framework

Every phase of this project applies **Problem -> Data -> Insight ->
Decision**:
- **Problem** is stated before any analysis begins (e.g. Phase 3's
  problem: "which model best discriminates future defaulters, and at
  what cost tradeoff?").
- **Data** is the specific slice/transformation needed to address that
  problem (e.g. the leakage-safe train/val/test split).
- **Insight** is what the analysis actually reveals (e.g. XGBoost's
  ROC-AUC advantage, or DTI's consistent top-3 SHAP ranking).
- **Decision** converts the insight into something actionable (e.g. the
  cost-minimizing decision threshold, or a segment-specific rate
  adjustment).

## Methodology

### Data Preparation (Phase 1)

Raw data is ingested, schema-validated, and cleaned: duplicates
removed, percentage strings parsed to numeric, employment length parsed
to years, a binary `default_flag` target constructed (`Charged Off`/
`Default` = 1, `Fully Paid` = 0, all other statuses excluded to avoid
mislabeling unresolved loans), and leakage-prone/identifier columns
dropped. The cleaned dataset is split into stratified train/validation/
test sets **before** any preprocessing statistic is computed, and the
preprocessing `ColumnTransformer` (median imputation + standardization
for numeric features, one-hot/ordinal encoding for categorical features)
is embedded inside each model's `Pipeline` rather than fit once
externally — ensuring every cross-validation fold refits its own
preprocessing statistics and never leaks validation-fold information
into training.

### Exploratory Analysis (Phase 2)

Descriptive statistics, default-rate-by-group analysis, correlation
analysis, and formal statistical testing (Pearson/Spearman correlation,
chi-square, Welch's t-test, ANOVA, Wilson-score confidence intervals)
were run against all seven research questions, each paired with a
plain-language business interpretation rather than a bare p-value.

### Machine Learning Models (Phase 3)

Three models were trained and compared:

- **Logistic Regression** — GridSearchCV over a small, exhaustively
  searchable space (`C`, penalty, class weight); retained in production
  for its interpretability even though it is not the top-ranked model.
- **Random Forest** — RandomizedSearchCV over a larger space (tree
  count, depth, leaf size, feature sampling); a nonlinear cross-check
  against Logistic Regression.
- **XGBoost** — RandomizedSearchCV over the largest space (boosting
  triad, subsampling, regularization); selected as the production model
  by test ROC-AUC.

All three were evaluated with Stratified 5-fold cross-validation
(chosen to keep enough positive/minority-class examples per fold given
the dataset's ~20-25% default rate) optimizing ROC-AUC (chosen over
accuracy specifically because accuracy is misleading under class
imbalance), then scored on a held-out test set across the full metric
suite: accuracy, precision, recall, specificity, F1, ROC-AUC, balanced
accuracy, Matthews correlation coefficient, log loss, Brier score, and
Expected Calibration Error. A cost-minimizing decision threshold (not
the default 0.50) was derived per model from the relative business cost
of a false negative (a missed defaulter) vs. a false positive (a
wrongly-declined good borrower).

### Explainability (Phase 4A)

`ExplainabilityEngine` wraps the production model with SHAP —
`TreeExplainer` (exact) for the tree-based models, `LinearExplainer`
(exact) for Logistic Regression — computed in log-odds (margin) space
for speed and additivity. Global explanations (summary/dependence/
decision plots, feature-interaction analysis, partial dependence/ICE)
and local per-borrower explanations (waterfall/force plots, top risk/
protective factors) are both translated into plain-language business
summaries before reaching the dashboard. `RiskScoringEngine` sits
alongside it, converting probability into a risk tier, confidence score,
and lending recommendation via `configurable_thresholds.py`'s
JSON-backed, business-editable threshold configuration.

### Borrower Segmentation (Phase 4B)

`SegmentationEngine` clusters borrowers on a deliberately **narrower**
feature space (numeric financial characteristics + ordinal grade,
excluding one-hot categorical columns to avoid distance-metric
distortion) using K-Means by default, cross-checked against
Agglomerative Clustering and Gaussian Mixture Models. The optimal
cluster count combines silhouette score, Calinski-Harabasz, and
Davies-Bouldin rankings rather than trusting any single metric.
Segments are named from the data itself (relative income/DTI/rate/
default-rate patterns) and cross-referenced against the supervised
model's predicted probabilities — agreement between the two independent
analytical lenses is itself a validation signal.

### Dashboard Design (Phase 5)

An eight-page Streamlit application built with `st.navigation`/`st.Page`
routing. The dashboard enforces a strict architectural boundary: `app/`
contains zero modeling logic, only orchestration over the three engines
via `app/common.py`'s cached loaders. Every plot returns a
`matplotlib.figure.Figure`; every text summary and exportable report
comes directly from an engine method.

### Testing (Phase 6)

265+ automated tests across five categories: unit tests (per-module,
~204 tests across Phases 1-4B), Streamlit `AppTest`-based dashboard
tests (18 tests — every page, the full prediction form, sidebar
controls), integration tests (18 tests — every cross-component seam,
using real artifacts, not mocks), and edge-case tests (25 tests —
missing values, empty datasets, invalid grades, negative income,
extreme DTI/loan amounts, unexpected categories, corrupted/missing
files, malformed input). The edge-case suite caught and led to fixing a
real bug: `RiskScoringEngine.predict_probability()` previously crashed
on a zero-row input instead of returning gracefully.

### Performance Optimization (Phase 6)

A `pyflakes` audit removed 10+ unused imports, one dead variable, and
three confirmed-unused dependencies. Three previously-uncached expensive
per-page computations (t-SNE/UMAP projection, cross-validated learning
curves, global SHAP computation) were wrapped in `st.cache_data`,
eliminating multi-second recomputation on every dashboard rerun. Full
measured numbers (startup time, prediction latency, memory usage,
ranked bottlenecks) are in `PERFORMANCE_REPORT.md`.

## Results

- All three models were successfully trained, tuned, and evaluated; the
  production model (XGBoost) was selected by test ROC-AUC with the
  full evaluation-metric suite and threshold analysis documented in
  `notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb`.
- Global and local SHAP explanations were generated and cross-validated
  against independent permutation/native importance measures.
- Borrower segmentation identified data-driven segments whose predicted
  risk agrees directionally with the supervised model's own ranking.
- The dashboard's 8 pages, all sidebar controls, the full prediction
  workflow, and every exportable report were verified end-to-end via
  automated headless testing.

*(Note: the numbers actually produced by this development environment
reflect a small synthetic test fixture, not the genuine ~37,515-row
Indiana extract — the methodology above is unchanged either way; re-run
`python -m src.train_models` against the real data for real findings.)*

## Limitations

See `README.md`'s "Known Limitations" section and `QA_CHECKLIST.md` for
the complete list: no legally protected class attributes in the data
(limiting the fairness assessment's scope), Indiana-only geographic
scope, historical data that does not reflect subsequent economic
conditions, threshold configuration that requires periodic human
review rather than automated feedback, and no production-grade
authentication layer.

## Recommendations

1. Use segment membership alongside LendingClub grade for pricing and
   origination-limit decisions, not grade alone.
2. Route by risk tier: streamlined approval for Low Risk, mandatory
   manual review for High Risk — implemented via the already-configurable
   `RiskThresholdConfig`, not a code change.
3. Prioritize verifying interest rate, DTI, and grade at application
   time, since these consistently rank highest across every model and
   importance method evaluated.
4. Use the supervised model for individual lending decisions and
   segmentation for portfolio/marketing strategy — the two are
   complementary, not redundant.

## Future Work

See `README.md`'s "Future Enhancements" and `PERFORMANCE_REPORT.md`
Section 9: macroeconomic features, automated drift monitoring, ensemble
scoring across all three models, persisted optimal-k evaluation for
faster segmentation cold starts, and PNG chart export wired into every
dashboard page.
