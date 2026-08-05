# Technical Documentation

**MGMT 590 LendingClub Loan Default Risk Capstone**

Covers: Architecture Overview, Module Descriptions, API/Class
Documentation, and the Configuration Guide. For deployment steps see
`DEPLOYMENT.md`; for day-to-day usage see `docs/USER_GUIDE.md` and
`docs/DEVELOPER_GUIDE.md`.

---

## 1. Architecture Overview

```
data/raw/*.csv
      |
      v
[Phase 1] src/utils.py, src/config.py
  ingest -> validate -> clean -> leakage-safe split -> preprocessing Pipeline
      |
      v
[Phase 3] src/model_utils.py, src/train_models.py
  3 tuned models (LR / RF / XGBoost), each preprocessing+classifier bundled
  in one sklearn Pipeline -> serialized to models/*.joblib
      |
      +-----------------------+------------------------+
      v                       v                        v
[Phase 4A]              [Phase 4A]                [Phase 4B]
src/risk_scoring.py      src/explainability.py     src/segmentation_engine.py
RiskScoringEngine        ExplainabilityEngine       SegmentationEngine
(+ configurable_          (SHAP)                    (+ cluster_analysis.py,
 thresholds.py,                                       cluster_visualization.py,
 interpretation_utils.py)                             segment_profiles.py)
      |                       |                        |
      +-----------------------+------------------------+
                              v
                    [Phase 5] app/app.py + app/app_pages/*.py
                    Streamlit dashboard -- orchestration ONLY
```

**The one architectural rule that matters most:** `app/` never contains
modeling logic. Every page imports from `src/` (via `app/common.py`'s
cached loaders) and displays whatever an engine method returns. This is
enforced by convention (see `docs/DEVELOPER_GUIDE.md`'s code-review
checklist) rather than by a language-level boundary, so it is worth
re-verifying on every new page.

## 2. Module Descriptions

| Module | Phase | Responsibility |
|---|---|---|
| `src/config.py` | 1 | All paths, constants, column groups, CV/search/business settings. Single source of truth every other module imports from. Supports `MGMT590_PROJECT_ROOT` and `MGMT590_LOG_LEVEL` environment variable overrides. |
| `src/utils.py` | 1 | Data ingestion, validation, cleaning, the shared preprocessing `ColumnTransformer` builder, leakage-safe train/val/test split, generic joblib serialization helpers, the shared logger factory. |
| `src/eda_utils.py` | 2 | Descriptive statistics, the shared plotting style constants (`FIGSIZE_STANDARD`, `COLOR_DEFAULT`/`COLOR_PAID`, `_apply_titles`), and every EDA chart function. |
| `src/model_utils.py` | 3 | Model `Pipeline` builders (one per algorithm), hyperparameter search wrappers, the full evaluation-metric suite, feature-importance extraction, and every model-diagnostic chart function. |
| `src/train_models.py` | 1, 3 | Orchestration entry point: `run_phase1_pipeline()` and `run_phase3_pipeline()`. The only module meant to be run as a script (`python -m src.train_models`). |
| `src/configurable_thresholds.py` | 4A | `RiskThresholdConfig` -- the JSON-backed, hot-editable risk-tier/action/rate/grade business rules. |
| `src/interpretation_utils.py` | 4A | Feature-name humanization, research-question linkage, business-summary text generation, fairness reporting, and the shared `ExportableReport` class. |
| `src/risk_scoring.py` | 4A | `RiskScoringEngine` (see Section 3) and the expanded threshold-optimization functions. |
| `src/explainability.py` | 4A | `ExplainabilityEngine` (see Section 3). |
| `src/cluster_analysis.py` | 4B | Clustering-specific preprocessing, dimensionality reduction (PCA/t-SNE/UMAP), the four clustering algorithms, and optimal-k evaluation. |
| `src/cluster_visualization.py` | 4B | Every cluster-related chart function (scatter, heatmap, parallel coordinates, radar, size distribution, feature-by-cluster). |
| `src/segment_profiles.py` | 4B | Per-cluster profiling, data-driven segment naming, business recommendations, segment comparison tables. |
| `src/segmentation_engine.py` | 4B | `SegmentationEngine` (see Section 3). |
| `app/app.py` | 5 | Dashboard entry point: `st.navigation` routing + shared sidebar. No ML logic. |
| `app/common.py` | 5, 6 | Every cached loader (`st.cache_resource`/`st.cache_data`), sidebar control, styling helper, and download helper shared across all 8 pages. |
| `app/app_pages/*.py` | 5 | The 8 dashboard pages -- pure orchestration, one file per page. |

## 3. Class / API Documentation

Full docstrings live in each module; this section is the quick-reference
surface a new developer or grader needs first.

### `RiskScoringEngine` (`src/risk_scoring.py`)

```python
RiskScoringEngine(
    model_key: str = config.PRODUCTION_MODEL_KEY,
    threshold_config: RiskThresholdConfig | None = None,
    pipeline: Pipeline | None = None,
    base_interest_rate: float = 10.0,
)
```

| Method | Returns | Purpose |
|---|---|---|
| `predict_probability(X)` | `np.ndarray` | Default probability per row |
| `predict(X, threshold=None)` | `np.ndarray` | Binary prediction at the cost-minimizing (or given) threshold |
| `calculate_risk_score(p)` | `float` | 0-100 risk score |
| `calculate_confidence_score(p)` | `float` | 0-100 confidence (distance from 0.5) |
| `assign_risk_tier(p)` | `str` | Tier name via `RiskThresholdConfig` |
| `recommend_lending_action(tier)` | `str` | Approve / Manual Review / Decline / ... |
| `recommend_interest_rate(tier, base_rate=None)` | `float` | Tier-adjusted rate |
| `recommend_loan_grade(p)` | `str` | Model-driven A-G grade |
| `generate_prediction_summary(borrower)` | `PredictionSummary` | Everything above, one call, one borrower |
| `generate_batch_summary(X)` | `pd.DataFrame` | Vectorized equivalent for many borrowers |
| `export_prediction_report(borrower)` | `ExportableReport` | Markdown/JSON-ready risk assessment |

### `ExplainabilityEngine` (`src/explainability.py`)

```python
ExplainabilityEngine(
    model_key: str = config.PRODUCTION_MODEL_KEY,
    pipeline: Pipeline | None = None,
    background_data: pd.DataFrame | None = None,
    risk_scoring_engine: RiskScoringEngine | None = None,
)
```

| Method | Returns | Purpose |
|---|---|---|
| `explain_prediction(borrower)` | `LocalExplanation` | SHAP values, top factors, business summary for one borrower |
| `explain_global_model(X=None)` | `GlobalExplanation` | Importance ranking + top/least influential + business summary |
| `summarize_feature_importance(X=None)` | `pd.DataFrame` | SHAP vs. permutation vs. native importance, side by side |
| `generate_shap_summary(X, plot_type)` | `Figure` | Global SHAP summary plot ("beeswarm" or "bar") |
| `generate_waterfall_plot(borrower)` | `Figure` | Local SHAP waterfall |
| `generate_force_plot(borrower)` | `Figure` | Local SHAP force plot |
| `generate_dependence_plot(feature, X, interaction_feature)` | `Figure` | SHAP dependence plot |
| `generate_decision_plot(X, n_samples=20)` | `Figure` | Multi-borrower decision plot |
| `analyze_feature_interactions(X, pairs)` | `dict` | Interaction analysis per pair |
| `generate_pdp_ice_plot(X, features, kind="both")` | `Figure` | Partial dependence / ICE |
| `generate_business_summary(borrower=None)` | `str` | Local or global narrative, dispatched on argument |
| `persist_explainability_artifacts(X=None, y=None)` | `None` | Serializes every Phase 4A artifact |

### `SegmentationEngine` (`src/segmentation_engine.py`)

```python
SegmentationEngine(
    n_clusters: int = config.DEFAULT_N_CLUSTERS,
    algorithm: str = config.DEFAULT_CLUSTERING_ALGORITHM,
    risk_scoring_engine: RiskScoringEngine | None = None,
    threshold_config: RiskThresholdConfig | None = None,
)
```

| Method | Returns | Purpose |
|---|---|---|
| `fit(X, default_flags=None, auto_select_k=False)` | `self` | Fit the full segmentation workflow |
| `predict_cluster(X)` | `np.ndarray` | Cluster id (raises `NotImplementedError` for algorithms without `.predict()`) |
| `assign_segment(X)` | `pd.Series` | Business segment NAME per row, via nearest-centroid (works for every algorithm) |
| `describe_segment(cluster_id)` | `str` | Executive paragraph for one segment |
| `generate_cluster_profile()` | `pd.DataFrame` | Full per-cluster profile table |
| `compare_segments()` | `pd.DataFrame` | Executive segment-comparison table |
| `recommend_business_actions(cluster_id=None)` | `SegmentRecommendation` or `dict` | Lending/rate/marketing/portfolio recommendations |
| `visualize_clusters(method)` | `Figure` | 2D cluster scatter ("pca", "tsne", or "umap") |
| `compare_with_supervised_models(X=None)` | `pd.DataFrame` | Cross-check against `RiskScoringEngine` |
| `export_segment_summary()` | `ExportableReport` | Full exportable segmentation report |
| `persist_segmentation_artifacts()` | `None` | Serializes every Phase 4B artifact |

## 4. Configuration Guide

All configuration lives in two places, by design (see
`src/configurable_thresholds.py`'s module docstring for the full
rationale):

| File | Contains | Who edits it |
|---|---|---|
| `src/config.py` | Paths, CV settings, hyperparameter search spaces, engineering constants | Developers, via a code change |
| `reports/risk_threshold_config.json` | Risk tiers, lending actions, interest-rate adjustments, loan-grade bands | Business/credit-policy stakeholders, via a text editor -- **no code change or redeploy required** |

`reports/risk_threshold_config.json` self-bootstraps with sensible
defaults the first time `RiskScoringEngine()`/`load_threshold_config()`
runs if it doesn't already exist. Call `RiskThresholdConfig.validate()`
after any manual edit (or simply reconstruct the engine -- validation
runs automatically on load) to catch a gap/overlap in tier boundaries
before it silently mis-scores borrowers.

**Environment variables** (see `DEPLOYMENT.md` Section 2 for deployment
context):

| Variable | Effect |
|---|---|
| `MGMT590_PROJECT_ROOT` | Override where `config.PROJECT_ROOT` resolves to |
| `MGMT590_LOG_LEVEL` | Override `config.LOG_LEVEL` (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

## 5. Notebooks

One notebook per phase, each a thin, readable wrapper around the
corresponding `src/` module(s) -- the real logic lives in the modules,
not the notebooks, so nothing here duplicates Section 2/3's content:

| Notebook | Phase |
|---|---|
| `MGMT590_LendingClub_Analysis.ipynb` | 1 -- data pipeline walkthrough |
| `MGMT590_LendingClub_EDA_Phase2.ipynb` | 2 -- full EDA + statistical testing |
| `MGMT590_LendingClub_Modeling_Phase3.ipynb` | 3 -- model training/tuning/comparison |
| `MGMT590_LendingClub_Explainability_Phase4A.ipynb` | 4A -- SHAP + risk scoring demonstration |
| `MGMT590_LendingClub_Segmentation_Phase4B.ipynb` | 4B -- clustering demonstration |
