# Performance Report — Phase 6

**MGMT 590 LendingClub Loan Default Risk Capstone**
**Measured against:** synthetic test-fixture data (~2,000 rows; the
project's real ~37,515-row Indiana extract will shift absolute numbers
but not the relative bottleneck ranking below). All timings below are
single-run measurements taken on the development container (a modest,
single-process CPU environment, not the eventual deployment host) using
Python's `time.perf_counter()` and Linux's `resource.getrusage()`; treat
them as directional, not as a guaranteed SLA.

---

## 1. Executive Summary

The application's cost is heavily front-loaded: model loading and SHAP
background sampling (paid **once per server process**, thanks to
`st.cache_resource`) dominate cold-start time, while every per-request
operation a user actually waits on — a single prediction, a single SHAP
explanation, a single segment lookup — completes in well under 100ms.
The two genuinely slow operations users may notice are **t-SNE
visualization** (~6.4s) and **SHAP background/segmentation model
fitting at cold start** (~13-19s combined), both of which Phase 6 has
already cached so they are paid at most once per session, not on every
click.

## 2. Application Startup Time

| Stage | Time |
|---|---|
| `RiskScoringEngine()` cold load (load model + threshold config) | 1.95s |
| `ExplainabilityEngine()` cold load (load model + build SHAP background sample) | 3.68s |
| `SegmentationEngine().fit()` cold fit (clip outliers, evaluate optimal k across 7 candidates, fit K-Means, profile, name segments) | 12.53s |
| Full dashboard cold start (library imports + first page render, via `AppTest`) | 17.7s |

**Interpretation:** `SegmentationEngine.fit()` is the single largest
contributor to cold start, because `fit()` runs the full optimal-k
evaluation (`config.N_CLUSTERS_CANDIDATES`, 7 candidate values of k,
each requiring a full K-Means fit + three validity metrics) every time,
not just when `auto_select_k=True` is requested. This cost is paid
**once per server process** via `st.cache_resource` — not once per user
— so it matters for the first visitor to a freshly (re)started server,
not for steady-state usage.

## 3. Prediction Latency

| Operation | Latency |
|---|---|
| Single-borrower prediction (`RiskScoringEngine.generate_prediction_summary`) | 20.8ms |
| Batch prediction, 225 borrowers (`predict_probability`) | 14.5ms total (0.064ms/row) |
| Single-borrower SHAP explanation (`ExplainabilityEngine.explain_prediction`) | 40.5ms |
| Global SHAP explanation, 200-row sample (`explain_global_model`) | 222.0ms |
| Segment assignment, 10 borrowers (`assign_segment`, nearest-centroid) | 99.5ms |

**Interpretation:** every per-borrower operation a user directly waits
on after clicking "Predict Risk" completes in under 50ms — well within
what feels instantaneous in a web UI. The SHAP waterfall/force plots add
matplotlib rendering time on top of the 40.5ms SHAP computation itself
(not separately measured here, but bounded by typical single-figure
matplotlib render times of a few tens of milliseconds).

## 4. Model-Level Training Time / Prediction Throughput / Memory (from Phase 3)

Recorded directly in `reports/model_comparison_table.csv` (`training_time_sec` = one refit on the full training set; `prediction_time_ms_per_1000` = inference latency; `memory_usage_kb` = actual serialized `.joblib` file size):

| Model | Training Time | Prediction Time (ms/1,000 rows) | Serialized Size |
|---|---|---|---|
| XGBoost (production) | 0.19s | 48.8ms | 638 KB |
| Random Forest | 0.52s | 109.6ms | 1,708 KB |
| Logistic Regression | 0.02s | 35.2ms | 8 KB |

**Interpretation:** the production model (XGBoost) is a reasonable
middle ground — meaningfully smaller and faster to score than Random
Forest, at a small serialized-size cost over Logistic Regression in
exchange for materially better discrimination (see Phase 3's ROC-AUC
comparison). None of the three models pose a throughput concern at
this project's scale.

## 5. Clustering / Dimensionality-Reduction Performance

| Operation | Time |
|---|---|
| PCA visualization (2 components) | 380.7ms |
| t-SNE visualization (full training set) | 6.40s |

**Interpretation:** t-SNE is, as expected, the slowest visualization in
the application by a wide margin — it has no incremental/cached
`.transform()` and re-embeds the full input jointly every time it's
computed. Phase 6 addressed this by caching t-SNE's result via
`app.common.get_cluster_visualization` (an `st.cache_data`-wrapped
function keyed on the segmentation model's algorithm/cluster-count), so
it is computed once per distinct segmentation configuration and reused
across every subsequent page view or user, not recomputed per click.
`config.DIMENSIONALITY_REDUCTION_SAMPLE_SIZE` also caps the row count
passed to t-SNE/UMAP in `SegmentationEngine.visualize_clusters` for
exactly this reason.

## 6. Memory Usage

Peak process RSS after loading all three engines (`RiskScoringEngine`,
`ExplainabilityEngine`, and a fitted `SegmentationEngine`) in a single
Python process: **~517 MB**. This includes the Python interpreter and
every imported library (pandas, scikit-learn, XGBoost, SHAP, UMAP,
matplotlib) — the actual model/data artifacts are a small fraction of
this (see Section 4's serialized sizes; the full training split is a
few hundred KB as CSV). Memory is dominated by library import overhead,
not by this project's own data or model footprint, and is well within
the free-tier memory limits of every deployment platform discussed in
`DEPLOYMENT.md`.

## 7. Largest Bottlenecks (Ranked)

1. **`SegmentationEngine.fit()`'s optimal-k evaluation** (~12.5s) — the
   single largest cold-start contributor. Mitigated by `st.cache_resource`
   (paid once per process); a further optimization would be to persist
   the optimal-k evaluation table itself (already done — see
   `config.OPTIMAL_K_ANALYSIS_PATH`) and skip re-evaluating it at
   `fit()` time when a cached result already exists on disk with
   matching data — **deferred to a future phase** (see Section 9) since
   it would require a cache-invalidation strategy (detecting when the
   underlying training data has changed) that's out of scope for this
   capstone's fixed-dataset use case.
2. **t-SNE visualization** (~6.4s) — mitigated via `st.cache_data`
   (Section 5); PCA remains the default/fastest visualization.
2 (tie). **`ExplainabilityEngine`'s SHAP background sampling at cold
   start** (~2s of the 3.68s load) — mitigated by `st.cache_resource`;
   further reduction would mean shrinking `config.SHAP_BACKGROUND_SAMPLE_SIZE`,
   trading background-sample stability for speed.
3. **Global SHAP explanation over larger samples** (222ms for 200 rows,
   scaling roughly linearly) — mitigated via the new
   `get_global_explanation` cache wrapper (Phase 6); uncached, this
   would be recomputed on every dashboard/explainability page rerun.

## 8. Optimization Improvements Made in Phase 6

- Cached three previously-uncached, expensive per-page computations
  (t-SNE/UMAP visualization, cross-validated learning curves, global
  SHAP explanation) via `st.cache_data`, eliminating redundant
  multi-second recomputation on every widget interaction on the same page.
- Removed three unused dependencies (`imbalanced-learn`, `plotly`,
  `python-dateutil`) from the dependency graph, shrinking install time
  and deployment image size.
- Fixed a genuine correctness bug surfaced during edge-case testing:
  `RiskScoringEngine.predict_probability()` previously raised a low-level
  sklearn error on a zero-row input instead of returning an empty array
  gracefully — now handled explicitly, avoiding an unnecessary failed
  pipeline call.
- Removed 10+ unused imports and one dead variable flagged by a
  `pyflakes` audit, reducing import overhead and improving readability.

## 9. Recommendations for Future Scaling

- **Persist and reuse the optimal-k evaluation** across `SegmentationEngine.fit()`
  calls when the underlying training data is unchanged, rather than
  recomputing all 7 K-Means fits every cold start.
- **Move to a background/async task queue** (e.g. for SHAP computation
  on very large borrower batches) if the real ~37,515-row dataset proves
  materially slower than this synthetic-fixture benchmark once real data
  is loaded — current numbers suggest headroom, but should be
  re-measured against the genuine data volume before assuming linear
  scaling holds.
- **Consider a lighter-weight UMAP-only visualization path** (skip
  t-SNE) if t-SNE's lack of an incremental `.transform()` becomes a
  recurring pain point as the segmentation model is retrained more often.
- **Add a lightweight model-serving cache warm-up script** that
  pre-populates `st.cache_resource`/`st.cache_data` entries at deployment
  time (e.g. a startup hook that calls each engine once) so the FIRST
  real user, not just subsequent ones, gets a fast response.
