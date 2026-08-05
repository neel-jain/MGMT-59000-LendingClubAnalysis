# Quality Assurance Checklist — Phase 6

**MGMT 590 LendingClub Loan Default Risk Capstone**
Status as of Phase 6 completion. Every item below was verified either
by an automated test (file:test referenced) or by manual inspection
noted inline. Run `pytest tests/ -v` to re-verify the automated items
yourself; **265/265 tests pass** as of this phase.

---

## ✔ Data Loading

- [x] Raw CSV loads with schema/row-count validation — `tests/test_utils.py`
- [x] Cleaned dataset persists and reloads correctly — `tests/test_integration.py::test_data_loading_produces_consistent_schema`
- [x] Train/validation/test splits are leakage-safe and schema-consistent — `tests/test_integration.py::test_data_loading_produces_consistent_schema`
- [x] Missing raw data file produces a clear `FileNotFoundError`, not a crash — `tests/test_edge_cases.py::test_load_dataframe_missing_file_raises_file_not_found_error`
- [x] Empty raw data file produces a clear `ValueError` — `tests/test_edge_cases.py::test_load_raw_data_empty_file_raises_value_error`

## ✔ Preprocessing

- [x] Serialized preprocessing pipeline loads and transforms new data without refitting — `tests/test_integration.py::test_serialized_preprocessing_pipeline_loads_and_transforms`
- [x] Missing numeric/categorical values are imputed, not rejected — `tests/test_edge_cases.py::test_predict_with_missing_numeric_value_does_not_crash`, `test_predict_with_missing_categorical_value_does_not_crash`
- [x] Out-of-vocabulary categories (invalid grade, unexpected home-ownership value) are handled via `handle_unknown` encoder settings, not raised — `tests/test_edge_cases.py::test_predict_with_invalid_loan_grade_handled_via_unknown_encoding`, `test_predict_with_unexpected_categorical_value_handled_gracefully`
- [x] Extra/unexpected input columns are silently dropped (`ColumnTransformer(remainder="drop")`), not raised — `tests/test_edge_cases.py::test_predict_with_extra_unexpected_column_is_ignored_gracefully`
- [x] A genuinely missing required column raises a clear error — `tests/test_edge_cases.py::test_predict_with_missing_required_column_raises_clear_error`

## ✔ Model Inference

- [x] All three serialized models (Logistic Regression, Random Forest, XGBoost) load and predict without retraining — `tests/test_integration.py::test_serialized_model_loads_and_predicts[*]`
- [x] `RiskScoringEngine` produces a complete, valid prediction summary end-to-end — `tests/test_integration.py::test_risk_scoring_engine_end_to_end`
- [x] Negative income, extreme loan amounts, and very high DTI all produce a valid (if extreme) probability rather than crashing — `tests/test_edge_cases.py::test_predict_with_negative_income_does_not_crash`, `test_predict_with_extremely_large_loan_amount_does_not_crash`, `test_predict_with_very_high_dti_does_not_crash`
- [x] A zero-row batch degrades gracefully to an empty result (fixed in Phase 6 — previously raised a low-level sklearn error) — `tests/test_edge_cases.py::test_predict_probability_on_empty_dataframe_returns_empty_array`
- [x] Single-borrower methods reject multi-row input with a clear `ValueError` — `tests/test_edge_cases.py::test_predict_with_multiple_rows_rejected_by_single_borrower_methods`
- [x] An unknown model key raises a clear `ValueError` at construction time — `tests/test_edge_cases.py::test_risk_scoring_engine_raises_clear_error_for_unknown_model_key`

## ✔ Explainability

- [x] `ExplainabilityEngine` produces a complete local explanation (SHAP values, top factors, business summary) end-to-end — `tests/test_integration.py::test_explainability_engine_end_to_end`
- [x] `ExplainabilityEngine` and `RiskScoringEngine` agree on the same borrower's predicted probability (both wrap the same production model) — `tests/test_integration.py::test_explainability_engine_uses_same_probability_as_risk_scoring_engine`
- [x] Waterfall, force, summary (beeswarm + bar), dependence, and decision plots all render without error — `tests/test_explainability.py` (26 tests)
- [x] Multi-row input to a local-explanation method raises a clear error — `tests/test_edge_cases.py::test_explainability_engine_rejects_multi_row_local_explanation`

## ✔ Segmentation

- [x] `SegmentationEngine` fits on the training split and assigns every borrower to a named segment — `tests/test_integration.py::test_segmentation_engine_end_to_end`
- [x] Segment assignment works for algorithms without a native `.predict()` (Agglomerative), via nearest-centroid — `tests/test_segmentation_engine.py::test_assign_segment_works_for_agglomerative`
- [x] `compare_with_supervised_models()` correctly cross-references `RiskScoringEngine`'s predicted probabilities — `tests/test_integration.py::test_segmentation_engine_cross_references_risk_scoring_engine`
- [x] Calling any method before `.fit()` raises a clear `RuntimeError`, not an internal `AttributeError` — `tests/test_edge_cases.py::test_segmentation_engine_methods_raise_before_fit`
- [x] All three engines (`RiskScoringEngine`, `ExplainabilityEngine`, and `SegmentationEngine`'s internal `RiskScoringEngine`) agree on the same production model key — `tests/test_integration.py::test_all_three_engines_agree_on_same_production_model_key`

## ✔ Dashboard Pages

- [x] All 8 required pages exist and load without exception — `tests/test_app.py::test_every_page_loads_without_exception[*]` (8 parametrized cases)
- [x] Executive Dashboard shows all required KPIs — manually verified against Phase 5 brief
- [x] Borrower Risk Prediction form submits end-to-end and shows all 6 required result metrics — `tests/test_app.py::test_prediction_form_submits_and_shows_results`
- [x] Business Insights covers all 7 research questions in the required format — `tests/test_app.py::test_business_insights_covers_all_seven_research_questions`
- [x] About Project covers all required documentation sections — `tests/test_app.py::test_about_project_covers_required_sections`

## ✔ Navigation

- [x] `st.navigation`/`st.Page` routing correctly switches between all 8 pages — `tests/test_app.py::test_every_page_loads_without_exception[*]` (via `at.switch_page`)
- [x] The default page (Executive Dashboard) loads on first visit — `tests/test_app.py::test_default_page_loads_without_exception`
- [x] No broken links, missing page files, or import errors across any page — verified via `pyflakes app/` (zero findings) plus the above

## ✔ Downloads

- [x] CSV downloads work for the cleaned dataset, model comparison table, segment comparison table — manually verified (`download_dataframe_button` calls across pages)
- [x] Markdown + JSON exportable reports work for risk assessments, borrower explanations, global explanations, and segment summaries — manually verified (`download_report_buttons` calls across pages)
- [x] PNG export helper exists (`download_figure_button` in `app/common.py`) for any chart — available, not yet wired into every page (see Known Limitations)

## ✔ Error Handling

- [x] `RiskScoringEngine` calls in the prediction page are wrapped in try/except with a friendly message and full traceback logged — `app/app_pages/borrower_risk_prediction.py`
- [x] `ExplainabilityEngine` calls in the prediction page fail gracefully without losing the already-computed risk score — `app/app_pages/borrower_risk_prediction.py`
- [x] Segmentation visualizations and the supervised-model cross-check are wrapped in try/except — `app/app_pages/borrower_segmentation.py`
- [x] Missing artifacts (models, reports, datasets) show a friendly notice with the exact command to fix it, on every page that depends on them — `app/common.py::render_missing_artifact_notice`, used across all data-dependent pages
- [x] Corrupted serialized files raise a clear, catchable exception rather than silently returning garbage — `tests/test_edge_cases.py::test_load_object_corrupted_file_raises_readable_error`

## ✔ Logging

- [x] Every `src/` module logs via the shared, idempotent `utils.get_logger()` (console + file handler, no duplicate handlers on repeated calls) — `tests/test_integration.py::test_get_logger_returns_configured_logger_with_handlers`, `test_get_logger_is_idempotent_no_duplicate_handlers`
- [x] Application session start is logged once per session (not on every rerun) — `app/app.py`
- [x] Prediction requests are logged with model/loan-amount/grade context — `app/app_pages/borrower_risk_prediction.py`
- [x] Engine-level errors are logged with full tracebacks (`exc_info=True`) before showing the user a friendly message — `app/app_pages/borrower_risk_prediction.py`, `borrower_segmentation.py`
- [x] `logs/pipeline.log` is produced by `python -m src.train_models` and is human-readable — `tests/test_integration.py::test_pipeline_log_file_exists_after_running_pipeline`
- [x] Log level is configurable via the `MGMT590_LOG_LEVEL` environment variable without a code change

## ✔ Performance

- [x] Expensive per-page computations (t-SNE/UMAP, learning curves, global SHAP) are cached via `st.cache_data`, not recomputed on every rerun — see `PERFORMANCE_REPORT.md` Section 8
- [x] Models and the segmentation engine are loaded/fit once per server process via `st.cache_resource`, never retrained on a page visit
- [x] Prediction and single-borrower SHAP latency are both well under 100ms — see `PERFORMANCE_REPORT.md` Section 3
- [x] Memory footprint (~517 MB with all three engines loaded) fits comfortably within free-tier deployment limits — see `PERFORMANCE_REPORT.md` Section 6 and `DEPLOYMENT.md`

## ✔ Documentation References

- [x] `README.md` reflects every phase through Phase 6, with an accurate project-structure diagram and roadmap table
- [x] `PERFORMANCE_REPORT.md` (this phase) documents startup/latency/memory measurements and bottlenecks
- [x] `DEPLOYMENT.md` (this phase) documents environment setup, launch instructions, and platform-specific deployment guidance
- [x] `app/app_pages/about_project.py` documents the business problem, research questions, PDID framework, methodology, tech stack, limitations, and future improvements, viewable live in the running application

---

## Known Limitations (Carried Forward, Not Regressions)

- No legally protected class attributes exist in the dataset — the
  fairness assessment can only speak to business/financial attribute
  parity, documented in both the Phase 4A notebook and the About
  Project page.
- `SegmentationEngine.fit()`'s optimal-k evaluation cost (~12.5s) is not
  yet persisted/reused across cold starts — see `PERFORMANCE_REPORT.md`
  Section 9 for the deferred optimization.
- PNG chart export (`download_figure_button`) exists in `app/common.py`
  but is not yet wired into every page's charts — CSV and Markdown/JSON
  report exports are fully wired; PDF report generation was evaluated
  and intentionally not implemented (see `DEPLOYMENT.md`'s notes) since
  it would add a heavy new dependency for marginal benefit over the
  existing Markdown/JSON exports.
- This QA pass and `PERFORMANCE_REPORT.md`'s measurements were both
  taken against the synthetic test fixture, not the real ~37,515-row
  Indiana extract — re-run both once the genuine data is loaded.
