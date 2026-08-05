"""
Generates notebooks/MGMT590_LendingClub_Explainability_Phase4A.ipynb
using nbformat.
Run once from the project root: python tests/build_notebook_phase4a.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# MGMT 590 — LendingClub Loan Default Risk (Indiana Borrowers)
## Phase 4A: Explainable AI & Risk Scoring

**Course:** MGMT 59000, Summer 2026, Section DY2 — Purdue University

**Builds on:** Phase 1 (data pipeline), Phase 2 (EDA), Phase 3 (Logistic
Regression / Random Forest / XGBoost, tuned and evaluated).

**Scope of this notebook:** demonstrate the four new reusable Phase 4A
modules -- `src/configurable_thresholds.py`, `src/interpretation_utils.py`,
`src/risk_scoring.py`, `src/explainability.py` -- against the Phase 3
production model. This notebook is a thin demonstration/validation layer;
essentially all logic lives in those modules so the Streamlit dashboard
(Phase 5) can import `RiskScoringEngine` and `ExplainabilityEngine`
directly without modification.

**Not implemented here (explicitly deferred):** the Streamlit dashboard
itself (Phase 5) and borrower clustering (Phase 4B).

> **Note on data:** as in Phases 1-3, this notebook runs against
> whatever is currently at `data/splits/` and `models/`. If these still
> reflect the synthetic test fixture rather than the real ~37,515-row
> Indiana LendingClub extract, treat every specific number, factor
> ranking, and business claim below as illustrative of the *mechanism*
> only, not as a real finding -- re-run Phases 1, 3, and this notebook
> once the genuine data is in place.
""")

code("""import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils, model_utils, interpretation_utils as iu
from src.configurable_thresholds import load_threshold_config
from src.risk_scoring import RiskScoringEngine, expand_threshold_analysis, plot_expanded_threshold_analysis
from src.explainability import ExplainabilityEngine, DEFAULT_INTERACTION_PAIRS

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")

code("""X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Production model: {config.PRODUCTION_MODEL_KEY}")
""")

# =============================================================================
# 1. CONFIGURABLE THRESHOLDS
# =============================================================================
md("""## 1. Configurable Business Thresholds

`src/configurable_thresholds.py` centralizes every business-policy
boundary (risk tiers, lending actions, interest-rate adjustments,
loan-grade bands) as a `RiskThresholdConfig` dataclass with a JSON load/
save API — a lending-operations stakeholder can edit
`reports/risk_threshold_config.json` directly, with no code change or
redeploy required. Loading it here also self-bootstraps that file on
first run.""")

code("""threshold_config = load_threshold_config()
pd.DataFrame([vars(t) for t in threshold_config.risk_tiers])
""")

code("""pd.DataFrame([vars(a) for a in threshold_config.lending_actions])
""")

code("""pd.Series(threshold_config.interest_rate_adjustment_bps, name="rate_adjustment_bps").to_frame()
""")

code("""pd.DataFrame([vars(g) for g in threshold_config.loan_grade_bands])
""")

md("""**Design note:** these tiers are DELIBERATELY distinct from Phase 3's
statistically cost-minimizing decision threshold for each model
(`reports/threshold_analysis.joblib`). Phase 3's threshold answers "where
do we draw the line to minimize expected cost"; the tiers here answer
"how do we communicate and act on a *spectrum* of risk" (including
partial actions like a rate markup or manual review, not just a binary
accept/decline). `RiskScoringEngine` uses both, for different purposes —
see Section 2.""")

# =============================================================================
# 2. RISK SCORING ENGINE
# =============================================================================
md("""## 2. Risk Scoring Engine

`RiskScoringEngine` wraps one fitted Phase 3 model and turns raw
borrower rows into a probability, a 0-100 risk score, a risk tier, a
confidence score, and recommended action / interest rate / loan grade —
every business boundary read from `RiskThresholdConfig`, nothing
hard-coded.""")

code("""risk_engine = RiskScoringEngine()  # defaults to config.PRODUCTION_MODEL_KEY
print(f"Model: {risk_engine.model_display_name}")
print(f"Phase 3 cost-minimizing decision threshold: {risk_engine._decision_threshold:.2f}")
""")

md("### 2.1 Single-borrower prediction summary")
code("""example_borrower = X_test.iloc[[0]]
summary = risk_engine.generate_prediction_summary(example_borrower)
pd.Series(summary.to_dict())
""")

md("""### 2.2 Portfolio-level risk-tier distribution

`generate_batch_summary` is the vectorized equivalent — useful for
reporting on an entire loan book rather than one application-time
decision.""")
code("""batch_summary = risk_engine.generate_batch_summary(X_test)
batch_summary["risk_tier"].value_counts().reindex(
    [t.name for t in threshold_config.risk_tiers]
).fillna(0).astype(int)
""")

code("""fig, ax = plt.subplots(figsize=(8, 5))
tier_order = [t.name for t in threshold_config.risk_tiers]
counts = batch_summary["risk_tier"].value_counts().reindex(tier_order).fillna(0)
colors = ["#27AE60", "#F39C12", "#E67E22", "#C0392B"]
ax.bar(counts.index, counts.values, color=colors)
ax.set_ylabel("Number of Loans (Test Set)")
ax.set_title("Portfolio Risk-Tier Distribution", fontsize=13, fontweight="bold", loc="left")
fig.tight_layout()
plt.show()
""")

md("""**Business interpretation:** this is the shape of the loan book under
the current model and thresholds — a portfolio skewed heavily toward
"Very High Risk" would suggest either the applicant pool itself is risky
or the tier boundaries need revisiting; skewed heavily toward "Low Risk"
with very few declines suggests thresholds may be too permissive for the
observed default rate.""")

md("### 2.3 Exportable risk-assessment report")
code("""report = risk_engine.export_prediction_report(example_borrower)
print(report.to_markdown())
""")

# =============================================================================
# 3. EXPANDED THRESHOLD OPTIMIZATION
# =============================================================================
md("""## 3. Expanded Threshold Optimization

Extends Phase 3's cost-minimizing threshold analysis with two more
business-facing figures: **approval rate** (what share of applicants
would be approved at this cutoff) and **false-positive/false-negative
rate** (as rates, not just counts folded into a cost number).""")

code("""proba_val = risk_engine.predict_probability(X_val)
expanded_table = expand_threshold_analysis(y_val, proba_val)
expanded_table.head(10)
""")

code("""fig = plot_expanded_threshold_analysis(
    expanded_table, recommended_threshold=risk_engine._decision_threshold,
    model_display_name=risk_engine.model_display_name,
)
plt.show()
""")

md("""**Why this threshold was selected:** `risk_engine._decision_threshold`
is Phase 3's cost-minimizing cutoff (see `model_utils.recommend_threshold`),
computed on the VALIDATION set using the relative false-negative/false-
positive cost weights in `config.COST_FALSE_NEGATIVE` /
`config.COST_FALSE_POSITIVE`. **Which error type is costlier for Lending
Club:** a false negative (approving a borrower who actually defaults)
loses the principal and accrued interest and triggers collections costs
— a much larger loss than the forgone interest margin of a false
positive (declining a borrower who would have repaid). This is why
`config.COST_FALSE_NEGATIVE` (5.0) is set well above
`config.COST_FALSE_POSITIVE` (1.0), and why the recommended threshold
sits below 0.50 rather than at it — the model is deliberately biased
toward catching more defaulters at the cost of declining somewhat more
good borrowers.""")

# =============================================================================
# 4. EXPLAINABILITY ENGINE -- GLOBAL
# =============================================================================
md("""## 4. Explainability Engine: Global Model Explanations

`ExplainabilityEngine` wraps the same production model with SHAP
(`TreeExplainer` for XGBoost/Random Forest, `LinearExplainer` for
Logistic Regression — see the module docstring in `explainability.py`
for why). SHAP values below are in the model's log-odds (margin) output
space, not raw probability — labeled accordingly on every plot.""")

code("""explain_engine = ExplainabilityEngine()  # shares config.PRODUCTION_MODEL_KEY
print(f"Explainer: {type(explain_engine.explainer).__name__}")
""")

md("### 4.1 SHAP Summary Plots (Beeswarm and Bar)")
code("""fig = explain_engine.generate_shap_summary(X_test, plot_type="beeswarm")
plt.show()
""")
md("""**Business interpretation:** each dot is one borrower; its horizontal
position is that borrower's SHAP value for the feature (how much that
feature pushed their prediction up/down), and its color is the
borrower's actual value for that feature (red = high, blue = low). A
feature where red dots cluster on the right and blue dots on the left
means higher values of this feature increase predicted default risk.""")

code("""fig = explain_engine.generate_shap_summary(X_test, plot_type="bar")
plt.show()
""")

md("### 4.2 Feature Importance Comparison")
code("""importance_table = explain_engine.summarize_feature_importance(X_test)
importance_table.head(15)
""")
md("""**Comparing three independent importance methods** (SHAP mean |value|,
Phase 3's permutation importance, and the model's native importance) —
features ranked highly across ALL THREE are the most trustworthy
candidates for "genuinely predictive," versus a feature that only one
method rates highly, which may reflect that method's own biases (e.g.
impurity importance's known bias toward high-cardinality features,
discussed in Phase 3).""")

md("### 4.3 Global Model Explanation and Executive Summary")
code("""global_explanation = explain_engine.explain_global_model(X_test)
print(global_explanation.business_summary)
""")

code("""print("Most influential variables:")
for f in global_explanation.top_features:
    print(f"  - {f}")
print("\\nLeast influential variables:")
for f in global_explanation.least_influential_features:
    print(f"  - {f}")
print("\\nPositive contributors to default risk (higher value -> higher risk):")
for f in global_explanation.positive_contributors:
    print(f"  - {f}")
print("\\nNegative contributors to default risk (higher value -> lower risk, i.e. protective):")
for f in global_explanation.negative_contributors:
    print(f"  - {f}")
""")

md("### 4.4 SHAP Dependence Plots")
code("""fig = explain_engine.generate_dependence_plot("dti", X_test, interaction_feature="int_rate")
plt.show()
""")
md("""**Business interpretation:** the x-axis is the borrower's actual DTI;
the y-axis is how much that DTI value contributed to their predicted
risk (in log-odds). An upward-sloping pattern confirms higher DTI raises
predicted risk; color (by interest rate) reveals whether that
relationship is stronger or weaker depending on the borrower's rate —
i.e. an interaction effect.""")

code("""fig = explain_engine.generate_dependence_plot("annual_inc", X_test, interaction_feature="dti")
plt.show()
""")

md("### 4.5 SHAP Decision Plot")
code("""fig = explain_engine.generate_decision_plot(X_test, n_samples=25)
plt.show()
""")
md("""**Business interpretation:** each line is one borrower's cumulative
SHAP trajectory from the model's average prediction (left) to their
final predicted risk (right). Lines that diverge sharply from the
bundle partway through highlight borrowers whose risk is driven by an
unusual feature combination rather than the typical pattern — useful for
spotting edge cases a credit analyst should review manually.""")

nb["cells"] = cells

# =============================================================================
# 5. EXPLAINABILITY ENGINE -- LOCAL (INDIVIDUAL BORROWER)
# =============================================================================
md("""## 5. Explainability Engine: Local (Individual Borrower) Explanations

For one specific borrower: SHAP waterfall and force plots, the top risk
and protective factors, and a plain-language business summary that
reads like something a credit analyst would write — no ML jargon.""")

code("""local_explanation = explain_engine.explain_prediction(example_borrower)
print(local_explanation.business_summary)
""")

code("""local_explanation.feature_contributions.head(10)
""")

md("### 5.1 SHAP Waterfall Plot")
code("""fig = explain_engine.generate_waterfall_plot(example_borrower)
plt.show()
""")
md("""**Business interpretation:** starting from the model's average
prediction (the base value, in log-odds), each bar shows how much one
feature moved this specific borrower's prediction up (red, toward higher
default risk) or down (blue, toward lower default risk), ending at their
final predicted value.""")

md("### 5.2 SHAP Force Plot")
code("""fig = explain_engine.generate_force_plot(example_borrower)
plt.show()
""")
md("""**Business interpretation:** the same additive attribution as the
waterfall plot, shown as opposing "push" forces — red features pushing
right (toward higher risk), blue features pushing left (toward lower
risk) — from the base value to the final prediction arrow.""")

md("### 5.3 Top Risk Factors and Top Protective Factors")
code("""print("Top risk factors:")
for f in local_explanation.top_risk_factors:
    print(f"  - {f}")
print("\\nTop protective factors:")
for f in local_explanation.top_protective_factors:
    print(f"  - {f}")
""")

# =============================================================================
# 6. FEATURE INTERACTION ANALYSIS
# =============================================================================
md("""## 6. Feature Interaction Analysis

The five interaction pairs from the Phase 4A brief. Pairs where BOTH
sides resolve to a single numeric/ordinal column (Income x DTI,
Grade x Interest Rate, Employment Length x Income) use a SHAP dependence
plot colored by the interacting feature. Pairs involving a one-hot
categorical feature (Purpose x Grade, Home Ownership x DTI) instead use
a mean-predicted-risk heatmap, since SHAP's scatter-style dependence
plot cannot represent a multi-column categorical feature on one axis
(see the design note in `ExplainabilityEngine.analyze_feature_interactions`).""")

code("""interaction_results = explain_engine.analyze_feature_interactions(X_test)

for (primary, interacting), result in interaction_results.items():
    print(f"=== {primary} x {interacting} ({result['kind']}) ===")
""")

for primary, interacting in [
    ("annual_inc", "dti"), ("grade", "int_rate"), ("emp_length_years", "annual_inc"),
    ("purpose", "grade"), ("home_ownership", "dti"),
]:
    code(f"""result = interaction_results[("{primary}", "{interacting}")]
result["figure"]
""")
    md(f"**{primary.replace('_', ' ').title()} x {interacting.replace('_', ' ').title()}:** "
       f"see the printed interpretation below the combined cell in Section 6 for the "
       f"generated business interpretation of this specific pair.")

code("""for (primary, interacting), result in interaction_results.items():
    print(f"--- {primary} x {interacting} ---")
    print(result["interpretation"])
    print()
""")

# =============================================================================
# 7. PARTIAL DEPENDENCE / ICE
# =============================================================================
md("""## 7. Partial Dependence and ICE Plots

A second, model-agnostic (non-SHAP) lens on marginal effects, computed
directly on the fitted `Pipeline` via scikit-learn's
`PartialDependenceDisplay`. The thick line is the Partial Dependence
(average effect across all borrowers); the thin lines are Individual
Conditional Expectation curves (one per borrower) — divergence among ICE
lines signals the marginal effect is NOT uniform across borrowers (an
interaction with some other feature, even though PDP alone wouldn't show
which one).""")

code("""fig = explain_engine.generate_pdp_ice_plot(X_test, features=["dti", "int_rate", "annual_inc"], kind="both")
plt.show()
""")

md("""**Business interpretation:** a rising PD line for DTI confirms the
marginal effect of debt burden on predicted risk holds "all else equal";
a flattening slope past some DTI level would indicate a threshold effect
(risk stops climbing meaningfully beyond that point) rather than a
purely linear relationship — relevant for setting a DTI-based underwriting
cutoff versus a continuous rate adjustment.""")

# =============================================================================
# 8. MODEL FAIRNESS
# =============================================================================
md("""## 8. Model Fairness Assessment

**Important limitation, stated up front:** this dataset contains no
legally protected class attributes (race, gender, age, religion,
national origin, etc.) — it was never collected with them. This
assessment can only speak to performance parity across the
BUSINESS/FINANCIAL attributes actually present. It cannot support or
refute any claim about fairness with respect to legally protected
classes, and should not be represented as doing so.""")

code("""proba_test = risk_engine.predict_probability(X_test)
fairness_table = iu.fairness_report(
    X_test, y_test, proba_test,
    group_columns=["home_ownership", "purpose", "grade"],
    threshold=risk_engine._decision_threshold,
)
fairness_table[["group_column", "group_value", "n_loans", "actual_default_rate", "recall", "precision", "roc_auc"]]
""")

code("""disparities = iu.summarize_fairness_disparities(fairness_table, metric="recall")
disparities
""")

md("""**How to read this:** the `recall_spread` column shows how much
recall (the share of actual defaulters correctly caught) varies across
categories of each grouping column. A large spread means the model
catches defaulters much more reliably in some groups (e.g. some loan
purposes) than others — worth investigating WHY (often a data-volume
issue: groups with fewer historical loans give the model less to learn
from) before concluding the model treats groups inequitably. This
assessment flags where to look, not a definitive fairness verdict.""")

# =============================================================================
# 9. RESEARCH QUESTION SUPPORT
# =============================================================================
md("""## 9. Connecting Findings to Research Questions

Every top-ranked feature in Section 4.3's importance table can be traced
back to a specific Phase 2 research question via
`interpretation_utils.link_feature_to_research_question`.""")

code("""importance_table["research_question"] = importance_table["feature"].apply(iu.link_feature_to_research_question)
importance_table[["feature_label", "mean_abs_shap", "research_question"]].head(10)
""")

md("""**Business recommendation, by research question** *(fill in the
specific numbers once run against the real Indiana extract)*:
- **RQ1** (borrower characteristics associated with default): the top
  entries in Section 4.3's table not already covered by RQ2-RQ6 below —
  prioritize collecting/verifying these fields at application time.
- **RQ2** (are LendingClub grades predictive?): check `grade`'s rank and
  SHAP direction in Section 4.3/4.4 — if it ranks highly with a positive
  SHAP relationship to risk, grades remain a useful signal even after
  controlling for the other features in this model.
- **RQ3** (variables related to higher interest rates): cross-reference
  which features rank highly for both default risk (this notebook) and
  `int_rate` itself (Phase 2's correlation analysis) — features
  important to both suggest LendingClub's own rate-setting already
  partially reflects the risk factors this model identifies.
- **RQ4** (does income relate to repayment success?): see
  `annual_inc`'s SHAP direction in Section 4.4 — a negative relationship
  (higher income -> lower risk) supports RQ4's hypothesis.
- **RQ5** (does DTI influence default?): Section 4.4's DTI dependence
  plot directly answers this, including whether the effect is linear or
  has a threshold.
- **RQ6** (does employment length matter?): check `emp_length_years`'s
  rank in Section 4.3 — a low rank across all three importance methods
  would suggest employment length matters less than commonly assumed
  once income, DTI, and grade are already known.
- **RQ7** (natural borrower groups before clustering): NOT addressed by
  this notebook — explicitly deferred to Phase 4B's clustering analysis.
""")

# =============================================================================
# 10. EXPORTABLE REPORTS
# =============================================================================
md("""## 10. Exportable Reports

Every report below is an `interpretation_utils.ExportableReport` —
`.to_markdown()` / `.to_json()` output is ready to be handed directly to
a future Streamlit `st.download_button()`, and `.save(path, fmt=...)`
writes it to disk.""")

code("""borrower_report = explain_engine.export_borrower_explanation_report(example_borrower)
print(borrower_report.to_markdown())
""")

code("""global_report = explain_engine.export_global_explanation_report(X_test)
print(global_report.to_markdown()[:2000])
""")

code("""risk_report = risk_engine.export_prediction_report(example_borrower)
print(risk_report.to_markdown())
""")

# =============================================================================
# 11. SAVE ARTIFACTS
# =============================================================================
md("""## 11. Persisting Reusable Explainability Artifacts

`ExplainabilityEngine.persist_explainability_artifacts()` serializes
every reusable Phase 4A artifact a future Streamlit dashboard needs
(feature-importance comparison table, business-summary templates, model
metadata, fairness report, feature-interaction interpretations) via
`joblib`, so the dashboard never needs to recompute SHAP values just to
render a cached summary table.""")

code("""explain_engine.persist_explainability_artifacts(X=X_test, y=y_test)

print("Saved artifacts:")
print(f"  SHAP importance table:       {config.SHAP_IMPORTANCE_PATH}")
print(f"  Business summary templates:  {config.BUSINESS_SUMMARY_TEMPLATES_PATH}")
print(f"  Model metadata:              {config.MODEL_METADATA_PATH}")
print(f"  Fairness report:             {config.FAIRNESS_REPORT_PATH}")
print(f"  Feature interaction summary: {config.FEATURE_INTERACTION_SUMMARY_PATH}")
print(f"  Risk threshold configuration: {config.RISK_THRESHOLD_CONFIG_PATH}")
""")

# =============================================================================
# 12. PHASE TRANSITION
# =============================================================================
md("""## 12. Preparing for Phase 4B / Phase 5

### How Explainable AI improved understanding of the predictive models

Phase 3 established WHICH model performs best; Phase 4A establishes WHY
it makes the predictions it does, at both the population level (Section
4) and the individual-borrower level (Section 5), cross-checked against
Phase 3's own permutation/native importance measures (Section 4.2) so no
single method's biases go unchallenged.

### How `ExplainabilityEngine` will integrate into Streamlit

Every method returns either a `matplotlib.figure.Figure` (ready for
`st.pyplot(fig)`) or a plain dataclass/DataFrame/string (ready for
`st.dataframe()`, `st.write()`, or `st.download_button()` via
`ExportableReport`). A borrower-detail page would call:

```python
engine = ExplainabilityEngine()  # cached via st.cache_resource
local = engine.explain_prediction(borrower_row)
st.pyplot(engine.generate_waterfall_plot(borrower_row))
st.write(local.business_summary)
st.download_button("Download explanation", engine.export_borrower_explanation_report(borrower_row).to_markdown())
```

### How `RiskScoringEngine` will integrate into Streamlit

A single-borrower scoring form would call:

```python
risk_engine = RiskScoringEngine()  # cached via st.cache_resource
summary = risk_engine.generate_prediction_summary(borrower_row)
st.metric("Risk Score", summary.risk_score)
st.metric("Risk Tier", summary.risk_tier)
st.metric("Recommended Action", summary.recommended_action)
```

A portfolio-level page would call `generate_batch_summary(X)` instead,
and `configurable_thresholds.load_threshold_config()` /
`RiskThresholdConfig.save()` could back an admin settings page letting a
lending-operations user adjust tier boundaries live.

### Public interfaces Phase 5 should use

| Module | Public interface |
|---|---|
| `src.configurable_thresholds` | `load_threshold_config()`, `RiskThresholdConfig` |
| `src.risk_scoring` | `RiskScoringEngine`, `PredictionSummary`, `expand_threshold_analysis`, `plot_expanded_threshold_analysis` |
| `src.explainability` | `ExplainabilityEngine`, `LocalExplanation`, `GlobalExplanation` |
| `src.interpretation_utils` | `humanize_feature_name`, `ExportableReport`, `fairness_report` |

### Explicitly deferred

- Borrower clustering (Phase 4B) — will use the same `X_train`/`X_test`
  splits and can layer cluster labels onto `RiskScoringEngine`'s batch
  summaries to show risk *by segment*, not just in aggregate.
- The Streamlit dashboard itself (Phase 5) — the architecture above is
  prepared for it; no dashboard code has been written in this phase.
""")

with open("notebooks/MGMT590_LendingClub_Explainability_Phase4A.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Phase 4A notebook written with {len(cells)} cells.")
