"""
Generates notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb using nbformat.
Run once from the project root: python tests/build_notebook_phase3.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# =============================================================================
# TITLE & INTRO
# =============================================================================
md("""# MGMT 590 — LendingClub Loan Default Risk (Indiana Borrowers)
## Phase 3: Supervised Machine Learning

**Course:** MGMT 59000, Summer 2026, Section DY2 — Purdue University

**Builds on:** Phase 1 (data pipeline, leakage-safe train/val/test split,
preprocessing `ColumnTransformer`) and Phase 2 (exploratory data
analysis, research-question analysis, statistical testing).

**Scope of this notebook:** train, tune, and evaluate three supervised
classifiers — Logistic Regression, Random Forest, and XGBoost — using
`src/model_utils.py` (new, additive Phase 3 module). For every model:
cross-validated hyperparameter search, the full evaluation-metric suite,
diagnostic visualizations, feature importance, and threshold
optimization. The notebook ends with an executive model-comparison
table, a production-model recommendation, a robustness assessment, and a
hand-off summary for Phase 4 (SHAP explanations + clustering — **not**
implemented here).

> **Note on data:** as in Phases 1-2, this notebook runs against
> whatever is currently at `data/splits/` (produced by
> `src/train_models.py`'s `run_phase1_pipeline()`). If you have not yet
> replaced the synthetic test fixture with the real ~37,515-row Indiana
> LendingClub extract, the numbers below reflect synthetic data and
> **should not be interpreted as real findings** — re-run Phase 1 and
> this notebook once the genuine data is in place.
""")

code("""import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

# A small number of FutureWarning/UserWarning messages come from a
# scikit-learn version newer than the one pinned in requirements.txt
# (LogisticRegression's `penalty` deprecation path). They do not affect
# correctness and are silenced here purely for notebook readability;
# on the pinned scikit-learn version (<1.6) they do not occur at all.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils, model_utils

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")

md("""## 1. Load the Phase 1 Splits and Preprocessing Pipeline

Reusing Phase 1's artifacts without modification: the leakage-safe
train/validation/test split (`utils.load_splits()`) and, conceptually,
the preprocessing pipeline builder (`utils.build_preprocessing_pipeline`)
that Phase 3's `model_utils.build_model_pipeline()` wraps around each
classifier — see the design-decision note in the next cell for why a
*fresh* (not the Phase 1 *fitted*) preprocessor is embedded per model.""")

code("""X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Default rate — train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")
""")

md("""### Design decision: preprocessing inside a cross-validated `Pipeline`

Phase 1 fit one preprocessing `ColumnTransformer` once on the full
`X_train` and serialized it (`pipelines/preprocessing_pipeline.joblib`) —
correct and sufficient for Phase 2's one-off EDA transforms. Phase 3
instead builds a **fresh, unfitted** preprocessor per model
(`utils.build_preprocessing_pipeline()`, the same construction function)
and wraps it together with the classifier in a single scikit-learn
`Pipeline` (`model_utils.build_model_pipeline`). This matters because
hyperparameter search repeatedly refits on different folds of the
training data — only a preprocessor living *inside* the `Pipeline` gets
refit correctly on each fold's training portion. Reusing the one
Phase 1 preprocessor (fit on the entire `X_train`) across cross-
validation folds would leak each validation fold's own statistics
(medians, means, encoder categories) into that fold's training step.
Both preprocessors are built from the exact same
`utils.build_preprocessing_pipeline()` function — nothing about the
transformation logic itself changes between phases.
""")

# =============================================================================
# 2. ALGORITHM OVERVIEW
# =============================================================================
md("""## 2. Model Development Overview

Three supervised classifiers are developed and compared, each serving a
different purpose in a production lending-decision pipeline:

| Model | Purpose | Interpretability | Complexity |
|---|---|---|---|
| **Logistic Regression** | Highly interpretable baseline — coefficients map directly to odds ratios a credit-risk committee can read and defend | High | Low |
| **Random Forest** | Captures nonlinear relationships and feature interactions without manual feature engineering | Medium | Medium-High |
| **XGBoost** | Candidate production model — typically the strongest raw discrimination on structured/tabular data | Medium-Low (needs SHAP, Phase 4) | High |

Full algorithm rationale (why chosen, advantages, disadvantages, business
tradeoffs, computational complexity) is documented in each pipeline
builder's docstring in `src/model_utils.py`
(`build_logistic_regression_pipeline`, `build_random_forest_pipeline`,
`build_xgboost_pipeline`) and repeated in context below as each model is
trained.

### Cross-validation and search-strategy choices

- **5-fold Stratified K-Fold** everywhere (`config.CV_FOLDS`): with a
  training set in the low thousands of loans and a minority (default)
  class around 20-25%, 5 folds keeps enough positive cases in every
  validation fold for stable estimates; 10 folds would roughly halve
  that per-fold count for little bias benefit. Stratification preserves
  the overall default rate in every fold.
- **Primary metric: ROC-AUC** (`config.CV_SCORING`), not accuracy — with
  an imbalanced target, a model can score 75-80% accuracy by predicting
  "no default" for everyone. ROC-AUC measures ranking quality across all
  thresholds, which matters because the business chooses its own
  operating threshold after training (Section 6).
- **GridSearchCV for Logistic Regression**: its parameter space (a
  handful of `C` values × 2 penalties) is small enough to search
  exhaustively, guaranteeing the grid's global optimum.
- **RandomizedSearchCV for Random Forest and XGBoost**: their
  combinatorial parameter spaces (tree count, depth, split/leaf sizes,
  feature sampling, learning rate, regularization, ...) are too large to
  grid-search exhaustively within a reasonable compute budget; sampling
  a fixed number of combinations explores the space far more efficiently
  per unit of compute (Bergstra & Bengio, 2012).
""")

# =============================================================================
# 3. HELPER: metrics table + business interpretation text
# =============================================================================
md("""## 3. Shared Metric-Interpretation Reference

Before training any model, here is what each Phase 3 metric means **in
the context of a lending decision**, and why the notebook does not stop
at accuracy:""")

code("""METRIC_BUSINESS_MEANING = {
    "accuracy": "Overall % correct — MISLEADING alone here: with ~20-25% default, "
                "predicting 'no default' for everyone already scores ~75-80%.",
    "precision": "Of loans flagged as risky, how many really default? High precision "
                 "= a 'decline' policy built on this model wastes few good customers.",
    "recall": "Of loans that actually default, how many did we catch? Directly drives "
              "how much bad debt is avoided.",
    "specificity": "Of loans actually repaid, how many did we correctly clear? Low "
                   "specificity = good borrowers declined unnecessarily (lost revenue).",
    "f1_score": "Harmonic mean of precision/recall — one number balancing both error "
                "types, more informative than accuracy under class imbalance.",
    "roc_auc": "Threshold-independent ranking quality: P(model scores a random "
               "defaulter higher than a random non-defaulter). Central here because "
               "the business sets its own operating threshold after training.",
    "average_precision": "Area under the Precision-Recall curve — more informative "
                          "than ROC-AUC when the positive class is rare and precision "
                          "at high recall matters most.",
    "balanced_accuracy": "Average of recall and specificity — an imbalance-robust "
                         "alternative to plain accuracy.",
    "matthews_corrcoef": "Single balanced summary of the full confusion matrix; "
                         "considered one of the most reliable metrics for imbalanced "
                         "binary classification.",
    "log_loss": "Penalizes confident-but-wrong probability estimates heavily — "
                "relevant if predicted probabilities feed pricing or risk-based cutoffs.",
    "brier_score": "Mean squared error of predicted probabilities — a simpler, "
                   "non-log complement to log loss.",
    "calibration_error": "Are the probabilities themselves trustworthy at face value "
                         "(independent of ranking quality)? Near 0 = 'when the model "
                         "says 30% risk, ~30% of those loans really default.'",
}

pd.DataFrame(
    [{"metric": k, "business_meaning": v} for k, v in METRIC_BUSINESS_MEANING.items()]
)
""")


def add_model_section(model_key: str, algorithm_intro_md: str,
                       validation_curve_param: str, validation_curve_range: str,
                       importance_cells_fn):
    """
    Append a full Phase 3 model section (train -> CV -> metrics ->
    visualizations -> feature importance -> executive summary) to the
    module-level `cells` list. Factored into one function and called
    once per model so the three (very similarly structured) sections
    don't require hand-duplicating notebook cell code three times.
    """
    display_name = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }[model_key]

    md(algorithm_intro_md)

    code(f"""result_{model_key} = model_utils.train_and_evaluate_model(
    "{model_key}", X_train, y_train, X_val, y_val, X_test, y_test,
)
print(f"Best params: {{result_{model_key}.best_params}}")
""")

    md(f"""### {display_name}: Cross-Validation Results

Fold-by-fold `config.CV_SCORING` for the best hyperparameter
combination found by the search, plus the mean and standard deviation
across the `config.CV_FOLDS` folds.""")
    code(f"""result_{model_key}.cv_fold_results
""")

    md(f"""### {display_name}: Evaluation Metrics (Test Set)

Computed at the default 0.50 threshold for like-for-like comparison
across models (Section 6 revisits the recommended, cost-minimizing
threshold instead).""")
    code(f"""pd.Series(result_{model_key}.test_metrics, name="{model_key}").to_frame()
""")

    md(f"""### {display_name}: Train / Validation / Test Comparison (Overfitting Check)

Comparing ROC-AUC across the three splits at the same 0.50 threshold —
a large train-vs-validation/test gap signals overfitting (high
variance); uniformly low scores across all three signal underfitting
(high bias).""")
    code(f"""pd.DataFrame({{
    "train": result_{model_key}.train_metrics,
    "validation": result_{model_key}.val_metrics,
    "test": result_{model_key}.test_metrics,
}}).T[["roc_auc", "f1_score", "recall", "precision", "log_loss"]]
""")

    md(f"### {display_name}: Diagnostic Visualizations")

    code(f"""fig = model_utils.plot_confusion_matrix_chart(
    y_test, (result_{model_key}.y_proba_test >= 0.5).astype(int),
    title=f"{display_name}: Confusion Matrix (Test Set)",
    subtitle="Counts of predicted vs. actual loan outcome at the 0.50 threshold",
)
plt.show()
""")
    md("""**Business interpretation:** the bottom-right and top-left cells are
correct calls; the bottom-left cell (predicted "Fully Paid" but actually
defaulted) is the costly miss the business most wants to shrink, while
the top-right cell (predicted "Default" but actually paid) represents
good customers turned away.""")

    code(f"""fig = model_utils.plot_roc_curve_chart(
    y_test, result_{model_key}.y_proba_test,
    title=f"{display_name}: ROC Curve (Test Set)",
    subtitle="Discrimination between defaulters and non-defaulters across all thresholds",
)
plt.show()
""")
    md("""**Business interpretation:** the further the curve bows toward the
top-left corner (away from the diagonal no-skill line), the better the
model separates future defaulters from future good borrowers regardless
of which threshold is ultimately chosen.""")

    code(f"""fig = model_utils.plot_pr_curve_chart(
    y_test, result_{model_key}.y_proba_test,
    title=f"{display_name}: Precision-Recall Curve (Test Set)",
    subtitle="More informative than ROC when the positive class (default) is the minority",
)
plt.show()
""")

    code(f"""fig = model_utils.plot_calibration_curve_chart(
    y_test, result_{model_key}.y_proba_test,
    title=f"{display_name}: Calibration Curve (Test Set)",
    subtitle="Do predicted probabilities match observed default rates?",
)
plt.show()
""")
    md("""**Business interpretation:** points on the diagonal mean predicted
probabilities can be taken at face value (e.g. used directly for
risk-based pricing); points below the diagonal mean the model
over-states risk in that probability range, points above mean it
under-states risk.""")

    code(f"""fig = model_utils.plot_probability_distribution_chart(
    y_test, result_{model_key}.y_proba_test,
    title=f"{display_name}: Predicted Probability Distribution by Actual Outcome",
    subtitle="Well-separated humps indicate strong discriminative power",
)
plt.show()
""")

    code(f"""fig = model_utils.plot_learning_curve_chart(
    result_{model_key}.best_estimator, X_train, y_train,
    title=f"{display_name}: Learning Curve",
    subtitle="Training vs. cross-validated ROC-AUC as training-set size grows",
)
plt.show()
""")
    md("""**Business interpretation:** a persistent, wide gap between the
training and cross-validation lines indicates high variance (the model
is overfitting the training data — more data or stronger regularization
would likely help); both lines converging at a low score instead
indicates high bias (the model is too simple/constrained for the
signal available — more data alone would not help much).""")

    code(f"""fig = model_utils.plot_validation_curve_chart(
    result_{model_key}.best_estimator, X_train, y_train,
    param_name="{validation_curve_param}",
    param_range={validation_curve_range},
    title=f"{display_name}: Validation Curve ({validation_curve_param.replace('classifier__', '')})",
    subtitle="Where does increasing complexity stop helping generalization?",
)
plt.show()
""")
    md(f"""**Business interpretation:** the point where the cross-validation
line plateaus or turns down while the training line keeps climbing marks
where `{validation_curve_param.replace('classifier__', '')}` starts
overfitting rather than improving genuine predictive power — a practical
ceiling for that hyperparameter beyond the one the search already found.""")

    md(f"### {display_name}: Feature Importance")
    importance_cells_fn(model_key, display_name)

    md(f"""### {display_name}: Threshold Analysis""")
    code(f"""fig = model_utils.plot_threshold_analysis_chart(
    result_{model_key}.threshold_table, result_{model_key}.recommended_threshold,
    title=f"{display_name}: Threshold Optimization",
    subtitle="Optimized on the VALIDATION set — test set stays untouched by this decision",
)
plt.show()
print(f"Recommended threshold: {{result_{model_key}.recommended_threshold:.2f}}")
""")

    md(f"""### {display_name}: Executive Summary

- **What did we discover?** Compare `result_{model_key}`'s test ROC-AUC/
  F1/recall (printed above) against the other two models once all three
  sections have run, and note the CV mean ± std stability from the
  cross-validation table.
- **Why does it matter?** The confusion-matrix cost balance (missed
  defaulters vs. wrongly declined good borrowers) for {display_name}
  reflects its interpretability/complexity tradeoff described in
  Section 2.
- **How should Lending Club use this information?** Whether
  {display_name} is better suited to policy explanation, a supporting
  model in an ensemble, or the primary production scorer is addressed in
  Section 7's comparison table and Section 8's recommendation.
- **Which research question(s) does it address?** Research Questions 1
  ("which borrower characteristics appear associated with default") and
  2 ("do LendingClub grades appear predictive") most directly, via the
  feature-importance table above.
""")


# ---------------------------------------------------------------------------
# Logistic Regression section
# ---------------------------------------------------------------------------

lr_intro = """## 4. Logistic Regression

**Why this algorithm:** included as the highly interpretable baseline.
Its coefficients translate directly into odds ratios a credit-risk
committee can read and defend, it trains in milliseconds, and it gives
every later model a bar to beat.

**Advantages:** transparent, well-calibrated by construction (linear
log-odds), cheap to train/predict, robust with modest data.

**Disadvantages:** assumes a linear (in log-odds) relationship between
features and outcome and no interaction effects unless engineered
manually; cannot capture nonlinear risk patterns Random Forest/XGBoost
can.

**Business tradeoff:** sacrifices some predictive power for maximum
auditability — often preferred for the parts of a credit decision that
must be explained to a regulator or a declined applicant.

**Computational complexity:** O(n_features × n_samples) per solver
iteration — trivial at this dataset's scale.

**Hyperparameter search space** (`config.LOGISTIC_REGRESSION_PARAM_GRID`,
searched exhaustively via `GridSearchCV`):
- `C` ∈ {0.001, 0.01, 0.1, 1, 10, 100} — inverse regularization strength;
  spans from heavy to negligible regularization.
- `penalty` ∈ {l1, l2} — l1 can zero out weak predictors (built-in
  feature selection), l2 shrinks all coefficients smoothly.
- `class_weight` ∈ {None, "balanced"} — tests whether explicitly
  rebalancing the minority (default) class improves ranking quality.
"""


def lr_importance_cells(model_key, display_name):
    code(f"""coef_table = result_{model_key}.feature_importance["coefficients"]
coef_table.head(15)
""")
    md("""**Interpretation:** for a one-unit increase in a standardized
numeric feature (or moving into a one-hot category), the odds of default
multiply by the odds ratio — e.g. an odds ratio of 1.20 means a 20%
increase in the odds of default; 0.80 means a 20% decrease. The
strongest positive predictors (odds ratio > 1) increase default risk;
the strongest negative predictors (odds ratio < 1) are protective.""")
    code(f"""fig = model_utils.plot_coefficient_chart(
    coef_table, title=f"{display_name}: Coefficient Plot (Top 15 by |coefficient|)",
    subtitle="Red = increases default odds, Blue = decreases default odds; OR = odds ratio",
)
plt.show()
""")
    code(f"""perm_table = result_{model_key}.feature_importance["permutation"]
perm_table.head(10)
""")
    code(f"""fig = model_utils.plot_importance_bar(
    perm_table, value_column="importance_mean",
    title=f"{display_name}: Permutation Importance (raw features)",
    subtitle="Drop in ROC-AUC when a raw feature's values are randomly shuffled",
)
plt.show()
""")


add_model_section(
    "logistic_regression", lr_intro,
    validation_curve_param="classifier__C",
    validation_curve_range="[0.001, 0.01, 0.1, 1, 10, 100]",
    importance_cells_fn=lr_importance_cells,
)

# ---------------------------------------------------------------------------
# Random Forest section
# ---------------------------------------------------------------------------

rf_intro = """## 5. Random Forest

**Why this algorithm:** an ensemble of decision trees (via bagging) that
captures nonlinear relationships and feature interactions (e.g. "high
DTI matters more when income is also low") without hand-engineering
them, while still offering intuitive impurity-based and permutation
feature importances.

**Advantages:** handles nonlinearity/interactions natively, robust to
outliers and unscaled features, low risk of severe overfitting thanks to
bagging, minimal preprocessing requirements.

**Disadvantages:** individual trees are uninterpretable (the ensemble
offers importance measures, not "why this one borrower" — that gap is
exactly what Phase 4's SHAP analysis will close); larger memory
footprint than a single linear model; impurity importance is biased
toward high-cardinality numeric features.

**Business tradeoff:** meaningfully better discrimination potential than
Logistic Regression at the cost of some interpretability and larger
serialized model size.

**Computational complexity:** O(n_trees × n_samples × log(n_samples) ×
n_features) for training; prediction is O(n_trees × tree_depth) — fast,
but slower than a single logistic regression scoring pass.

**Hyperparameter search space**
(`config.RANDOM_FOREST_PARAM_DISTRIBUTIONS`, searched via
`RandomizedSearchCV` with `n_iter=20` — see Section 2 for why randomized
rather than grid search is used here):
- `n_estimators` ∈ {200, 300, 400, 500} — more trees generally reduce
  variance but with diminishing returns past a point.
- `max_depth` ∈ {4, 6, 8, 10, 12, 16, None} — controls how much each
  tree can overfit individually.
- `min_samples_split`, `min_samples_leaf` — additional overfitting
  controls at the leaf level.
- `max_features` ∈ {"sqrt", "log2", 0.5} — how many features each split
  considers, which decorrelates trees from one another.
- `class_weight` ∈ {None, "balanced", "balanced_subsample"} — tests
  minority-class rebalancing strategies.
"""


def rf_importance_cells(model_key, display_name):
    code(f"""impurity_table = result_{model_key}.feature_importance["impurity"]
impurity_table.head(15)
""")
    md("""**Caveat:** impurity importance is biased toward high-cardinality /
continuous numeric features and can overstate importance for features
with many possible split points — permutation importance below is
computed as a cross-check.""")
    code(f"""fig = model_utils.plot_importance_bar(
    impurity_table, value_column="importance",
    title=f"{display_name}: Impurity-Based Feature Importance",
    subtitle="Average decrease in node impurity (Gini) attributable to each feature",
)
plt.show()
""")
    code(f"""perm_table = result_{model_key}.feature_importance["permutation"]
perm_table.head(15)
""")
    code(f"""fig = model_utils.plot_importance_bar(
    perm_table, value_column="importance_mean",
    title=f"{display_name}: Permutation Importance (raw features)",
    subtitle="Drop in ROC-AUC when a raw feature's values are randomly shuffled",
)
plt.show()
""")
    code("""comparison = impurity_table.merge(
    perm_table, on="feature", how="outer"
).sort_values("importance_mean", ascending=False)
comparison[["feature", "importance", "importance_mean"]].rename(
    columns={"importance": "impurity_importance", "importance_mean": "permutation_importance"}
).head(10)
""")
    md("""**Comparing the two methods:** features that rank highly under BOTH
impurity and permutation importance are the most trustworthy candidates
for "truly predictive" — a feature high on impurity but low on
permutation importance is a signal of the impurity-bias caveat above
(often a high-cardinality numeric column that gets many candidate splits
without necessarily carrying unique predictive signal).""")


add_model_section(
    "random_forest", rf_intro,
    validation_curve_param="classifier__n_estimators",
    validation_curve_range="[50, 100, 200, 300, 400, 500, 600]",
    importance_cells_fn=rf_importance_cells,
)

# ---------------------------------------------------------------------------
# XGBoost section
# ---------------------------------------------------------------------------

xgb_intro = """## 6. XGBoost

**Why this algorithm:** gradient-boosted trees trained sequentially,
each new tree correcting the previous ensemble's residual errors, which
typically yields the strongest raw discriminative performance among
tree-based methods on structured/tabular data — making it the candidate
production model.

**Advantages:** state-of-the-art tabular performance, built-in
regularization (L1/L2, `min_child_weight`, `gamma`) that helps control
overfitting, native handling of missing values, fast histogram-based
split finding, multiple importance views (gain/weight/cover).

**Disadvantages:** the largest hyperparameter surface of the three
(highest tuning cost), least directly interpretable without an
additional layer (SHAP, Phase 4), can overfit small/noisy datasets if
not regularized carefully.

**Business tradeoff:** typically the best accuracy/ROC-AUC of the
three, at the cost of being a "black box" without supplementary
explainability tooling and a heavier tuning/maintenance burden.

**Computational complexity:** O(n_trees × n_samples × n_features ×
log(n_samples)) for training (similar order to Random Forest, but
sequential across trees rather than independent/parallel, though each
tree's construction is itself parallelized internally).

**Hyperparameter search space** (`config.XGBOOST_PARAM_DISTRIBUTIONS`,
extended at runtime with a data-driven `scale_pos_weight` candidate —
see `model_utils._xgboost_param_distributions` — and searched via
`RandomizedSearchCV` with `n_iter=30`):
- `n_estimators`, `max_depth`, `learning_rate` — the core boosting
  triad: how many trees, how complex each one, how much each one
  contributes.
- `subsample`, `colsample_bytree` — row/column subsampling per tree,
  which reduces overfitting and correlation between trees (similar
  spirit to Random Forest's `max_features`).
- `min_child_weight`, `gamma` — minimum leaf weight / minimum split
  loss reduction required to make a split, both directly limiting tree
  complexity.
- `reg_alpha`, `reg_lambda` — L1/L2 regularization on leaf weights.
- `scale_pos_weight` ∈ {1.0, `negative_count/positive_count`} — tests
  whether explicitly reweighting the minority (default) class improves
  ranking quality, mirroring Random Forest's `class_weight` options.
"""


def xgb_importance_cells(model_key, display_name):
    code(f"""gwc_table = result_{model_key}.feature_importance["gain_weight_cover"]
gwc_table.head(15)
""")
    md("""**The three XGBoost-native importance types:**
- **Gain** — average improvement in the loss function contributed by a
  feature; generally the most reliable single measure of "how much does
  this feature actually help predictions."
- **Weight** — how many times a feature is used to split across all
  trees; can overstate importance for features with many possible split
  points (the same caveat as Random Forest's impurity importance).
- **Cover** — average number of samples affected by splits on a
  feature; a proxy for how broadly a feature's splits apply.""")
    code(f"""fig = model_utils.plot_importance_bar(
    gwc_table, value_column="gain",
    title=f"{display_name}: Gain Importance",
    subtitle="Average improvement in the loss function contributed by each feature",
)
plt.show()
""")
    code(f"""fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
top_weight = gwc_table.nlargest(10, "weight").sort_values("weight")
top_cover = gwc_table.nlargest(10, "cover").sort_values("cover")
axes[0].barh(top_weight["feature"], top_weight["weight"], color="#2E86AB")
axes[0].set_title("Weight Importance (Top 10)", fontsize=11, fontweight="bold", loc="left")
axes[1].barh(top_cover["feature"], top_cover["cover"], color="#C0392B")
axes[1].set_title("Cover Importance (Top 10)", fontsize=11, fontweight="bold", loc="left")
fig.suptitle(f"{display_name}: Weight vs. Cover Importance", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()
""")
    code(f"""perm_table = result_{model_key}.feature_importance["permutation"]
perm_table.head(15)
""")
    code(f"""fig = model_utils.plot_importance_bar(
    perm_table, value_column="importance_mean",
    title=f"{display_name}: Permutation Importance (raw features)",
    subtitle="Drop in ROC-AUC when a raw feature's values are randomly shuffled",
)
plt.show()
""")
    md("""**Comparing methods:** gain and permutation importance tend to agree
on the truly predictive features; weight and cover are more useful for
understanding *how* the model uses a feature (frequently in small,
targeted splits vs. broad, high-impact ones) than *whether* it matters.""")


add_model_section(
    "xgboost", xgb_intro,
    validation_curve_param="classifier__max_depth",
    validation_curve_range="[2, 3, 4, 5, 6, 8, 10]",
    importance_cells_fn=xgb_importance_cells,
)

# =============================================================================
# 7. MODEL COMPARISON
# =============================================================================
md("""## 7. Executive Model Comparison

All three models' test-set metrics, cross-validation summary, timing,
and qualitative deployment attributes in one table, ranked by test
ROC-AUC (`config.CV_SCORING`, the metric optimized during tuning).""")

code("""all_results = [result_logistic_regression, result_random_forest, result_xgboost]
comparison_table = model_utils.build_model_comparison_table(all_results)
comparison_table
""")

md("""### Reading the comparison table

- **rank** — 1 = highest test ROC-AUC.
- **roc_auc, f1_score, recall, precision, specificity, balanced_accuracy,
  accuracy** — see Section 3 for what each means in a lending context;
  no single column should be read in isolation.
- **cv_roc_auc_mean / cv_roc_auc_std** — how stable each model's
  ranking performance was across the 5 cross-validation folds; a high
  std relative to the mean signals a less reliable estimate.
- **training_time_sec** — wall time to refit the tuned pipeline once on
  the full training set (the relevant number for periodic retraining in
  production, not the one-time hyperparameter search itself).
- **prediction_time_ms_per_1000** — inference latency, relevant for a
  real-time or batch-scoring decision-support application.
- **memory_usage_kb** — actual serialized (`joblib`) file size on disk
  for each fitted pipeline — a real measurement, not an estimate.
- **interpretability / complexity / deployment_readiness** — qualitative
  attributes fixed in `model_utils.MODEL_QUALITATIVE_ATTRIBUTES` (see
  that module for the reasoning behind each), included because a
  production model choice is never based on ROC-AUC alone.
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

metrics_to_plot = ["roc_auc", "f1_score", "recall", "precision", "balanced_accuracy"]
plot_df = comparison_table.set_index("model")[metrics_to_plot]
plot_df.plot(kind="bar", ax=axes[0], color=["#2E86AB", "#C0392B", "#F39C12", "#27AE60", "#8E44AD"])
axes[0].set_title("Test-Set Metrics by Model", fontsize=12, fontweight="bold", loc="left")
axes[0].set_ylabel("Score")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].tick_params(axis="x", rotation=15)

timing_df = comparison_table.set_index("model")[["training_time_sec", "prediction_time_ms_per_1000"]]
timing_df.plot(kind="bar", ax=axes[1], color=["#2E86AB", "#C0392B"], secondary_y="prediction_time_ms_per_1000")
axes[1].set_title("Training Time vs. Prediction Latency", fontsize=12, fontweight="bold", loc="left")
axes[1].set_ylabel("Training time (sec)")
axes[1].tick_params(axis="x", rotation=15)

fig.suptitle("Model Comparison: Performance and Computational Cost", fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
plt.show()
""")

md("""### Recommended Production Model

*(Fill in the specific numbers once this notebook has been run against
the real Indiana extract — the reasoning framework below stays fixed
regardless of the exact scores.)*

Recommendation logic:
1. If XGBoost's test ROC-AUC materially exceeds Random Forest's and
   Logistic Regression's (and its cross-validation std is not
   dramatically wider, i.e. its advantage is stable), **XGBoost is the
   recommended production scorer** — consistent with it typically
   offering the strongest raw discrimination on structured data, with
   Phase 4's SHAP analysis addressing its interpretability gap.
2. If the gap between XGBoost and Random Forest is small while
   XGBoost's tuning/maintenance surface is meaningfully larger, prefer
   **Random Forest** for a better complexity-to-performance ratio.
3. **Logistic Regression is retained regardless of rank** as the
   transparent reference model — useful for explaining declines to
   individual applicants and for regulatory review — rather than being
   discarded once a stronger model is found.

This ranking directly answers **Research Question 2** ("do LendingClub
grades appear predictive?" — see each model's feature-importance table,
where `grade` should appear as a top predictor if so) and contributes to
**Research Question 1** ("which borrower characteristics appear
associated with default?").
""")

# =============================================================================
# 8. MODEL ROBUSTNESS
# =============================================================================
md("""## 8. Model Robustness: Overfitting, Underfitting, Bias, Variance

Combining each model's train/validation/test ROC-AUC (Section 4-6's
"Overfitting Check" tables) with its learning curve to assess
generalization.""")

code("""robustness_rows = []
for key, result in zip(["logistic_regression", "random_forest", "xgboost"], all_results):
    train_auc = result.train_metrics["roc_auc"]
    val_auc = result.val_metrics["roc_auc"]
    test_auc = result.test_metrics["roc_auc"]
    robustness_rows.append({
        "model": model_utils.MODEL_DISPLAY_NAMES[key],
        "train_roc_auc": train_auc,
        "val_roc_auc": val_auc,
        "test_roc_auc": test_auc,
        "train_minus_test_gap": train_auc - test_auc,
    })

robustness_table = pd.DataFrame(robustness_rows)
robustness_table
""")

md("""**How to read the gap column:**
- A **large positive gap** (train ROC-AUC well above test ROC-AUC)
  indicates **high variance / overfitting** — the model has partly
  memorized the training data rather than learning generalizable
  patterns. Tree ensembles (Random Forest, XGBoost) are more prone to
  this than Logistic Regression given their greater flexibility, which
  is exactly why their hyperparameter searches include explicit
  regularization controls (`max_depth`, `min_samples_leaf`, `reg_alpha`/
  `reg_lambda`, `gamma`).
- A **small gap but uniformly low scores** across train/val/test
  indicates **high bias / underfitting** — the model (or its features)
  is too constrained to capture the available signal; more data would
  not help much, but more/better features or a more flexible model
  class would.
- A **small gap with strong scores** across all three splits is the
  target: the model generalizes well.

**Would additional data likely help?** Referring back to each model's
learning curve (Sections 4-6): if the cross-validation line is still
rising and has not plateaued by the full training-set size, more data
would likely continue improving that model. If it has already
plateaued, additional data of the *same kind* offers diminishing
returns — the more valuable investment would be additional or better
*features* (e.g. payment-history detail, macroeconomic indicators for
the loan's origination period) rather than additional rows.
""")

# =============================================================================
# 9. BUSINESS INTERPRETATION (CROSS-MODEL)
# =============================================================================
md("""## 9. Business Interpretation Across All Models

**What did we learn?** Across Logistic Regression's coefficients, Random
Forest's impurity/permutation importances, and XGBoost's gain/weight/
cover/permutation importances (Sections 4-6), identify the borrower
characteristics that consistently rank as top predictors across ALL
THREE model types — features that only one model rates highly are less
trustworthy than those confirmed across independent modeling approaches.

**Which variables matter most?** *(populate once run against the real
extract)* — cross-reference `result_logistic_regression.feature_importance`,
`result_random_forest.feature_importance`, and
`result_xgboost.feature_importance` for features appearing in the top 10
of at least two of the three models.

**Which borrower characteristics increase default risk?** For Logistic
Regression specifically, any feature with an odds ratio > 1 in the
coefficient table directly answers this — e.g. higher interest rate,
higher DTI, and lower loan grade are the typical candidates worth
checking against this dataset's actual coefficients.

**How should Lending Club change lending decisions?**
1. Use the recommended production model's (Section 7) probability output
   at the **recommended threshold** (Section 4-6's Threshold Analysis,
   not 0.50) as a decision-support score, not a fully automated
   accept/decline switch.
2. Prioritize collecting/verifying the top-ranked predictive features at
   application time, since data quality on these fields has an outsized
   effect on model reliability.
3. Retain Logistic Regression's coefficient table as the plain-language
   explanation attached to any automated decision, satisfying adverse-
   action-notice-style transparency requirements even when XGBoost is
   the production scorer.

**Which research questions were answered?**
- **RQ1** (borrower characteristics associated with default) and **RQ2**
  (are LendingClub grades predictive) — directly, via every model's
  feature importance.
- **RQ3** (variables related to higher interest rates) — indirectly
  informed by which features correlate most with `int_rate` in the
  Logistic Regression coefficients and tree-based importances.
- **RQ4** (does income relate to repayment success) and **RQ5** (does
  DTI influence default) — directly testable from each model's
  importance ranking for `annual_inc` and `dti`.
- **RQ6** (does employment length matter) — directly testable from
  `emp_length_years`'s importance ranking across all three models.
- **RQ7** (natural borrower groups before clustering) — not addressed by
  supervised models; explicitly deferred to Phase 4's clustering
  analysis (see Section 11).
""")

# =============================================================================
# 10. SAVE ARTIFACTS
# =============================================================================
md("""## 10. Serialized Artifacts

`src/train_models.py`'s `run_phase3_pipeline()` (which this notebook's
per-model training cells mirror) persists every artifact below via
`joblib`. Re-running the cell below confirms everything this notebook
produced is also available for Phase 4/5 to load directly, without
re-running training.""")

code("""for key, result in zip(["logistic_regression", "random_forest", "xgboost"], all_results):
    model_path = {
        "logistic_regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
        "random_forest": config.RANDOM_FOREST_MODEL_PATH,
        "xgboost": config.XGBOOST_MODEL_PATH,
    }[key]
    utils.save_object(result.best_estimator, model_path)

utils.save_object({k: r.test_metrics for k, r in zip(["logistic_regression", "random_forest", "xgboost"], all_results)}, config.EVALUATION_METRICS_PATH)
utils.save_object({k: r.cv_fold_results for k, r in zip(["logistic_regression", "random_forest", "xgboost"], all_results)}, config.CV_RESULTS_PATH)
utils.save_object({k: r.feature_importance for k, r in zip(["logistic_regression", "random_forest", "xgboost"], all_results)}, config.FEATURE_IMPORTANCE_PATH)
utils.save_object({k: r.y_proba_test for k, r in zip(["logistic_regression", "random_forest", "xgboost"], all_results)}, config.PROBABILITY_PREDICTIONS_PATH)
utils.save_object({k: r.threshold_table for k, r in zip(["logistic_regression", "random_forest", "xgboost"], all_results)}, config.THRESHOLD_ANALYSIS_PATH)
comparison_table.to_csv(config.MODEL_COMPARISON_TABLE_PATH, index=False)

print("Saved artifacts:")
print(f"  Models:              {config.MODELS_DIR}")
print(f"  Evaluation metrics:  {config.EVALUATION_METRICS_PATH}")
print(f"  CV results:          {config.CV_RESULTS_PATH}")
print(f"  Feature importance:  {config.FEATURE_IMPORTANCE_PATH}")
print(f"  Probability preds:   {config.PROBABILITY_PREDICTIONS_PATH}")
print(f"  Threshold analysis:  {config.THRESHOLD_ANALYSIS_PATH}")
print(f"  Comparison table:    {config.MODEL_COMPARISON_TABLE_PATH}")
""")

md("""*(Equivalently, `python -m src.train_models` runs Phase 1 and Phase 3
end-to-end from the command line and produces the same artifacts — see
`src/train_models.py`'s `run_phase3_pipeline()`.)*""")

# =============================================================================
# 11. PHASE TRANSITION
# =============================================================================
md("""## 11. Preparing for Phase 4 (SHAP Explanations + Clustering)

**This notebook does NOT implement SHAP or clustering** — both are
explicitly Phase 4 scope. This section prepares the hand-off.

### Which model will be used for SHAP explanations?
The model recommended as the production scorer in Section 7 (typically
XGBoost, given its usual discrimination advantage) — SHAP's
TreeExplainer is both exact and fast for gradient-boosted trees, making
it the natural pairing. Logistic Regression's coefficients already
provide global interpretability without SHAP; Random Forest could also
use SHAP's TreeExplainer if it becomes the recommended model instead.

### Which model will serve as the production model?
Per Section 7's ranked comparison table and recommendation — subject to
re-confirmation once run against the real Indiana extract rather than
the current synthetic fixture.

### Which borrower characteristics appear most influential?
The features consistently ranked highly across Logistic Regression
coefficients, Random Forest impurity/permutation importance, and
XGBoost gain/weight/cover/permutation importance (Section 9) — these
are the strongest candidates for SHAP to explain at the individual-loan
level in Phase 4, since global importance agreement across independent
methods increases confidence that Phase 4's per-loan explanations will
be stable and trustworthy rather than artifacts of one model's
idiosyncrasies.

### How will clustering complement the supervised models?
Research Question 7 ("can natural borrower groups already be observed
before clustering?") is explicitly NOT answered by any supervised model
here — Logistic Regression, Random Forest, and XGBoost all predict a
single binary outcome (`default_flag`) for a given borrower, but none of
them group borrowers into cohorts. Phase 4's clustering (e.g. K-Means or
hierarchical clustering on borrower features) will identify unsupervised
segments — for example, "high-income / low-utilization / long tenure"
vs. "high-DTI / high-utilization / short tenure" borrower archetypes —
that the supervised models' probability scores and feature importances
can then be overlaid onto, showing *which segments* carry the highest
predicted risk and *why*, giving Lending Club both a risk **score**
(from this notebook) and a risk **narrative** (from Phase 4) for each
borrower segment.

### Explicitly deferred to later phases
- SHAP value computation and force/summary/dependence plots (Phase 4).
- Borrower clustering / segmentation (Phase 4).
- The Streamlit decision-support application (Phase 5).
- Final deployment and documentation (Phase 6).
""")

nb["cells"] = cells

with open("notebooks/MGMT590_LendingClub_Modeling_Phase3.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Phase 3 notebook written with {len(cells)} cells.")
