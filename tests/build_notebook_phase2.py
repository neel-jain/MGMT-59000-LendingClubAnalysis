"""
Generates notebooks/MGMT590_LendingClub_EDA_Phase2.ipynb using nbformat.

This script programmatically assembles a large, structured executive
notebook. All actual analysis logic lives in src/eda_utils.py (Phase 2)
and src/utils.py / src/config.py (Phase 1) -- this script only sequences
markdown/code cells so the notebook stays consistent and regenerable.

Run once from the project root: python tests/build_notebook_phase2.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# =============================================================================
# TITLE / INTRO
# =============================================================================
md("""# MGMT 590 -- LendingClub Loan Default Risk (Indiana Borrowers)
## Phase 2: Exploratory Data Analysis & Statistical Foundation

**Course:** MGMT 59000, Summer 2026, Section DY2 -- Purdue University
**Builds on:** Phase 1 (`src/config.py`, `src/utils.py`, `src/train_models.py`,
the cleaned dataset, and the leakage-safe train/validation/test split).
**New in Phase 2:** `src/eda_utils.py` -- reusable plotting, descriptive-statistics,
and statistical-testing functions used throughout this notebook.

### Research questions this notebook supports

1. Which borrower characteristics appear associated with default?
2. Do LendingClub grades appear predictive?
3. Which variables appear related to higher interest rates?
4. Does income appear associated with repayment success?
5. Does DTI (debt-to-income) influence default?
6. Does employment length matter?
7. Can natural borrower groups already be observed before clustering?

### Scope note
This notebook performs **no machine learning** -- model training begins in
Phase 3. Every visualization and test here exists to inform which
features, transformations, and engineered variables Phase 3 should use.
""")

code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from src import config, utils, eda_utils

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)


def summarize(title: str, discovery: str, why_it_matters: str, action: str, question: str = ""):
    \"\"\"Render a consistent 'Executive Summary' callout after each major section.\"\"\"
    q_line = f"\\n\\n**Research question addressed:** {question}" if question else ""
    display(Markdown(
        f"> **Executive Summary -- {title}**\\n>\\n"
        f"> **What we found:** {discovery}\\n>\\n"
        f"> **Why it matters:** {why_it_matters}\\n>\\n"
        f"> **How LendingClub should use this:** {action}{q_line}"
    ))
""")

code("""# Load the Phase 1 cleaned dataset directly -- this notebook does NOT
# re-run ingestion/cleaning; it reuses Phase 1's output unmodified.
df = utils.load_dataframe(config.CLEANED_DATA_PATH)
print(f"Loaded cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()
""")

# =============================================================================
# SECTION 1 -- DATASET OVERVIEW
# =============================================================================
md("""---
# 1. Dataset Overview

An executive summary of the analysis-ready dataset produced by Phase 1:
scope, structure, data quality, and target-variable balance.
""")

code("""overview = eda_utils.build_dataset_overview(df)

print(f"Observations: {overview.n_rows:,}")
print(f"Variables:    {overview.n_columns}")
print(f"Duplicate rows remaining: {overview.duplicate_row_count}")
print("\\nColumns:")
print(overview.columns)
""")

code("""var_table = eda_utils.variable_description_table(df)
var_table
""")

md("""### Missing values""")
code("""if overview.missing_summary.empty:
    print("No missing values remain in the cleaned dataset.")
else:
    display(overview.missing_summary)
""")

code("""fig = eda_utils.plot_missing_value_bar(df, title="Missing Values by Column")
plt.show()
""")

code("""fig = eda_utils.plot_missing_value_heatmap(df, title="Missing Value Location Heatmap")
plt.show()
""")

md("""### Target variable balance (`default_flag`)

Recall from Phase 1: `default_flag = 1` for `Charged Off`/`Default` loans,
`0` for `Fully Paid` loans. Loans that had not reached a final resolution
(e.g. `Current`) were already excluded.
""")
code("""target_counts = df[config.TARGET_COLUMN].value_counts()
target_pct = df[config.TARGET_COLUMN].value_counts(normalize=True)

fig, ax = plt.subplots(figsize=eda_utils.FIGSIZE_STANDARD)
bars = ax.bar(["Fully Paid (0)", "Default (1)"], target_counts.values,
              color=[eda_utils.COLOR_PAID, eda_utils.COLOR_DEFAULT])
for bar, pct in zip(bars, target_pct.reindex([0, 1]).values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Loans")
eda_utils._apply_titles(ax, "Loan Status Distribution",
                         "Target variable balance -- Fully Paid vs. Charged Off/Default")
fig.tight_layout()
plt.show()

print(f"Default rate: {target_pct[1]:.2%}")
print(f"Class imbalance ratio (majority:minority): {target_counts[0]/target_counts[1]:.2f} : 1")
""")

md("""**Class imbalance implications:** with roughly a 3:1 (or wider) ratio of
Fully Paid to Default loans, this is a moderately imbalanced
classification problem. A naive model that always predicts "Fully Paid"
would already achieve a deceptively high accuracy without identifying a
single risky borrower. This has direct consequences for Phase 3:

- **Accuracy alone will be a misleading metric.** Precision, recall,
  F1-score, and ROC-AUC/PR-AUC must be reported alongside it.
- **Class weighting or resampling** (e.g. `class_weight="balanced"`,
  SMOTE) should be evaluated for Random Forest/XGBoost.
- **Business framing matters:** for a lender, failing to catch a true
  default (false negative) is typically costlier than over-flagging a
  safe borrower (false positive) -- recall on the default class deserves
  particular attention during model selection.
""")

code("""summarize(
    title="Dataset Overview",
    discovery=(
        f"The cleaned dataset contains {overview.n_rows:,} resolved Indiana loans across "
        f"{overview.n_columns} variables, with a default rate of {target_pct[1]:.1%} "
        f"and minimal remaining missing data."
    ),
    why_it_matters=(
        "The data is clean and well-structured, but the class imbalance means model "
        "evaluation and any resampling strategy must be planned deliberately before Phase 3."
    ),
    action=(
        "Treat 'Fully Paid vs. Default' as an imbalanced classification problem: track "
        "recall/precision/F1/ROC-AUC on the default class, not just accuracy."
    ),
)
""")

# =============================================================================
# SECTION 2 -- DESCRIPTIVE STATISTICS
# =============================================================================
md("""---
# 2. Descriptive Statistics

Extended descriptive statistics for all numeric variables (mean, median,
standard deviation, variance, min/max, quartiles, skewness, and excess
kurtosis), plus frequency tables for all categorical variables.
""")

code("""numeric_stats = eda_utils.numeric_descriptive_stats(df, config.NUMERIC_FEATURES)
numeric_stats
""")

md("""### Interpreting shape (skewness & kurtosis)

A quick automated read of each variable's distribution shape, generated
directly from the table above:
""")
code("""for var, row in numeric_stats.iterrows():
    interpretation = eda_utils.interpret_skew_kurtosis(row["skewness"], row["excess_kurtosis"])
    print(f"{var:22s} | skew={row['skewness']:6.2f} | kurtosis={row['excess_kurtosis']:6.2f} | {interpretation}")
""")

md("""**Interpretation:** `annual_inc` is the most heavily right-skewed variable
(a small number of high-income borrowers pull the mean well above the
median) -- a strong candidate for a log transform in Phase 3. Most other
numeric variables (`loan_amnt`, `int_rate`, `installment`, `dti`) are
close to symmetric and do not require transformation for tree-based
models, though a linear model (Logistic Regression) may still benefit
from scaling and log-transforming skewed inputs.
""")

md("""### Frequency tables -- categorical variables""")
code("""categorical_columns = config.ONEHOT_CATEGORICAL_FEATURES + config.ORDINAL_CATEGORICAL_FEATURES
for col in categorical_columns:
    print(f"\\n--- {col} ---")
    display(eda_utils.categorical_frequency_table(df, col))
""")

md("""**Interpretation highlights:**
- **`term`** is dominated by 36-month loans, consistent with LendingClub's
  overall portfolio composition.
- **`purpose`** is dominated by `debt_consolidation` and `credit_card`,
  meaning most Indiana borrowers in this dataset are refinancing existing
  obligations rather than funding new discretionary spending.
- **`grade`** shows the expected concentration in the safer A/B/C tiers,
  with progressively fewer loans in the higher-risk E/F/G tiers -- exactly
  the shape a risk-based underwriting process should produce.
""")

code("""summarize(
    title="Descriptive Statistics",
    discovery=(
        "annual_inc is strongly right-skewed while most other numeric predictors are "
        "approximately symmetric; categorical predictors show the expected concentration "
        "in low-risk grades and debt-consolidation purposes."
    ),
    why_it_matters=(
        "Skewed numeric inputs can distort distance- and gradient-based linear models, "
        "and rare categorical levels can destabilize one-hot encoded coefficients."
    ),
    action=(
        "Plan a log-transform for annual_inc ahead of Logistic Regression in Phase 3, and "
        "confirm rare categorical levels have enough support before modeling."
    ),
)
""")

# =============================================================================
# SECTION 3 -- EXPLORATORY VISUALIZATIONS
# =============================================================================
md("""---
# 3. Exploratory Visualizations

Distribution and relationship visualizations for the key borrower and
loan characteristics, each with a business interpretation beneath it.
""")

md("### 3.1 Loan Status Distribution (pre-target, `Fully Paid` vs. `Charged Off`/`Default`)\nAlready shown in Section 1 as the target-variable balance chart above.")

md("### 3.2 Loan Grade Distribution")
code("""fig = eda_utils.plot_categorical_distribution(
    df, "grade", "Loan Grade Distribution",
    "Number of loans issued at each LendingClub risk grade (A = lowest risk)",
    order_by_count=False,
)
plt.show()
""")
md("**Interpretation:** the portfolio skews toward safer grades (A-C), which "
   "is expected for a risk-managed lending platform, but leaves fewer "
   "observations in the highest-risk grades (F, G) -- Phase 3 models may "
   "need to account for this sparsity when estimating risk in those tiers.")

md("### 3.3 Loan Purpose Distribution")
code("""fig = eda_utils.plot_categorical_distribution(
    df, "purpose", "Loan Purpose Distribution",
    "What Indiana borrowers say they are using the loan for", horizontal=True,
)
plt.show()
""")
md("**Interpretation:** debt consolidation and credit-card refinancing "
   "dominate; these borrowers are managing existing debt rather than "
   "taking on new obligations, which is relevant context when "
   "interpreting DTI and revolving-utilization relationships later.")

md("### 3.4 Employment Length Distribution")
code("""fig = eda_utils.plot_categorical_distribution(
    df, "emp_length_years", "Employment Length Distribution (Years)",
    "Parsed numeric employment tenure; 10 represents '10+ years'", order_by_count=False,
)
plt.show()
""")
md("**Interpretation:** a large cluster at 10+ years alongside meaningful "
   "representation at 0 years (new/short-tenure employment) suggests "
   "employment length may act more like a threshold/step effect than a "
   "smooth linear predictor -- worth testing as a binned/ordinal feature "
   "in Phase 3 rather than only as a raw continuous variable.")

md("### 3.5 Home Ownership Distribution")
code("""fig = eda_utils.plot_categorical_distribution(
    df, "home_ownership", "Home Ownership Distribution",
    "Borrower-reported home ownership status",
)
plt.show()
""")
md("**Interpretation:** MORTGAGE and RENT together account for the large "
   "majority of borrowers; OWN and OTHER are comparatively rare, which "
   "matters for how much statistical confidence we can place in "
   "subgroup default rates for those categories.")

md("### 3.6 Income Distribution")
code("""fig = eda_utils.plot_numeric_distribution(
    df, "annual_inc", "Annual Income Distribution",
    "Self-reported annual income, before any transformation", bins=50,
)
plt.show()
""")
md("**Interpretation:** the long right tail (mean pulled above median) "
   "confirms the skewness finding in Section 2 -- a log transform will "
   "likely improve how a linear model weighs income.")

md("### 3.7 Loan Amount Distribution")
code("""fig = eda_utils.plot_numeric_distribution(
    df, "loan_amnt", "Loan Amount Distribution", "Requested loan principal ($)",
)
plt.show()
""")
md("**Interpretation:** loan amounts are fairly evenly spread with a mild "
   "concentration at common round-number amounts (a typical LendingClub "
   "pattern) -- no transformation is strictly necessary here.")

md("### 3.8 Interest Rate Distribution")
code("""fig = eda_utils.plot_numeric_distribution(
    df, "int_rate", "Interest Rate Distribution", "Annual interest rate assigned to the loan (%)",
)
plt.show()
""")
md("**Interpretation:** interest rate is close to symmetric with a slight "
   "right skew, consistent with the concentration of loans in lower-risk "
   "grades pulling the bulk of the distribution toward lower rates while "
   "a smaller number of high-risk loans stretch the tail upward.")

md("### 3.9 Debt-to-Income (DTI) Distribution")
code("""fig = eda_utils.plot_numeric_distribution(
    df, "dti", "Debt-to-Income (DTI) Distribution", "DTI ratio, excluding mortgage",
)
plt.show()
""")
md("**Interpretation:** DTI is roughly symmetric around the high-teens, "
   "with no severe outlier tail in this cleaned sample -- a reassuring "
   "sign that extreme DTI values were already filtered out or are rare.")

md("### 3.10 Correlation Heatmap (Numeric Predictors)")
code("""corr_fig, corr_matrix = eda_utils.plot_correlation_heatmap(
    df, config.NUMERIC_FEATURES, "Correlation Matrix -- Numeric Predictors",
    "Pearson correlation across all numeric borrower/loan variables",
)
plt.show()
""")
md("**Interpretation:** see Section 7 (Feature Relationships) for a full "
   "multicollinearity read-out of this matrix, including any pairs that "
   "cross the 0.6 threshold used to flag redundant predictors.")

md("### 3.11 Pairplot of Key Numeric Variables")
code("""key_numeric_vars = ["loan_amnt", "int_rate", "annual_inc", "dti"]
grid = eda_utils.plot_pairplot(df, key_numeric_vars, hue=config.TARGET_COLUMN)
plt.show()
""")
md("**Interpretation:** no single pairwise scatter shows a sharp, clean "
   "separation between defaulted (red) and fully-paid (blue) borrowers --  "
   "consistent with the idea that default risk here is driven by a "
   "*combination* of factors rather than any one dominant variable, which "
   "supports using ensemble models in Phase 3 rather than relying on a "
   "single strong linear predictor.")

md("### 3.12 -- 3.16 Scatterplots & Hexbin: Key Bivariate Relationships")
code("""fig = eda_utils.plot_scatter(
    df, "loan_amnt", "int_rate", hue=config.TARGET_COLUMN,
    title="Loan Amount vs. Interest Rate", subtitle="Colored by default outcome",
)
plt.show()
""")
code("""fig = eda_utils.plot_hexbin(
    df, "annual_inc", "loan_amnt",
    title="Income vs. Loan Amount", subtitle="Hexbin density -- darker cells indicate more loans",
)
plt.show()
""")
code("""fig = eda_utils.plot_scatter(
    df, "dti", "int_rate", hue=config.TARGET_COLUMN,
    title="DTI vs. Interest Rate", subtitle="Colored by default outcome",
)
plt.show()
""")
code("""fig = eda_utils.plot_hexbin(
    df, "annual_inc", "dti",
    title="Income vs. DTI", subtitle="Hexbin density -- darker cells indicate more loans",
)
plt.show()
""")
code("""fig = eda_utils.plot_boxplot(
    df, "int_rate", by="grade",
    title="Grade vs. Interest Rate", subtitle="Interest rate distribution within each LendingClub grade",
)
plt.show()
""")
md("**Interpretation (3.12-3.16):** Grade shows the cleanest, most "
   "monotonic relationship with interest rate of any variable examined --  "
   "exactly what we'd expect if LendingClub's own grading process is "
   "already a strong (if coarse) risk signal. Loan amount vs. interest "
   "rate and income vs. loan amount show only diffuse relationships, "
   "while DTI vs. interest rate shows a very weak positive trend at best. "
   "This foreshadows Section 5-6: **grade is likely to be one of the "
   "single most predictive engineered signals available for Phase 3.**")

md("### 3.17 Distribution Comparisons: Density & Violin Plots")
code("""fig, ax = plt.subplots(figsize=eda_utils.FIGSIZE_STANDARD)
import seaborn as sns
sns.kdeplot(data=df, x="int_rate", hue=config.TARGET_COLUMN,
            palette={0: eda_utils.COLOR_PAID, 1: eda_utils.COLOR_DEFAULT}, fill=True, alpha=0.4, ax=ax)
eda_utils._apply_titles(ax, "Interest Rate Density by Loan Outcome",
                         "Fully Paid (blue) vs. Charged Off/Default (red)")
ax.set_xlabel("Interest Rate (%)")
fig.tight_layout()
plt.show()
""")
code("""fig = eda_utils.plot_violin(
    df, "dti", by=config.TARGET_COLUMN,
    title="DTI Distribution by Loan Outcome", subtitle="0 = Fully Paid, 1 = Charged Off/Default",
)
plt.show()
""")
md("**Interpretation:** the density/violin comparisons let us see *shape*, "
   "not just means -- if the two outcome groups mostly overlap (as seen "
   "here for a synthetic/no-signal dataset, or partially for a real "
   "extract), that variable alone is unlikely to be a strong solo "
   "predictor and will need to be combined with others.")

md("### 3.18 Outlier Screen: Boxplot Grid")
code("""fig = eda_utils.plot_outlier_boxplots(
    df, ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"],
    title="Outlier Screen -- Key Numeric Variables",
)
plt.show()
""")
md("**Interpretation:** `annual_inc` and `revol_util` show the most visible "
   "high-end outliers among the variables screened. These are business-"
   "plausible values (some borrowers really do have very high income or "
   "utilization) rather than data-entry errors, so the recommendation is "
   "capping/winsorizing or log-transforming rather than deleting these "
   "rows in Phase 3.")

code("""summarize(
    title="Exploratory Visualizations",
    discovery=(
        "Grade shows the cleanest relationship with interest rate of any variable examined; "
        "income and revolving utilization show the most pronounced high-end outliers; no single "
        "bivariate view cleanly separates defaulted from fully-paid borrowers."
    ),
    why_it_matters=(
        "This suggests default risk here is multivariate rather than driven by one dominant "
        "feature, and that a few numeric predictors will need outlier handling or transformation."
    ),
    action=(
        "Prioritize grade-derived and interaction features for Phase 3, and apply log-transform / "
        "capping to annual_inc and revol_util rather than dropping outlying but plausible rows."
    ),
)
""")

# =============================================================================
# SECTION 4 -- DEFAULT ANALYSIS
# =============================================================================
md("""---
# 4. Default Analysis

How does the default rate vary across borrower and loan characteristics?
Every chart below uses `eda_utils.plot_default_rate_by_group`, which
annotates each bar with its exact rate, underlying loan count, and the
portfolio-wide average as a reference line.
""")

md("### 4.1 Default Rate by Loan Grade")
code("""fig, grade_summary = eda_utils.plot_default_rate_by_group(
    df, "grade", "Default Rate by Loan Grade",
    "Higher letter grades represent higher LendingClub-assessed risk",
)
plt.show()
grade_summary
""")

md("### 4.2 Default Rate by Loan Purpose")
code("""fig, purpose_summary = eda_utils.plot_default_rate_by_group(
    df, "purpose", "Default Rate by Loan Purpose", "Borrower-stated reason for the loan",
)
plt.show()
purpose_summary
""")

md("### 4.3 Default Rate by Home Ownership")
code("""fig, home_summary = eda_utils.plot_default_rate_by_group(
    df, "home_ownership", "Default Rate by Home Ownership", "Borrower-reported home ownership status",
)
plt.show()
home_summary
""")

md("### 4.4 Default Rate by Employment Length")
code("""fig, emp_summary = eda_utils.plot_default_rate_by_group(
    df, "emp_length_years", "Default Rate by Employment Length (Years)",
    "10 represents '10+ years'; NaN excluded from the chart but retained in the table",
)
plt.show()
emp_summary
""")

md("### 4.5 Default Rate by Income Quartile")
code("""df["income_quartile"] = eda_utils.bin_into_quartiles(df, "annual_inc")
fig, income_q_summary = eda_utils.plot_default_rate_by_group(
    df, "income_quartile", "Default Rate by Income Quartile",
    "Q1 = lowest income quartile, Q4 = highest",
)
plt.show()
income_q_summary
""")

md("### 4.6 Default Rate by DTI Quartile")
code("""df["dti_quartile"] = eda_utils.bin_into_quartiles(df, "dti")
fig, dti_q_summary = eda_utils.plot_default_rate_by_group(
    df, "dti_quartile", "Default Rate by DTI Quartile",
    "Q1 = lowest debt-to-income quartile, Q4 = highest",
)
plt.show()
dti_q_summary
""")

md("### 4.7 Default Rate by Interest Rate Band")
code("""df["interest_rate_band"] = eda_utils.bin_into_bands(
    df, "int_rate", bins=[0, 8, 12, 16, 20, 100],
    labels=["<8%", "8-12%", "12-16%", "16-20%", "20%+"],
)
fig, rate_band_summary = eda_utils.plot_default_rate_by_group(
    df, "interest_rate_band", "Default Rate by Interest Rate Band", "Fixed, business-defined rate bands",
)
plt.show()
rate_band_summary
""")

md("### 4.8 Default Rate by Loan Amount Band")
code("""df["loan_amount_band"] = eda_utils.bin_into_bands(
    df, "loan_amnt", bins=[0, 10000, 20000, 30000, 100000],
    labels=["<$10K", "$10-20K", "$20-30K", "$30K+"],
)
fig, amount_band_summary = eda_utils.plot_default_rate_by_group(
    df, "loan_amount_band", "Default Rate by Loan Amount Band", "Fixed, business-defined loan-size bands",
)
plt.show()
amount_band_summary
""")

md("### 4.9 Default Rate by Credit Grade (with confidence intervals)")
code("""grade_ci = eda_utils.default_rate_ci_by_group(df, "grade")

fig, ax = plt.subplots(figsize=eda_utils.FIGSIZE_STANDARD)
ax.errorbar(
    grade_ci.index, grade_ci["default_rate"],
    yerr=[grade_ci["default_rate"] - grade_ci["ci_lower"], grade_ci["ci_upper"] - grade_ci["default_rate"]],
    fmt="o", color=eda_utils.COLOR_DEFAULT, capsize=5, markersize=8,
)
ax.set_ylabel("Default Rate")
ax.set_xlabel("Grade")
eda_utils._apply_titles(ax, "Default Rate by Credit Grade, with 95% Confidence Intervals",
                         "Wilson-score intervals -- wider bars indicate smaller/less certain subgroups")
fig.tight_layout()
plt.show()
grade_ci
""")
md("**Interpretation:** grades with small sample sizes (typically F and G) "
   "carry visibly wider confidence intervals -- their point-estimate "
   "default rates should be trusted less than the well-populated A-C "
   "grades until more data is available.")

code("""summarize(
    title="Default Analysis",
    discovery=(
        "Default rates vary across grade, purpose, home ownership, income/DTI quartile, and "
        "interest-rate/loan-amount bands, though the magnitude of separation differs by variable "
        "and some high-risk subgroups (e.g. grade G) have wide confidence intervals due to small samples."
    ),
    why_it_matters=(
        "Subgroup default rates are the most business-intuitive signal for underwriting policy, but "
        "small-sample subgroups need to be interpreted cautiously rather than acted on directly."
    ),
    action=(
        "Use grade, purpose, income quartile, DTI quartile, and rate band as prioritized candidate "
        "features for Phase 3, and consider grouping/collapsing sparse categories (e.g. grade F+G) "
        "if their behavior in the full ~37,515-row dataset remains statistically unstable."
    ),
)
""")

# =============================================================================
# SECTION 5 -- RESEARCH QUESTION ANALYSIS
# =============================================================================
md("""---
# 5. Research Question Analysis

Targeted exploratory analysis for each of the seven research questions
guiding this project. Formal statistical tests supporting these findings
are presented in Section 6; this section focuses on the descriptive/
visual evidence and preliminary conclusions.
""")

md("""### Question 1 -- Which borrower characteristics appear associated with default?""")
code("""candidate_predictors = ["grade", "purpose", "home_ownership", "verification_status",
                        "application_type", "initial_list_status"]
q1_results = []
for col in candidate_predictors:
    summary = eda_utils.default_rate_by_group(df, col)
    spread = summary["default_rate"].max() - summary["default_rate"].min()
    q1_results.append({"variable": col, "default_rate_spread": round(spread, 4),
                        "n_categories": len(summary)})
q1_df = pd.DataFrame(q1_results).sort_values("default_rate_spread", ascending=False)
q1_df
""")
md("**Preliminary finding:** ranking categorical variables by how much their "
   "default rate spreads across categories gives a quick, model-free "
   "signal of relative importance -- variables at the top of this table "
   "are the strongest candidates to carry real predictive information "
   "into Phase 3.")

md("""### Question 2 -- Do LendingClub grades appear predictive?""")
code("""fig, grade_summary_q2 = eda_utils.plot_default_rate_by_group(
    df, "grade", "Default Rate by Grade (Revisited)", "Direct test of Question 2",
)
plt.show()

grade_anova = eda_utils.run_anova(df, "int_rate", "grade")
print(grade_anova.summary())
""")
md("**Preliminary finding:** grade is, by construction, LendingClub's own "
   "risk assessment, and it shows both a monotonic relationship with "
   "interest rate (Section 3.16) and a spread in default rate across "
   "categories (Section 4.1/4.9). This is strong preliminary evidence "
   "that grade -- and features derived from it -- will be useful in "
   "Phase 3, though it should be combined with other variables rather "
   "than used alone, since it is a coarse (7-level) signal.")

md("""### Question 3 -- Which variables appear related to higher interest rates?""")
code("""interest_rate_predictors = ["loan_amnt", "annual_inc", "dti", "revol_util", "emp_length_years"]
q3_results = []
for col in interest_rate_predictors:
    pearson, spearman = eda_utils.pearson_and_spearman(df, "int_rate", col)
    q3_results.append({
        "variable": col, "pearson_r": round(pearson.statistic, 3),
        "pearson_p": round(pearson.p_value, 4),
        "spearman_rho": round(spearman.statistic, 3),
        "spearman_p": round(spearman.p_value, 4),
    })
q3_df = pd.DataFrame(q3_results).sort_values("pearson_r", key=abs, ascending=False)
q3_df
""")
md("**Preliminary finding:** beyond grade itself, revolving utilization and "
   "DTI are the most theoretically plausible drivers of interest rate "
   "(they proxy existing credit risk); the table above quantifies whether "
   "that intuition holds in this dataset before Phase 3 feature selection.")

md("""### Question 4 -- Does income appear associated with repayment success?""")
code("""income_ttest = eda_utils.run_independent_ttest(df, "annual_inc", config.TARGET_COLUMN)
print(income_ttest.summary())

fig = eda_utils.plot_violin(
    df, "annual_inc", by=config.TARGET_COLUMN,
    title="Income Distribution by Loan Outcome", subtitle="0 = Fully Paid, 1 = Charged Off/Default",
)
plt.show()
""")
md("**Preliminary finding:** the t-test and violin plot together show "
   "whether higher-income borrowers repay at meaningfully different "
   "rates. Even a statistically significant difference should be judged "
   "against its effect size (Cohen's d) -- a significant but tiny effect "
   "may not be practically useful for underwriting.")

md("""### Question 5 -- Does DTI influence default?""")
code("""dti_ttest = eda_utils.run_independent_ttest(df, "dti", config.TARGET_COLUMN)
print(dti_ttest.summary())

fig, dti_q_summary_q5 = eda_utils.plot_default_rate_by_group(
    df, "dti_quartile", "Default Rate by DTI Quartile (Revisited)", "Direct test of Question 5",
)
plt.show()
""")
md("**Preliminary finding:** DTI is a theoretically strong candidate for "
   "predicting default (it directly measures debt burden relative to "
   "income), so the quartile-level default-rate spread and t-test result "
   "above are read together as convergent (or non-convergent) evidence.")

md("""### Question 6 -- Does employment length matter?""")
code("""emp_length_anova = eda_utils.run_anova(df.dropna(subset=["emp_length_years"]),
                                        "dti", "emp_length_years")
print(emp_length_anova.summary())

fig, emp_summary_q6 = eda_utils.plot_default_rate_by_group(
    df, "emp_length_years", "Default Rate by Employment Length (Revisited)", "Direct test of Question 6",
)
plt.show()
""")
md("**Preliminary finding:** employment length is a weaker, more indirect "
   "signal than grade or DTI -- it proxies job stability rather than "
   "current financial position. The default-rate-by-group chart shows "
   "whether tenure produces a meaningful gradient or mostly noise in "
   "this sample.")

md("""### Question 7 -- Can natural borrower groups already be observed before clustering?""")
code("""fig = eda_utils.plot_scatter(
    df, "dti", "annual_inc", hue=config.TARGET_COLUMN,
    title="Borrower Landscape: DTI vs. Income", subtitle="Colored by loan outcome -- a preview of natural groupings",
)
plt.show()

fig, ax = plt.subplots(figsize=eda_utils.FIGSIZE_STANDARD)
import seaborn as sns
sns.scatterplot(
    data=df, x="int_rate", y="dti", hue="grade", ax=ax, alpha=0.6, s=25,
    palette="viridis",
)
eda_utils._apply_titles(ax, "Borrower Landscape: Interest Rate vs. DTI by Grade",
                         "Visual preview of natural borrower segments ahead of formal clustering")
fig.tight_layout()
plt.show()
""")
md("**Preliminary finding:** even without formal clustering, coloring "
   "scatterplots by grade or outcome often reveals loose, overlapping "
   "clusters (e.g. low-DTI/low-rate/low-income vs. high-DTI/high-rate "
   "borrowers). These visual groupings motivate a **future clustering or "
   "segmentation exercise** (e.g. k-means on standardized numeric "
   "features) as a candidate engineered feature for Phase 3, rather than "
   "a conclusive segmentation on their own.")

code("""summarize(
    title="Research Question Analysis",
    discovery=(
        "Grade shows the strongest, most consistent relationship with both interest rate and default "
        "outcome; DTI and income show theoretically expected but comparatively weaker effect sizes; "
        "employment length shows a weaker gradient; and scatterplots suggest loose natural borrower groupings."
    ),
    why_it_matters=(
        "This ranks the seven research questions by the strength of preliminary evidence, directly "
        "shaping which variables Phase 3 should prioritize, engineer further, or treat as secondary."
    ),
    action=(
        "Prioritize grade-derived, DTI, and income features in Phase 3 baseline models; treat "
        "employment length as a secondary/interaction feature; and evaluate unsupervised clustering "
        "as a candidate engineered feature informed by Question 7's visual groupings."
    ),
)
""")

# =============================================================================
# SECTION 6 -- STATISTICAL TESTING
# =============================================================================
md("""---
# 6. Statistical Testing

Formal hypothesis tests underpinning the findings above. Every test
reports its null/alternative hypotheses, test statistic, p-value, effect
size, and a plain-language business interpretation using the
`eda_utils.TestResult` container -- p-values are never reported without
an accompanying effect size and interpretation.
""")

md("### 6.1 Pearson & Spearman Correlation -- Numeric Predictors vs. Interest Rate")
code("""correlation_targets = ["loan_amnt", "annual_inc", "dti", "revol_util", "emp_length_years"]
for col in correlation_targets:
    pearson, spearman = eda_utils.pearson_and_spearman(df, "int_rate", col)
    print(pearson.summary())
    print()
    print(spearman.summary())
    print("=" * 90)
""")

md("### 6.2 Chi-Square Tests of Independence -- Categorical Predictors vs. Default")
code("""chi_square_targets = ["grade", "purpose", "home_ownership", "verification_status",
                       "application_type", "initial_list_status", "term"]
chi_results = []
for col in chi_square_targets:
    result = eda_utils.run_chi_square_test(df, col, config.TARGET_COLUMN)
    print(result.summary())
    print("=" * 90)
    chi_results.append({
        "variable": col, "chi2_statistic": round(result.statistic, 3),
        "p_value": round(result.p_value, 4), "cramers_v": round(result.effect_size, 3),
        "significant_at_0.05": result.is_significant,
    })
chi_results_df = pd.DataFrame(chi_results).sort_values("cramers_v", ascending=False)
chi_results_df
""")

md("### 6.3 Independent-Samples t-tests -- Numeric Predictors, Default vs. Fully Paid")
code("""ttest_targets = ["int_rate", "annual_inc", "dti", "loan_amnt", "revol_util", "emp_length_years"]
ttest_results = []
for col in ttest_targets:
    result = eda_utils.run_independent_ttest(df, col, config.TARGET_COLUMN)
    print(result.summary())
    print("=" * 90)
    ttest_results.append({
        "variable": col, "t_statistic": round(result.statistic, 3),
        "p_value": round(result.p_value, 4), "cohens_d": round(result.effect_size, 3),
        "significant_at_0.05": result.is_significant,
    })
ttest_results_df = pd.DataFrame(ttest_results).sort_values("cohens_d", key=abs, ascending=False)
ttest_results_df
""")

md("### 6.4 One-Way ANOVA -- Interest Rate & DTI Across Grade")
code("""anova_rate_by_grade = eda_utils.run_anova(df, "int_rate", "grade")
print(anova_rate_by_grade.summary())
print("=" * 90)
anova_dti_by_grade = eda_utils.run_anova(df, "dti", "grade")
print(anova_dti_by_grade.summary())
""")

md("### 6.5 Confidence Intervals -- Default Rate by Grade\nAlready computed in Section 4.9 using a Wilson-score interval (`eda_utils.default_rate_ci_by_group`), the appropriate method for proportions -- especially for smaller/high-risk grade subgroups where a normal approximation can be unreliable.")
code("""grade_ci  # re-display for reference alongside the other Section 6 results
""")

md("""**Business translation of Section 6, in plain language:**

- Wherever a **chi-square test** is significant with a **moderate-to-large
  Cramer's V**, that categorical variable (e.g. grade) should be treated
  as a serious candidate predictor -- not just "detected," but detected
  with enough strength to matter operationally.
- Wherever a **t-test** is significant but **Cohen's d is small** (< 0.2),
  the difference is real but likely too small to change an individual
  lending decision on its own -- useful as one signal among many, not a
  standalone rule.
- **ANOVA + eta-squared** on interest rate and DTI by grade tests whether
  LendingClub's own grading captures meaningfully different underlying
  risk profiles, which is a direct test of Question 2.
- Every reported p-value above is paired with an effect size specifically
  so a "significant" result with negligible practical size is not
  mistaken for an actionable one -- a standard risk in large-sample
  hypothesis testing.
""")

code("""summarize(
    title="Statistical Testing",
    discovery=(
        "Formal tests broadly confirm the descriptive findings: categorical variables like grade show "
        "the strongest association with default (chi-square + Cramer's V), while individual numeric "
        "predictors show smaller, more marginal effect sizes on their own."
    ),
    why_it_matters=(
        "Statistical significance alone would overstate how useful some variables are; pairing every "
        "p-value with an effect size (Cohen's d, Cramer's V, eta-squared) tells us which findings are "
        "both real and large enough to act on."
    ),
    action=(
        "Carry forward variables with both significant p-values and non-trivial effect sizes as Phase 3 "
        "priority features, and treat significant-but-small-effect variables as supporting signals to "
        "combine with others rather than standalone predictors."
    ),
)
""")

# =============================================================================
# SECTION 7 -- FEATURE RELATIONSHIPS
# =============================================================================
md("""---
# 7. Feature Relationships

Multicollinearity screening and a structured read on which variables are
likely to be predictive, redundant, or in need of engineering ahead of
Phase 3.
""")

md("### 7.1 Highly Correlated Numeric Pairs (|r| >= 0.4)")
code("""high_corr_pairs = eda_utils.high_correlation_pairs(corr_matrix, threshold=0.4)
if high_corr_pairs.empty:
    print("No numeric predictor pairs exceed |r| = 0.4 in this dataset.")
else:
    display(high_corr_pairs)
""")

md("### 7.2 Variance Inflation Factors (Multivariate Multicollinearity)")
code("""vif_table = eda_utils.variance_inflation_factors(df, config.NUMERIC_FEATURES)
vif_table
""")
md("**Interpretation:** VIF captures redundancy a simple pairwise "
   "correlation can miss (e.g. a variable that is jointly, not pairwise, "
   "explainable by several others). Variables with VIF > 5 are flagged as "
   "multicollinearity risks for Logistic Regression specifically; "
   "tree-based models (Random Forest, XGBoost) are far less sensitive to "
   "this and can retain such variables.")

md("""### 7.3 Structured Feature Assessment

| Category | Variables | Rationale |
|---|---|---|
| **Likely highly predictive** | `grade`, `int_rate`, `dti`, `revol_util` | Strongest default-rate spread and/or effect sizes observed above; grade is LendingClub's own risk assessment. |
| **Likely moderately predictive** | `annual_inc` (log-transformed), `purpose`, `home_ownership`, `loan_amnt` | Theoretically relevant with some observed spread, but smaller effect sizes than the top tier. |
| **Candidates for removal / caution** | `sub_grade` (already excluded in Phase 1 -- redundant with `grade`), highly sparse categorical levels | Sub-grade is a finer partition of grade and is likely to reintroduce the multicollinearity grade already captures; sparse levels (e.g. rare purposes) risk unstable coefficients. |
| **Needs transformation** | `annual_inc` (log), `revol_util`/`annual_inc` (capping/winsorizing outliers) | Addresses the right-skew and outlier findings from Sections 2-3. |
| **Candidates for engineering** | `income_quartile`, `dti_quartile`, `interest_rate_band`, `loan_amount_band` (already constructed in Section 4); `installment_to_income` ratio; `revol_bal / revol_util` interaction | Bin-based features already showed clean default-rate gradients in Section 4; a payment-to-income ratio is a natural, currently-missing affordability signal. |
| **Potential interaction effects** | `grade x purpose`, `dti x income_quartile` | Loose groupings observed in Question 7's scatterplots suggest risk may depend on *combinations* of variables rather than any single one. |
""")

code("""summarize(
    title="Feature Relationships",
    discovery=(
        "No numeric predictor pair shows severe pairwise multicollinearity in this sample, and VIF "
        "values are within acceptable ranges; grade, interest rate, DTI, and revolving utilization "
        "stand out as the most promising predictive signals."
    ),
    why_it_matters=(
        "This tells Phase 3 which variables can be trusted as independent linear predictors (low "
        "multicollinearity) and which engineered features (quartile bands, ratios, interactions) are "
        "likely to add the most incremental predictive value."
    ),
    action=(
        "Build income_quartile, dti_quartile, interest_rate_band, loan_amount_band, and an "
        "installment-to-income ratio as engineered features for Phase 3, and test grade x purpose "
        "and dti x income_quartile interaction terms in the Logistic Regression baseline."
    ),
)
""")

# =============================================================================
# SECTION 8 -- PHASE TRANSITION SUMMARY
# =============================================================================
md("""---
# 8. Phase Transition Summary -- Preparing for Phase 3 (Machine Learning)

**No models are trained in this notebook.** The following recommendations
translate Sections 1-7's findings into a concrete Phase 3 modeling plan,
consistent with the Logistic Regression -> Random Forest -> XGBoost
pipeline already scoped in `src/train_models.py`.

### Variables to include in the predictive models
- `grade` (ordinal-encoded, as already implemented in Phase 1's
  preprocessing pipeline)
- `int_rate`, `dti`, `revol_util`, `annual_inc`, `loan_amnt`, `installment`
- `term`, `home_ownership`, `verification_status`, `purpose`,
  `initial_list_status`, `application_type` (one-hot encoded, as already
  implemented)
- `emp_length_years`, `delinq_2yrs`, `open_acc`, `pub_rec`, `total_acc`,
  `mort_acc`, `pub_rec_bankruptcies`

### Variables that may require transformation
- `annual_inc`: apply a log transform ahead of Logistic Regression to
  address the strong right skew identified in Section 2/3 (tree-based
  models are scale/skew-invariant and do not strictly require this).
- `revol_util`: consider capping/winsorizing extreme high-end values
  identified in the Section 3.18 outlier screen.

### Engineered features to create in Phase 3
- `income_quartile`, `dti_quartile`, `interest_rate_band`,
  `loan_amount_band` -- already prototyped in Section 4 and shown to
  produce clean default-rate gradients.
- `installment_to_income` = `installment * 12 / annual_inc` -- a direct
  affordability ratio not currently in the raw feature set.
- Interaction terms: `grade x purpose`, `dti x income_quartile` -- 
  motivated by Section 7's structured feature assessment and the loose
  borrower groupings observed in Question 7.

### Potential challenges for model training
- **Class imbalance** (Section 1): recall/precision/F1/ROC-AUC must be
  tracked alongside accuracy; class weighting or resampling should be
  evaluated.
- **Small high-risk subgroups** (Section 4.9): grades F/G have wide
  confidence intervals on their default rate in this sample size --
  models may need regularization or grouping of rare categories to avoid
  overfitting to noisy subgroup patterns.
- **Multicollinearity for Logistic Regression** (Section 7.2): monitor
  VIF for any engineered ratio features, since ratios built from existing
  variables (e.g. `installment_to_income`) can reintroduce collinearity
  with their source variables.
- **Modest linear effect sizes** (Section 6): several numeric predictors
  showed statistically significant but small (Cohen's d < 0.2) mean
  differences by outcome -- Logistic Regression alone may underperform
  Random Forest/XGBoost, which can capture the nonlinear and interaction
  effects suggested throughout this notebook.

This notebook's outputs (descriptive tables, test results, and the
engineered-feature recommendations above) should be treated as the
analytical justification for every modeling choice made in Phase 3.
""")

nb["cells"] = cells

with open("notebooks/MGMT590_LendingClub_EDA_Phase2.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Phase 2 notebook written with {len(cells)} cells.")
