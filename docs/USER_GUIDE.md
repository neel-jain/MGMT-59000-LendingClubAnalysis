# User Guide

**LendingClub Loan Default Risk Dashboard — for Lending Executives, Underwriters, and Portfolio Managers**

No programming experience required. This guide walks through every
dashboard page and what you can do on it.

---

## Getting Started

Ask your technical team to launch the dashboard (`streamlit run app/app.py`
from the project folder) and open the URL it prints — typically
`http://localhost:8501`. The dashboard opens on the **Executive
Dashboard** page.

## The Sidebar (always visible)

- **Navigation** — jump to any of the 8 pages.
- **Scoring Model** — choose which of the three trained models (Logistic
  Regression, Random Forest, XGBoost) powers the pages that use it.
  XGBoost is the default/recommended production model.
- **Borrower Filters** — narrow the Executive Dashboard and Exploratory
  Analysis pages to a subset of borrowers (grade, purpose, income range,
  employment length, home ownership, interest rate, DTI, loan amount).
- **Download Options** — grab the full borrower dataset or the model
  comparison table as a CSV.
- **About This Project** — a one-paragraph project summary (full detail
  on the About Project page).

## Page 1: Executive Dashboard

Your at-a-glance portfolio view: total loans, default rate, average
income/loan amount/interest rate/DTI/employment length, a grade
distribution chart, the top risk-driving variables, and a written
executive summary. Use the sidebar filters to see these numbers for a
specific slice of the portfolio (e.g. only Grade D-G loans).

## Page 2: Exploratory Analysis

Interactive charts (distributions, default rate by group, correlations,
scatter plots) that respond live to the sidebar filters. Use this to
answer "what does our portfolio actually look like" questions before
making a policy change.

## Page 3: Model Comparison

Compares the three trained models side by side: a performance table, an
executive recommendation, and detailed diagnostic charts (ROC curve,
precision-recall curve, confusion matrix, calibration curve, learning
curve) selectable per model.

## Page 4: Borrower Risk Prediction

The core decision tool. Fill in a borrower's income, employment length,
home ownership, loan amount, purpose, interest rate, loan grade, and DTI
(an "Advanced" section has additional credit-profile fields with
sensible defaults already filled in), then click **Predict Risk**. You
get:

- Probability of default, risk tier, confidence score
- Recommended lending action, suggested interest rate, model-driven grade
- A probability gauge and risk meter
- The top 5 risk factors and top 3 protective factors driving THIS score
- An executive summary in plain language
- A SHAP waterfall and force plot (visual breakdowns of the score)
- Downloadable risk assessment and explanation reports

## Page 5: Borrower Segmentation

Shows the natural borrower groups the system has identified (e.g. "Prime
Borrowers," "High Risk Borrowers" -- exact names depend on your data).
Select a segment to see its typical profile and tailored business
recommendations (lending policy, rate strategy, underwriting approach,
marketing strategy, portfolio notes). Includes visual maps of how
segments relate to each other and a cross-check against the supervised
model's risk scores.

## Page 6: Business Insights

Reads like a short executive report: each of the project's seven
research questions, paired with its finding, supporting evidence, a
chart, a business recommendation, and the decision impact.

## Page 7: Model Explainability

The population-level companion to Page 4's individual explanations:
which variables matter most overall, SHAP summary charts, and how
specific variables (like DTI or interest rate) relate to predicted risk
across the whole portfolio.

## Page 8: About Project

The full project write-up -- business problem, research questions,
methodology, technology stack, limitations, and future improvements.

## Frequently Asked Questions

**Why do I see different numbers if I switch the Scoring Model dropdown?**
Each model was trained independently and scores borrowers slightly
differently -- that's expected and is exactly what Page 3 helps you
compare.

**What if a page shows a warning instead of data?**
It means a required file hasn't been generated yet (usually because the
underlying data/model pipeline hasn't been run). The warning tells you
the exact command to run -- pass it to your technical team.

**Can I change the risk-tier thresholds (what counts as "High Risk")?**
Yes -- a technical team member can edit `reports/risk_threshold_config.json`
directly in a text editor; no code change or restart of the underlying
models is required, only a dashboard refresh.
