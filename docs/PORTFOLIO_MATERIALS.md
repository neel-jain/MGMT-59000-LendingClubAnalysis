# Portfolio Materials

Ready-to-use descriptions of this project for a resume, LinkedIn,
portfolio website, and interviews.

---

## Resume Project Description (bullet form)

> **LendingClub Loan Default Risk Platform** — Built a full-stack
> lending-risk analytics platform (Python, scikit-learn, XGBoost, SHAP,
> Streamlit) predicting loan default for Indiana borrowers; designed
> three reusable engines (risk scoring, SHAP explainability, borrower
> segmentation) wrapped in an 8-page executive dashboard, backed by
> 265+ automated tests (unit, integration, edge-case, and UI-level).

## LinkedIn Project Post

> I just finished a capstone project I'm genuinely proud of: a
> lending-risk platform for LendingClub's Indiana borrowers that goes
> well past "train a model and report accuracy."
>
> It's built as four reusable engines:
> A **RiskScoringEngine** that turns a borrower's application into a
> probability, a risk tier, and a concrete lending recommendation
> An **ExplainabilityEngine** using SHAP that explains every
> prediction in plain business language — no black box
> A **SegmentationEngine** that finds natural borrower groups and
> cross-validates them against the supervised model's own predictions
> An 8-page **Streamlit dashboard** that orchestrates all three —
> with zero modeling logic baked into the UI itself
>
> Comparing Logistic Regression, Random Forest, and XGBoost across a
> full evaluation suite (not just accuracy — ROC-AUC, calibration,
> Matthews correlation coefficient, and more), then explaining *why*
> each prediction happened, then finding the natural customer segments
> underneath it all — this is the project that taught me the most about
> what separates a class assignment from something closer to a real
> product.
>
> 265+ automated tests, a deployment guide, and a full technical report
> later, I'm sharing it here. #MachineLearning #DataScience #FinTech
> #ExplainableAI #Streamlit

## Portfolio Website Description

> **LendingClub Loan Default Risk Platform**
> A production-style lending analytics application built end-to-end:
> data pipeline, three tuned supervised models, SHAP-based
> explainability, unsupervised borrower segmentation, and an executive
> Streamlit dashboard. Designed around a strict architectural boundary
> — all machine-learning logic lives in independently-tested,
> reusable Python engines; the dashboard only orchestrates them.
> Backed by 265+ automated tests spanning unit, integration, edge-case,
> and UI-level verification.
>
> **Stack:** Python, pandas, scikit-learn, XGBoost, SHAP, UMAP,
> Streamlit, pytest.
> **Highlights:** leakage-safe ML pipeline design, cost-based decision
> threshold optimization, data-driven (not assumption-driven) customer
> segment naming, and a fully documented deployment guide for four
> hosting platforms.

## 30-Second Project Summary (elevator pitch)

> "I built a lending-risk platform that predicts whether a LendingClub
> borrower will default, explains exactly why using SHAP, and groups
> borrowers into natural segments for portfolio strategy — all wired
> into a Streamlit dashboard a lending executive could actually use.
> It's not just a model; it's three reusable engines and 265 automated
> tests behind them."

## 2-Minute Interview Explanation

> "The core problem was: LendingClub needs to decide whether to lend to
> a borrower, and a raw probability isn't enough — you need to explain
> the decision and understand the portfolio it sits in. So I built this
> as three separate, reusable engines rather than one monolithic
> script.
>
> First, a RiskScoringEngine wraps whichever of three trained models —
> Logistic Regression, Random Forest, or XGBoost — performs best, and
> converts its probability into a risk tier, a confidence score, and a
> concrete recommendation, using business thresholds that are stored in
> an editable JSON file, not hardcoded, so a credit-policy team could
> adjust them without touching code.
>
> Second, an ExplainabilityEngine uses SHAP to explain every prediction
> — both at the individual-borrower level, so you can see exactly which
> factors pushed one specific applicant's score up or down, and at the
> population level, so you know which variables matter most overall.
>
> Third, a SegmentationEngine finds natural borrower groups through
> clustering — completely independent of the supervised model — and
> then I cross-check those two independent analyses against each other.
> When a cluster the model never saw the label for lines up with the
> supervised model's own risk ranking, that's real validation, not just
> a nice chart.
>
> All three get wired into an eight-page Streamlit dashboard, but the
> dashboard itself contains zero modeling logic — every page just calls
> one of the engines and displays what comes back. And the whole thing
> is backed by 265-plus automated tests, including a suite specifically
> designed to throw broken inputs at it — missing values, negative
> income, corrupted files — to make sure it fails gracefully instead of
> crashing.
>
> The part I'm most proud of is probably the leakage-prevention design:
> preprocessing lives inside each model's own scikit-learn Pipeline, so
> every cross-validation fold refits its own statistics rather than
> leaking information from validation data into training — that's a
> subtle mistake that's easy to make and hard to notice until it's
> already inflated your reported accuracy."
