# Live Demonstration Script

A click-by-click guide for demoing the dashboard live, timed to slot
into Slide 11 of the presentation (~3-5 minutes).

## Before You Start

```bash
python -m src.train_models
python -c "from src.explainability import ExplainabilityEngine; ExplainabilityEngine().persist_explainability_artifacts()"
python -c "
from src import utils
from src.segmentation_engine import SegmentationEngine
X_train, _, _, y_train, _, _ = utils.load_splits()
e = SegmentationEngine(); e.fit(X_train, default_flags=y_train); e.persist_segmentation_artifacts()
"
streamlit run app/app.py
```

Open the app in a browser tab BEFORE your presentation starts, and let
the first page load fully (so cold-start costs — see
`PERFORMANCE_REPORT.md` — are already paid before you're on camera).

## Step 1 — Dashboard Overview (30 seconds)

Land on **Executive Dashboard**. Point out: total loans, default rate,
grade distribution chart, top risk factors, and the executive summary
paragraph at the bottom. Say: *"This is the first thing a lending
executive sees — portfolio health at a glance."*

## Step 2 — Exploratory Analysis (30 seconds)

Click **Exploratory Analysis** in the sidebar. Adjust one sidebar filter
(e.g. Loan Grade) and show the charts update live. Say: *"Every chart
here responds to the same sidebar filters, so an analyst can slice the
portfolio however they need."*

## Step 3 — Borrower Risk Prediction (90 seconds — the centerpiece)

Click **Borrower Risk Prediction**. Fill in a borrower profile (or use
the pre-filled defaults) — call out 2-3 fields as you go (income, DTI,
grade). Click **Predict Risk**.

Walk through the results top to bottom:
- Probability of Default, Risk Tier, Confidence Score, Recommended Action
- The probability gauge and risk meter
- **"Why This Prediction?"** — the executive summary sentence, then the
  top 5 risk factors and top 3 protective factors
- The SHAP waterfall plot — *"this shows exactly how each factor pushed
  the prediction up or down from the average borrower"*
- Scroll to the exportable reports and click one download button to
  show it works

## Step 4 — Borrower Segmentation (45 seconds)

Click **Borrower Segmentation**. Show the segment comparison table,
then select a segment from the dropdown and show its profile and
recommendations. Point at the PCA tab briefly (skip t-SNE live — it's
the slowest single interaction, per `PERFORMANCE_REPORT.md` — unless
you have time to spare).

## Step 5 — Wrap-Up (15 seconds)

Return to **Executive Dashboard** or **Business Insights** to close on
a portfolio-level view. Say: *"Every number you just saw traced back to
one of three reusable engines — nothing in this dashboard's code
computes a prediction, an explanation, or a segment itself."*

## If Something Goes Wrong

- **A page shows a warning instead of data:** you skipped the "Before
  You Start" setup step — run the missing command it names, or fall
  back to narrating over the thumbnails in
  `docs/MGMT590_Capstone_Presentation.pptx`.
- **The app is slow on first load:** this is the one-time cold-start
  cost (model loading, SHAP background sample, segmentation fit) —
  narrate through it ("the first load does more work than every
  request after it") rather than sitting in silence.
- **Running remotely / no live app available:** narrate this exact
  script over static screenshots (see `README.md`'s Screenshots
  section for the suggested set to have ready).
