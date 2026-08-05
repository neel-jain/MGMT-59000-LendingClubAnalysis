# Deployment Guide — Phase 6

**MGMT 590 LendingClub Loan Default Risk Capstone**

This guide covers environment setup, launch instructions, and
platform-specific deployment guidance. **No deployment has actually been
performed** — this is preparation and guidance only, per the Phase 6 brief.

---

## 1. Prerequisites

- Python 3.12 (see `runtime.txt`)
- The real ~37,515-row Indiana LendingClub extract at
  `data/raw/lendingclub_indiana_raw.csv` (or the synthetic test fixture
  via `python tests/generate_synthetic_fixture.py` for a smoke test)

## 2. Environment Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Full development environment (notebooks, pytest, everything):
pip install -r requirements.txt

# OR, for a deployment-only environment (lighter, no Jupyter/pytest):
pip install -r requirements-app.txt
```

### Environment variables (optional)

| Variable | Purpose | Default |
|---|---|---|
| `MGMT590_PROJECT_ROOT` | Override the project root (useful if data/model artifacts are mounted at a different path than the source code, e.g. a container with a read-only image + writable volume) | Directory containing `src/config.py`'s parent |
| `MGMT590_LOG_LEVEL` | Set the logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) without a code change | `INFO` |

Streamlit's own configuration (`.streamlit/config.toml`) controls the
theme and, if needed, the server port — see Streamlit's configuration
documentation for the full list (`server.port`, `server.address`, etc.),
all of which can also be set via `STREAMLIT_SERVER_PORT`-style
environment variables without editing the file.

## 3. Generating Required Artifacts (once, before first launch)

The dashboard reads serialized artifacts; it never trains anything
itself. Generate them once:

```bash
python -m src.train_models   # Phase 1 (data pipeline) + Phase 3 (model training)

python -c "
from src.explainability import ExplainabilityEngine
ExplainabilityEngine().persist_explainability_artifacts()
"

python -c "
from src import utils
from src.segmentation_engine import SegmentationEngine
X_train, _, _, y_train, _, _ = utils.load_splits()
engine = SegmentationEngine()
engine.fit(X_train, default_flags=y_train)
engine.persist_segmentation_artifacts()
"
```

## 4. Launch Instructions

**Local / any server with a persistent filesystem:**

```bash
streamlit run app/app.py
```

Run this from the **project root** (not from inside `app/`) so
`.streamlit/config.toml` and the relative artifact paths in `src/config.py`
resolve correctly.

**Verify the app before deploying** (headless, no browser needed):

```bash
pytest tests/test_app.py tests/test_integration.py tests/test_edge_cases.py -v
```

## 5. Folder & Dependency Verification Checklist

Before deploying, confirm:

- [ ] `data/raw/lendingclub_indiana_raw.csv` exists (the real extract, not the synthetic fixture, for a production deployment)
- [ ] `models/*.joblib` (all three supervised models + the clustering model) exist
- [ ] `pipelines/preprocessing_pipeline.joblib` exists
- [ ] `reports/*.joblib`, `reports/explainability/*.joblib`, `reports/segmentation/*.joblib` exist
- [ ] `reports/risk_threshold_config.json` exists (auto-bootstraps on first `RiskScoringEngine()` construction if missing)
- [ ] `pip check` reports no dependency conflicts against your chosen requirements file
- [ ] `pytest tests/ -q` passes in the target environment (library versions can differ subtly between platforms — always re-verify)

## 6. Deployment Platform Guidance

None of these have been used to actually deploy this application — the
notes below are preparation guidance based on each platform's documented
capabilities as of this project's development.

### Streamlit Community Cloud
- **Best fit for:** this project specifically — it is purpose-built for
  Streamlit apps, free for public repositories, and requires no
  Dockerfile or server configuration.
- **Setup:** connect a GitHub repository, point it at `app/app.py`, and
  specify `requirements-app.txt` as the dependency file in the app
  settings.
- **Caveat:** the generated artifacts (`models/`, `reports/`, `data/processed/`,
  `pipelines/`) must be committed to the repository (or fetched via a
  startup script) since Community Cloud does not run an arbitrary
  pre-deploy build step by default — the pipeline commands in Section 3
  need to be run and their outputs committed before pushing, or wired
  into a startup hook.
- **Resource limits:** free tier is memory-constrained (historically
  ~1GB); this app's measured ~517MB footprint (Section 6 of
  `PERFORMANCE_REPORT.md`) leaves some headroom but is worth monitoring,
  especially once the real (larger) dataset replaces the synthetic fixture.

### Render
- **Best fit for:** teams wanting more control over the runtime
  environment (a proper Dockerfile or native Python web service) than
  Streamlit Community Cloud offers, while still being simpler to
  configure than a raw VM.
- **Setup:** create a "Web Service", point the build command at
  `pip install -r requirements-app.txt`, and the start command at
  `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`
  (Render injects `$PORT`; Streamlit must bind to it and to `0.0.0.0`,
  not `localhost`, to be reachable).
- **Caveat:** Render's free tier spins down idle services, so the first
  request after idling will pay the full cold-start cost measured in
  `PERFORMANCE_REPORT.md` Section 2 (~18s) — acceptable for a
  demo/portfolio use case, less so for a always-on production service.

### Railway
- **Best fit for:** similar profile to Render — more infrastructure
  control than Streamlit Community Cloud, simple Git-push deployment.
- **Setup:** similar to Render — a `Procfile` or start command of
  `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`,
  with `requirements-app.txt` (or a `nixpacks.toml`/Dockerfile) driving
  the build.
- **Caveat:** Railway's free tier is usage-metered (compute-hours) rather
  than always-free — worth checking current pricing before a long-lived
  demo deployment.

### Hugging Face Spaces
- **Best fit for:** a project already framed around ML/data science
  (as this one is) that wants a "Spaces" listing alongside the model/
  data-science community, with straightforward Streamlit SDK support.
- **Setup:** create a Space with the "Streamlit" SDK selected, push this
  repository's contents (including generated artifacts, or a startup
  script that generates them), and Spaces handles the
  `streamlit run app.py`-equivalent invocation automatically based on
  its own configuration file (`README.md` YAML front-matter specifying
  `sdk: streamlit` and the entry file).
- **Caveat:** free-tier Spaces have modest CPU/memory allocations and
  will sleep after inactivity, similar to Render/Railway's free tiers —
  the cold-start numbers in `PERFORMANCE_REPORT.md` again apply.

### Choosing among these four
For this specific project (a course capstone intended for live
demonstration to faculty), **Streamlit Community Cloud** is the
simplest and most purpose-built choice — no Dockerfile, no manual port
binding, and a URL immediately shareable for a live demo. Render/
Railway/Hugging Face Spaces become more attractive if the project later
needs a custom domain, a background job/cron for periodic model
retraining, or resource limits beyond what Community Cloud's free tier
offers.

## 7. Post-Deployment Smoke Test

After deploying anywhere, manually verify (or automate via
`tests/test_app.py`'s pattern against the live URL, if the platform
supports headless browser testing):

1. The Executive Dashboard loads and shows non-empty KPIs.
2. Borrower Risk Prediction: fill the form, click Predict, confirm all
   6 result metrics and both SHAP plots render.
3. Borrower Segmentation: confirm the PCA/t-SNE tabs both render (t-SNE
   will be the slowest single interaction — see `PERFORMANCE_REPORT.md`).
4. At least one CSV download and one Markdown report download succeed.

## 8. Security & Privacy Notes

- No API keys, credentials, or sensitive configuration are hardcoded
  anywhere in this codebase — all paths are computed from `Path(__file__)`
  or the `MGMT590_PROJECT_ROOT` environment variable.
- The application only reads local files under the project's own `data/`,
  `models/`, `pipelines/`, and `reports/` directories — it does not accept
  arbitrary file paths from user input anywhere, so there is no path-
  traversal surface from the Streamlit UI.
- All user-supplied form inputs (Borrower Risk Prediction page) are
  numeric/categorical widgets with bounded ranges (`st.number_input`'s
  `min_value`, `st.slider`'s fixed range, `st.selectbox`'s fixed option
  list) — there is no free-text field that reaches the model or the
  filesystem.
- **Limitation, stated plainly:** this project makes no claim about
  production-grade authentication, authorization, or rate-limiting.
  Deploying it publicly means anyone with the URL can use the
  prediction/explanation/segmentation tools — acceptable for a
  portfolio/demo deployment, but a genuine production lending platform
  would need an authentication layer this project does not implement.
- The dataset itself contains no legally protected class attributes and
  no personally identifying borrower information (LendingClub's public
  data is already anonymized at the source) — see `QA_CHECKLIST.md` and
  the About Project page for the corresponding fairness-assessment
  limitation.
