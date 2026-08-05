"""
Generates notebooks/MGMT590_LendingClub_Analysis.ipynb using nbformat.
Run once from the project root: python tests/build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# MGMT 590 — LendingClub Loan Default Risk (Indiana Borrowers)
## Phase 1: Project Architecture & Data Foundation

**Course:** MGMT 59000, Summer 2026, Section DY2 — Purdue University
**Scope of this notebook:** data ingestion, validation, cleaning, feature
engineering, leakage-safe train/validation/test splitting, and
preprocessing-pipeline construction/serialization.

This notebook is a thin, readable wrapper around the reusable project
modules `src/config.py`, `src/utils.py`, and `src/train_models.py`. All
real logic lives in those modules so that Phase 2+ notebooks/scripts can
import and reuse it without duplication or drift.

> **Note on data:** the real Indiana LendingClub extract
> (~37,515 rows, ~27.5 MB) should be placed at `data/raw/lendingclub_indiana_raw.csv`
> before running this notebook for the actual project. A small synthetic
> fixture generator (`tests/generate_synthetic_fixture.py`) is included
> only to verify the pipeline runs correctly in the absence of the real
> file — it is NOT a substitute for the genuine dataset.
""")

code("""# If the real raw data file is not yet present, uncomment the line below
# to generate a small synthetic fixture so the rest of this notebook runs
# end-to-end. Replace with the real LendingClub Indiana extract for the
# actual analysis.

# import subprocess, sys
# subprocess.run([sys.executable, "../tests/generate_synthetic_fixture.py"])
""")

code("""import sys
from pathlib import Path

# Allow imports from the project's src/ package when running from notebooks/
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np

from src import config, utils

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
""")

md("## 1. Project Setup\nEnsure all project directories exist.")
code("""utils.ensure_directories()
print("Project root:", config.PROJECT_ROOT)
print("Raw data expected at:", config.RAW_DATA_PATH)
""")

md("""## 2. Data Ingestion
Load the raw CSV extract using the reusable `load_raw_data()` loader,
which raises clear, actionable errors if the file is missing, empty, or
unparsable.""")
code("""raw_df = utils.load_raw_data()
print("Raw shape:", raw_df.shape)
raw_df.head()
""")

code("""raw_df.info()
""")

md("""## 3. Data Validation
`validate_dataset()` runs a structured battery of checks — schema drift,
missing values, duplicate rows, dtypes, and known business-rule
violations — and returns a report dict without mutating the data.""")
code("""validation_report = utils.validate_dataset(raw_df)

print(f"Rows: {validation_report['n_rows']:,} | Columns: {validation_report['n_columns']}")
print(f"Duplicate rows: {validation_report['duplicate_row_count']:,}")
print(f"Missing expected columns: {validation_report['missing_columns']}")
print("\\nColumns with missing values:")
validation_report["missing_values"]
""")

code("""print("Invalid / out-of-scope value checks:")
for check, count in validation_report["invalid_values"].items():
    print(f"  {check}: {count:,}")
""")

code("""validation_report["dtypes"]
""")

md("""## 4. Data Cleaning & Feature Engineering
`clean_dataset()` orchestrates the full Phase 1 cleaning sequence:

1. Filter to Indiana borrowers (`addr_state == 'IN'`)
2. Remove exact duplicate rows
3. Convert percentage-string columns (`int_rate`, `revol_util`) to numeric
4. Parse `emp_length` free text into numeric years (`emp_length_years`)
5. Build the binary target `default_flag`
   (`Charged Off`/`Default` → 1, `Fully Paid` → 0; all other statuses —
   e.g. `Current`, `Late (31-120 days)` — are **removed** since those
   loans have not reached a final resolution)
6. Drop identifier / free-text / leakage-prone / superseded columns

Each step is also available individually in `src/utils.py` for
inspection or reuse.""")
code("""cleaned_df = utils.clean_dataset(raw_df)
print("Cleaned shape:", cleaned_df.shape)
cleaned_df.head()
""")

code("""print("Target class balance (default_flag):")
cleaned_df[config.TARGET_COLUMN].value_counts(normalize=True).round(4)
""")

code("""# Persist the cleaned dataset for downstream phases / reporting.
utils.save_dataframe(cleaned_df, config.CLEANED_DATA_PATH)
""")

md("""## 5. Train / Validation / Test Split (Leakage-Safe)
`split_data()` carves out the **test** set first, then the
**validation** set from the remaining pool, stratifying on the binary
target at each step. No preprocessing statistics are ever computed using
validation or test rows — the preprocessing pipeline in the next section
is fit **only** on `X_train`.""")
code("""X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(cleaned_df)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Default rate — train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")
""")

code("""utils.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
""")

md("""## 6. Preprocessing Pipeline (ColumnTransformer + Pipeline)
`build_preprocessing_pipeline()` returns an **unfitted**
`ColumnTransformer` with three branches:

- **Numeric** (`SimpleImputer(median)` → `StandardScaler`)
- **One-hot categorical** (`SimpleImputer(most_frequent)` → `OneHotEncoder`)
- **Ordinal categorical** — `grade`, encoded with its natural risk
  ordering A < B < ... < G (`SimpleImputer(most_frequent)` →
  `OrdinalEncoder`)

It is fit **only** on `X_train` to prevent data leakage, then reused
(`.transform()`) on validation/test/live data in later phases.""")
code("""preprocessor = utils.build_preprocessing_pipeline()
preprocessor.fit(X_train)

feature_names = utils.get_output_feature_names(preprocessor)
print(f"Fitted. Output feature count: {len(feature_names)}")
feature_names[:15]
""")

code("""X_train_transformed = preprocessor.transform(X_train)
print("Transformed training feature matrix shape:", X_train_transformed.shape)
""")

md("## 7. Serialize the Fitted Preprocessor\nSaved via `joblib` for reuse in Phase 2 model training and the final Streamlit app.")
code("""utils.save_object(preprocessor, config.PREPROCESSOR_PATH)
print("Saved to:", config.PREPROCESSOR_PATH)
""")

md("""## 8. Phase 1 Summary

| Artifact | Path |
|---|---|
| Cleaned dataset | `data/processed/lendingclub_indiana_cleaned.csv` |
| Train/val/test splits | `data/splits/*.csv` |
| Fitted preprocessing pipeline | `pipelines/preprocessing_pipeline.joblib` |
| Pipeline run log | `logs/pipeline.log` |

**Not implemented in Phase 1 (by design):** Logistic Regression, Random
Forest, and XGBoost model training and evaluation. These are stubbed out
with fixed signatures in `src/train_models.py`
(`train_logistic_regression`, `train_random_forest`, `train_xgboost`,
`evaluate_model`) and will be implemented in Phase 2 without requiring
any changes to the modules built here.

### Equivalent one-line pipeline run
Everything above can also be run non-interactively via:
```python
from src.train_models import run_phase1_pipeline
artifacts = run_phase1_pipeline()
```
or from the command line: `python -m src.train_models`
""")

nb["cells"] = cells

with open("notebooks/MGMT590_LendingClub_Analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
