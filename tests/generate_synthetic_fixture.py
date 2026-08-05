"""
generate_synthetic_fixture.py
==============================
Generates a SYNTHETIC LendingClub-shaped CSV purely so the Phase 1
pipeline (ingestion -> validation -> cleaning -> split -> preprocessing)
can be executed and verified end-to-end in this environment, where the
real ~37,515-row Indiana LendingClub extract is not available.

This is a TEST FIXTURE ONLY — it is not real LendingClub data and must
NOT be used for any actual analysis or reporting. Replace the file at
config.RAW_DATA_PATH with the genuine data export before running the
real project pipeline.

Usage
-----
    python tests/generate_synthetic_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

RNG = np.random.default_rng(seed=config.RANDOM_STATE)
N_ROWS = 2_000  # small synthetic sample; real file has ~37,515 rows


def generate_synthetic_dataframe(n_rows: int = N_ROWS) -> pd.DataFrame:
    """Build a synthetic dataframe matching the expected raw schema."""
    grades = RNG.choice(list("ABCDEFG"), size=n_rows, p=[.25, .25, .2, .15, .08, .04, .03])
    terms = RNG.choice([" 36 months", " 60 months"], size=n_rows, p=[.7, .3])
    home_ownership = RNG.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_rows, p=[.4, .45, .13, .02]
    )
    verification = RNG.choice(
        ["Verified", "Source Verified", "Not Verified"], size=n_rows
    )
    purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "other",
         "major_purchase", "small_business"],
        size=n_rows, p=[.45, .2, .1, .15, .05, .05],
    )
    emp_length_options = list(config.EMP_LENGTH_MAP.keys()) + [np.nan]
    emp_length = RNG.choice(emp_length_options, size=n_rows)

    loan_status = RNG.choice(
        ["Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)"],
        size=n_rows, p=[.62, .18, .02, .15, .03],
    )

    int_rate = np.round(RNG.normal(13, 4, n_rows).clip(5, 31), 2)
    revol_util = np.round(RNG.normal(45, 22, n_rows).clip(0, 150), 1)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "member_id": np.arange(100_000, 100_000 + n_rows),
            "loan_amnt": RNG.integers(1000, 40000, n_rows),
            "term": terms,
            "int_rate": [f"{v}%" for v in int_rate],
            "installment": np.round(RNG.uniform(30, 1500, n_rows), 2),
            "grade": grades,
            "sub_grade": [f"{g}{RNG.integers(1, 6)}" for g in grades],
            "emp_title": RNG.choice(["Teacher", "Nurse", "Manager", np.nan], size=n_rows),
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": np.round(RNG.lognormal(10.8, 0.5, n_rows), 2),
            "verification_status": verification,
            "issue_d": RNG.choice(["Jan-2015", "Jun-2016", "Mar-2017"], size=n_rows),
            "loan_status": loan_status,
            "purpose": purpose,
            "title": purpose,
            "zip_code": [f"{RNG.integers(460, 479)}xx" for _ in range(n_rows)],
            "addr_state": RNG.choice(["IN"] * 9 + ["OH"], size=n_rows),
            "dti": np.round(RNG.normal(18, 8, n_rows).clip(0, 50), 2),
            "delinq_2yrs": RNG.poisson(0.3, n_rows),
            "earliest_cr_line": RNG.choice(["Jan-1998", "Jun-2002", "Mar-2010"], size=n_rows),
            "open_acc": RNG.integers(1, 25, n_rows),
            "pub_rec": RNG.poisson(0.15, n_rows),
            "revol_bal": RNG.integers(0, 60000, n_rows),
            "revol_util": [f"{v}%" for v in revol_util],
            "total_acc": RNG.integers(2, 60, n_rows),
            "initial_list_status": RNG.choice(["w", "f"], size=n_rows),
            "application_type": RNG.choice(
                ["Individual", "Joint App"], size=n_rows, p=[.92, .08]
            ),
            "mort_acc": RNG.integers(0, 8, n_rows),
            "pub_rec_bankruptcies": RNG.poisson(0.08, n_rows),
            "desc": np.nan,
            "url": np.nan,
        }
    )

    # Inject a handful of duplicate rows and missing values, mirroring the
    # kinds of data-quality issues validate_dataset() is designed to catch.
    df = pd.concat([df, df.sample(15, random_state=config.RANDOM_STATE)], ignore_index=True)
    missing_idx = RNG.choice(df.index, size=40, replace=False)
    df.loc[missing_idx, "annual_inc"] = np.nan

    return df


if __name__ == "__main__":
    synthetic_df = generate_synthetic_dataframe()
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(config.RAW_DATA_PATH, index=False)
    print(
        f"Synthetic TEST FIXTURE written to {config.RAW_DATA_PATH} "
        f"({len(synthetic_df):,} rows). This is NOT real LendingClub data."
    )
