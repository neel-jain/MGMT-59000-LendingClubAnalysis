"""
winsorize_data.py
================
Utility script for winsorizing LendingClub features that have extreme
outliers, specifically `annual_inc` and `dti`.

This script is intended to be run from the project root:
    python -m src.winsorize_data

It reads the raw or cleaned dataset, clips extreme values using an IQR
rule, and writes a winsorized copy of the dataset for regression-ready
analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config, utils


def run_winsorization(
    input_path: Path | None = None,
    output_path: Path | None = None,
    columns: list[str] = config.WINSORIZE_FEATURES,
    iqr_multiplier: float = config.WINSORIZE_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Load data, winsorize selected columns, and save the result."""
    input_path = input_path or config.CLEANED_DATA_PATH
    output_path = output_path or config.WINSORIZED_DATA_PATH

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at {input_path}. Run the phase1 pipeline or place a cleaned dataset there first."
        )

    df = pd.read_csv(input_path)
    winsorized = utils.winsorize_columns(df, columns=columns, iqr_multiplier=iqr_multiplier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    winsorized.to_csv(output_path, index=False)
    utils.logger.info(
        "Saved winsorized dataset with shape %s to %s",
        winsorized.shape,
        output_path,
    )
    return winsorized


if __name__ == "__main__":
    run_winsorization()
