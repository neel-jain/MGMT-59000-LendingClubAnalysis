"""
Force random_forest for segment vs supervised-model comparison

Usage: from project root run with the project's venv python:
.venv\Scripts\python.exe scripts/force_random_forest_compare.py

This script fits SegmentationEngine on the Phase 1 training split
and prints/saves the segment comparison table using `random_forest`'s
predicted probabilities.
"""
from pathlib import Path
import sys

# ensure imports resolve when run from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils
from src.segmentation_engine import SegmentationEngine

OUT = Path("reports/forced_random_forest_comparison.csv")


def main():
    print("Loading train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()
    print(f"Training rows: {len(X_train)}")

    engine = SegmentationEngine()
    print("Fitting SegmentationEngine on training split...")
    engine.fit(X_train, default_flags=y_train)

    print("Computing segment comparison using random_forest predictions...")
    comparison = engine.compare_with_supervised_models(model_key="random_forest")

    print("Segment comparison:")
    print(comparison.to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT, index=False)
    print(f"Saved comparison to {OUT}")

    if "mean_predicted_probability" in comparison.columns and "average_default_rate" in comparison.columns:
        corr = comparison["mean_predicted_probability"].corr(comparison["average_default_rate"]) 
        print("\nCorrelation between mean_predicted_probability and average_default_rate:", corr)


if __name__ == "__main__":
    main()
