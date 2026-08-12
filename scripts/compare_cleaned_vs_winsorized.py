"""
Compare model per-cluster predicted probabilities when using the
original cleaned dataset vs the winsorized dataset.

Saves CSVs under `reports/` and prints a concise comparison.
"""
from pathlib import Path
import sys
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils, config
from src.risk_scoring import RiskScoringEngine
from src.segmentation_engine import SegmentationEngine

OUT_CLEAN = Path("reports/cleaned_model_prob_by_cluster.csv")
OUT_WINS = Path("reports/winsorized_model_prob_by_cluster.csv")
model_keys = ["logistic_regression", "random_forest", "xgboost"]

# Helper to compute per-cluster means given a DataFrame with target
def compute_cluster_means(df, dataset_label):
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(df)
    engine = SegmentationEngine()
    engine.fit(X_train, default_flags=y_train)
    labels = engine.fit_result.labels

    rows_df = None
    for key in model_keys:
        rs = RiskScoringEngine(model_key=key)
        proba = rs.predict_probability(X_train)
        df_tmp = pd.DataFrame({"cluster": labels, "proba": proba})
        mean_by_cluster = df_tmp.groupby("cluster").mean().reset_index().rename(columns={"proba": key})
        if rows_df is None:
            rows_df = mean_by_cluster
        else:
            rows_df = rows_df.merge(mean_by_cluster, on="cluster")
    rows_df = rows_df.merge(engine.generate_cluster_profile().reset_index()[["cluster", "average_default_rate"]], on="cluster")
    rows_df["dataset"] = dataset_label
    rows_df["segment_name"] = rows_df["cluster"].map(engine.fit_result.segment_names)
    return rows_df, engine

# Load cleaned dataset
if not config.CLEANED_DATA_PATH.exists():
    print(f"Cleaned dataset not found at {config.CLEANED_DATA_PATH}. Run Phase1 to generate it.")
    sys.exit(1)
cleaned_df = pd.read_csv(config.CLEANED_DATA_PATH)
cleaned_rows, engine_clean = compute_cluster_means(cleaned_df, "cleaned")
cleaned_rows.to_csv(OUT_CLEAN, index=False)
print(f"Saved cleaned per-cluster means to {OUT_CLEAN}")

# Load winsorized dataset
if not config.WINSORIZED_DATA_PATH.exists():
    print(f"Winsorized dataset not found at {config.WINSORIZED_DATA_PATH}. Run winsorize step first.")
    sys.exit(1)
wins_df = pd.read_csv(config.WINSORIZED_DATA_PATH)
wins_rows, engine_wins = compute_cluster_means(wins_df, "winsorized")
wins_rows.to_csv(OUT_WINS, index=False)
print(f"Saved winsorized per-cluster means to {OUT_WINS}")

# Align by segment_name (cluster numeric ids may permute between fits)
cleaned_rows = cleaned_rows.rename(columns={"segment_name": "segment_name_cleaned"})
wins_rows = wins_rows.rename(columns={"segment_name": "segment_name_wins"})
merge = cleaned_rows.merge(wins_rows, left_on="segment_name_cleaned", right_on="segment_name_wins", suffixes=("_cleaned", "_wins"))
cols_to_show = ["segment_name_cleaned"] + [f"{k}_cleaned" for k in model_keys] + [f"{k}_wins" for k in model_keys] + ["average_default_rate_cleaned"]
print("\nPer-segment comparison (cleaned vs winsorized):")
print(merge[cols_to_show].to_string(index=False))

# compute correlations between cleaned and winsorized per-model means (aligned by segment name)
for k in model_keys:
    c = merge[f"{k}_cleaned"].corr(merge[f"{k}_wins"])
    print(f"Correlation of {k} means (cleaned vs winsorized): {c:.6f}")

print("\nDone.")
