"""
Compare per-borrower and per-cluster predicted probabilities across saved models.
Saves `reports/model_prob_comparison_by_cluster.csv` and prints correlation matrices.
"""
from pathlib import Path
import sys
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils
from src.risk_scoring import RiskScoringEngine
from src.segmentation_engine import SegmentationEngine

OUT = Path("reports/model_prob_comparison_by_cluster.csv")
model_keys = ["logistic_regression", "random_forest", "xgboost"]

# load splits and fit segmentation on X_train
X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()
engine = SegmentationEngine()
engine.fit(X_train, default_flags=y_train)
labels = engine.fit_result.labels

# per-borrower probabilities
proba_df = pd.DataFrame(index=X_train.index)
for key in model_keys:
    rs = RiskScoringEngine(model_key=key)
    proba = rs.predict_probability(X_train)
    proba_df[key] = proba

# correlations across models (per-borrower)
corr_per_borrower = proba_df.corr()
print("\nPer-borrower probability correlation matrix:")
print(corr_per_borrower)

# add cluster labels and compute per-cluster means
proba_df["cluster"] = labels
cluster_means = proba_df.groupby("cluster").mean().reset_index()
# map cluster id to segment name
segment_names = engine.fit_result.segment_names
cluster_means["segment_name"] = cluster_means["cluster"].map(segment_names)
# reorder columns
cols = ["segment_name", "cluster"] + model_keys
cluster_means = cluster_means[cols]
cluster_means = cluster_means.sort_values(by=model_keys[0], ascending=False)

# compute correlation of per-cluster means vs average_default_rate
profile = engine.generate_cluster_profile().reset_index()
cluster_profile = profile[["cluster", "average_default_rate"]]
cluster_merge = cluster_means.merge(cluster_profile, on="cluster")

# per-cluster correlation matrix across models
cluster_corr = cluster_means[model_keys].corr()
print("\nPer-cluster mean probability correlation matrix:")
print(cluster_corr)

print("\nPer-cluster means merged with average_default_rate:")
print(cluster_merge.to_string(index=False))

OUT.parent.mkdir(parents=True, exist_ok=True)
cluster_merge.to_csv(OUT, index=False)
print(f"\nSaved per-cluster comparison to {OUT}")
