"""
Compare predicted probability scales across saved model artifacts.

Print classifier type, overall mean predicted probability on X_train,
and per-cluster means via SegmentationEngine.compare_with_supervised_models.
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils
from src.risk_scoring import RiskScoringEngine
from src.segmentation_engine import SegmentationEngine

model_keys = ["logistic_regression", "random_forest", "xgboost"]

X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()
engine = SegmentationEngine()
engine.fit(X_train, default_flags=y_train)

for key in model_keys:
    print(f"\n--- {key} ---")
    rs = RiskScoringEngine(model_key=key)
    clf = rs.pipeline.named_steps['classifier']
    print("Classifier:", type(clf))
    try:
        proba = rs.predict_probability(X_train)
        print("Overall mean predicted probability:", proba.mean())
    except Exception as e:
        print("Error predicting on X_train:", e)
    comp = engine.compare_with_supervised_models(model_key=key)
    print(comp[["segment_name", "mean_predicted_probability", "average_default_rate"]].to_string(index=False))
