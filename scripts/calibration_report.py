"""
Compute Brier score and ECE for each saved model on the test split.
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils
from src.risk_scoring import RiskScoringEngine
from src.model_utils import expected_calibration_error
from sklearn.metrics import brier_score_loss

X_train, X_val, X_test, y_train, y_val, y_test = utils.load_splits()

for key in ("logistic_regression", "random_forest", "xgboost"):
    print(f"\n--- {key} ---")
    rs = RiskScoringEngine(model_key=key)
    y_proba = rs.predict_probability(X_test)
    brier = brier_score_loss(y_test, y_proba)
    ece = expected_calibration_error(y_test.values, y_proba)
    print(f"Mean proba (test): {y_proba.mean():.6f}")
    print(f"Brier: {brier:.6f}, ECE: {ece:.6f}")
