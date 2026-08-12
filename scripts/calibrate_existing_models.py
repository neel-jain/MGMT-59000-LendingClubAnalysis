"""Calibrate existing trained model pipelines on the saved validation split.

This script loads `models/*.joblib` (logistic, random forest, xgboost),
wraps each with `CalibratedClassifierCV(estimator=..., cv='prefit')`, fits
on `data/splits/X_val.csv` / `y_val.csv`, and saves a calibrated version
back to the canonical model path while keeping a `_uncalibrated` backup.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, utils

MODEL_FILES = {
    'logistic_regression': config.LOGISTIC_REGRESSION_MODEL_PATH,
    'random_forest': config.RANDOM_FOREST_MODEL_PATH,
    'xgboost': config.XGBOOST_MODEL_PATH,
}

if __name__ == '__main__':
    X_val = pd.read_csv(config.X_VAL_PATH)
    y_val = pd.read_csv(config.Y_VAL_PATH).squeeze()

    for key, path in MODEL_FILES.items():
        if not path.exists():
            print('Skipping', key, '- model file not found at', path)
            continue
        print('\nCalibrating', key, 'from', path)
        pipeline = joblib.load(path)
        try:
            calibrator = CalibratedClassifierCV(estimator=pipeline, method=config.CALIBRATION_METHOD, cv='prefit')
            calibrator.fit(X_val, y_val)
        except Exception as exc:
            print('  Calibration failed for', key, exc)
            continue
        # backup
        backup = path.with_name(path.stem + '_uncalibrated' + path.suffix)
        joblib.dump(pipeline, backup)
        joblib.dump(calibrator, path)
        print('  Saved calibrated model to', path)
        print('  Backup of uncalibrated saved to', backup)
    print('\nDone.')
