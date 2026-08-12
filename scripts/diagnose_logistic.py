"""
Compute per-cluster mean predicted probability and compare to average default rate
for the `logistic_regression` model. Prints pipeline info and correlation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.common import get_segmentation_engine, load_splits_cached, get_risk_engine
import numpy as np
import pandas as pd

engine = get_segmentation_engine()
if engine is None:
    print('Segmentation engine missing')
    raise SystemExit(1)

rs = get_risk_engine('logistic_regression')
if rs is None:
    print('Logistic risk engine not available (artifact missing)')
    raise SystemExit(1)

print('Loaded risk engine model_key=', rs.model_key)
print('Pipeline type:', type(rs.pipeline))
if hasattr(rs.pipeline, 'named_steps'):
    for name, step in rs.pipeline.named_steps.items():
        print(' -', name, type(step))

X_train, X_val, X_test, y_train, y_val, y_test = load_splits_cached()
X = engine._X_train_raw.reset_index(drop=True)
proba = rs.predict_probability(X)
print('Predictions computed:', proba.shape, 'mean=', np.nanmean(proba))

# attach y
y = y_train.reset_index(drop=True)

df = X.copy()
df['cluster'] = engine.fit_result.labels
if len(y)==len(df):
    df['default_flag']=y
else:
    df['default_flag']=pd.NA

df['pred_proba']=proba

grp = df.groupby('cluster').agg(
    n_borrowers=('pred_proba','size'),
    mean_predicted_probability=('pred_proba','mean'),
    std_predicted_probability=('pred_proba','std'),
    average_default_rate=('default_flag','mean')
)
print('\nPer-cluster:')
print(grp)
print('\nCorrelation:', grp['mean_predicted_probability'].corr(grp['average_default_rate']))

# Save
out = Path('reports') / 'segment_prediction_logistic_diagnostics.csv'
out.parent.mkdir(exist_ok=True)
grp.to_csv(out)
print('Saved', out)
