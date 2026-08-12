import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

models_dir = Path(__file__).resolve().parents[1] / 'models'
for p in sorted(models_dir.glob('*.joblib')):
    print('FILE:', p.name)
    obj = joblib.load(p)
    print('  type:', type(obj))
    if isinstance(obj, CalibratedClassifierCV):
        print('  -> CalibratedClassifierCV')
    if isinstance(obj, Pipeline):
        try:
            clf = obj.named_steps.get('classifier')
            print('  -> Pipeline with classifier type:', type(clf))
            if isinstance(clf, CalibratedClassifierCV):
                print('     -> classifier is CalibratedClassifierCV')
        except Exception as e:
            print('  -> Pipeline inspect error:', e)
    print()
