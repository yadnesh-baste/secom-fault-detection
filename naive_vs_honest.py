"""Why many SECOM write-ups look better than reality: accuracy on imbalanced data and shuffled splits."""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
exec(open("analysis.py").read().split("# Rolling time-based validation")[0])
m = models["LightGBM"]
p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
out = dict(always_pass_accuracy=float(1-y.mean()), shuffled_cv_auc=float(roc_auc_score(y, p)))
print(out); json.dump(out, open("naive.json", "w"))
