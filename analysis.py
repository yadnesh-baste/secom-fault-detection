"""SECOM: predicting failing semiconductor production runs from 590 sensor signals.
Time-ordered split: train on the first 70% of runs, test on the last 30% (what a fab would actually face)."""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
import lightgbm as lgb

df = pd.read_csv("data/uci-secom.csv", parse_dates=["Time"])
# This CSV copy swapped day and month whenever the day was <= 12 (the real data runs Jul-Oct 2008).
# Repair those dates; the file's row order is already chronological, so we keep it.
bad = ~df.Time.dt.month.isin([7, 8, 9, 10])
df.loc[bad, "Time"] = pd.to_datetime(dict(year=df.Time[bad].dt.year, month=df.Time[bad].dt.day,
    day=df.Time[bad].dt.month, hour=df.Time[bad].dt.hour, minute=df.Time[bad].dt.minute))
y = (df["Pass/Fail"] == 1).astype(int).values
X = df.drop(columns=["Time", "Pass/Fail"])

class DropSparse(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing=0.5): self.max_missing = max_missing
    def fit(self, X, y=None):
        self.keep_ = X.columns[X.isna().mean() <= self.max_missing]; return self
    def transform(self, X): return X[self.keep_]

class DropCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, thr=0.95): self.thr = thr
    def fit(self, X, y=None):
        c = np.abs(np.corrcoef(X, rowvar=False)); c = np.nan_to_num(c)
        drop = set()
        for i in range(c.shape[0]):
            if i in drop: continue
            for j in range(i+1, c.shape[0]):
                if c[i, j] > self.thr: drop.add(j)
        self.keep_ = [i for i in range(c.shape[0]) if i not in drop]; return self
    def transform(self, X): return X[:, self.keep_]

def prep(): return [("sparse", DropSparse()), ("impute", SimpleImputer(strategy="median")),
                    ("const", VarianceThreshold(1e-8)), ("corr", DropCorrelated())]

models = {
 "Logistic regression": Pipeline(prep()+[("scale", StandardScaler()),
     ("clf", LogisticRegression(C=0.01, penalty="l2", class_weight="balanced", max_iter=5000))]),
 "Random forest": Pipeline(prep()+[("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
     class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
 "LightGBM": Pipeline(prep()+[("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
     min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
     scale_pos_weight=(y==0).sum()/(y==1).sum(), random_state=42, verbose=-1))]),
}

# Rolling time-based validation: each fold trains on all earlier runs and scores the next block.
tss = TimeSeriesSplit(n_splits=5)
oof = {k: np.full(len(y), np.nan) for k in models}
for tr, te in tss.split(X):
    for name, m in models.items():
        m.fit(X.iloc[tr], y[tr]); oof[name][te] = m.predict_proba(X.iloc[te])[:, 1]
mask = ~np.isnan(oof["LightGBM"]); yt = y[mask]
def recall_at(p, frac):
    k = int(np.ceil(frac*len(p))); return yt[np.argsort(-p)[:k]].sum()/yt.sum()
res = {}
for name in models:
    p = oof[name][mask]
    res[name] = dict(roc_auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p),
                     recall_at_10=recall_at(p, .10), recall_at_20=recall_at(p, .20), recall_at_30=recall_at(p, .30))
# Bootstrap 90% interval for the best model's ROC-AUC (few failures -> report uncertainty)
rng = np.random.default_rng(0); pb = oof["LightGBM"][mask]; boots = []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    if yt[i].sum() > 0: boots.append(roc_auc_score(yt[i], pb[i]))
# Feature importance from a model fit on all data
final = models["LightGBM"].fit(X, y)
kept_cols = final.named_steps["sparse"].keep_
after_const = kept_cols[final.named_steps["const"].get_support()]
after_corr = after_const[final.named_steps["corr"].keep_]
imp = pd.Series(final.named_steps["clf"].booster_.feature_importance("gain"), index=after_corr).sort_values(ascending=False)
summary = dict(n_runs=len(df), n_features_raw=X.shape[1], fail_rate=float(y.mean()), n_fails=int(y.sum()),
    period=[str(df.Time.min().date()), str(df.Time.max().date())],
    features_after_prep=len(after_corr), evaluated_runs=int(mask.sum()), evaluated_fails=int(yt.sum()),
    lgbm_auc_90ci=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
    top_sensors={f"sensor_{k}": float(v) for k, v in imp.head(10).items()}, models=res)
print(json.dumps(summary, indent=1, default=float))
json.dump(summary, open("results.json", "w"), indent=1, default=float)
np.save("oof.npy", np.vstack([oof[k][mask] for k in models])); np.save("y_eval.npy", yt)
df.loc[mask, ["Time"]].assign(y=yt).to_csv("eval_index.csv", index=False)
