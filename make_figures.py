"""Draws the README figures from results.json and the saved predictions. Run after analysis.py."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, SLATE, RULE, SIG, BLUE, TEAL = "#16233B", "#5B6878", "#D5DBE0", "#C77F00", "#1F5FAD", "#2E8B7A"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.color": "#EEF1F3"})

r = json.load(open("results.json")); nv = json.load(open("naive.json"))
oof = np.load("oof.npy"); yt = np.load("y_eval.npy"); names = list(r["models"])
curves = {}
fig, ax = plt.subplots(figsize=(7, 4.4))
for nm, p, c in zip(names, oof, [SLATE, TEAL, BLUE]):
    o = np.argsort(-p); cum = np.concatenate([[0], np.cumsum(yt[o])/yt.sum()*100]); x = np.linspace(0, 100, len(cum))
    ax.plot(x, cum, color=c, lw=2.2 if nm == "LightGBM" else 1.6, label=f'{nm} (AUC {r["models"][nm]["roc_auc"]:.2f})')
    idx = np.linspace(0, len(cum)-1, 41).astype(int); curves[nm] = [[round(float(x[i]), 1), round(float(cum[i]), 1)] for i in idx]
ax.plot([0, 100], [0, 100], ls="--", color=RULE, label="Random inspection"); ax.axvline(10, color=SIG, lw=1, ls=":")
ax.set(xlabel="% of production runs inspected (highest risk first)", ylabel="% of failing runs caught", xlim=(0, 100), ylim=(0, 100))
ax.set_title("Inspecting the riskiest runs first catches failures sooner"); ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig("figures/fig_inspection_curve.png"); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.2))
lab = ["Shuffled 5-fold CV\n(common in tutorials)", "Rolling time-based CV\n(what a fab would see)"]
v = [nv["shuffled_cv_auc"], r["models"]["LightGBM"]["roc_auc"]]
ax.barh(lab, v, color=[RULE, BLUE]); ax.axvline(0.5, color=SLATE, ls="--", lw=1); ax.set_xlim(0.4, 0.8)
for i, val in enumerate(v): ax.text(val+0.005, i, f"{val:.2f}", va="center", color=INK, fontweight="bold")
ax.set_xlabel("ROC-AUC, same LightGBM model (0.5 = coin flip)"); ax.set_title("Validation choice changes the answer"); ax.grid(axis="y")
fig.savefig("figures/fig_validation.png"); plt.close()
ts = r["top_sensors"]; fig, ax = plt.subplots(figsize=(7, 3.6)); k = list(ts)[::-1]; vv = np.array([ts[x] for x in k]); vv = vv/vv.max()
ax.barh([x.replace("sensor_", "Sensor ") for x in k], vv, color=BLUE); ax.set_xlabel("Relative importance (gain)"); ax.set_title("Signals to check first in root-cause analysis"); ax.grid(axis="y")
fig.savefig("figures/fig_top_sensors.png"); plt.close()
json.dump(curves, open("curves.json", "w"))
