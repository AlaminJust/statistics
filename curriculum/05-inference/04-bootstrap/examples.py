"""05.04 The bootstrap — standard errors and intervals for any statistic.

Run:  python curriculum/05-inference/04-bootstrap/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(54)
df = pd.read_csv(ROOT / "data" / "students.csv")
B = 10_000


def bootstrap(data, statistic, n_boot=B, rng=rng):
    data = np.asarray(data)
    n = len(data)
    idx = rng.integers(0, n, size=(n_boot, n))       # resample n rows WITH replacement, n_boot times
    return np.array([statistic(data[i]) for i in idx])


print("Worked example 1: a resample by hand")
small = np.array([3, 7, 8, 12, 20])
for i in range(3):
    print(f"  resample {i + 1}: {np.sort(rng.choice(small, size=5, replace=True))}")

print("\nWorked example 2: bootstrap the mean exam score — compare with the formula")
x = df["exam_score"].to_numpy()
boot_means = bootstrap(x, np.mean)
print(f"  bootstrap SE {boot_means.std(ddof=1):.3f}   formula s/√n = {x.std(ddof=1) / np.sqrt(len(x)):.3f}")
print(f"  percentile 95% interval [{np.percentile(boot_means, 2.5):.2f}, {np.percentile(boot_means, 97.5):.2f}]"
      f"   t-interval [62.00, 65.86]")

print("\nWorked example 3: the median commute — no simple formula")
c = df["commute_minutes"].to_numpy()
boot_med = bootstrap(c, np.median)
print(f"  median {np.median(c)}, bootstrap SE {boot_med.std(ddof=1):.2f}")
print(f"  percentile 95% interval [{np.percentile(boot_med, 2.5):.1f}, {np.percentile(boot_med, 97.5):.1f}]")
print(f"  distinct values among bootstrap medians: {len(np.unique(boot_med))} (medians of integer data are lumpy)")

print("\nWorked example 3b: the MEAN commute (skewed data) — t vs percentile vs BCa")
t_lo, t_hi = stats.t.interval(0.95, len(c) - 1, loc=c.mean(), scale=c.std(ddof=1) / np.sqrt(len(c)))
boot_cmean = bootstrap(c, np.mean)
res = stats.bootstrap((c,), np.mean, n_resamples=B, method="BCa", random_state=1)
print(f"  mean {c.mean():.2f}")
print(f"  t-interval          [{t_lo:.2f}, {t_hi:.2f}]  (symmetric)")
print(f"  percentile interval [{np.percentile(boot_cmean, 2.5):.2f}, {np.percentile(boot_cmean, 97.5):.2f}]")
print(f"  BCa interval        [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}]"
      "  (shifted right: corrects for skew)")

print("\nWorked example 4: correlation between study hours and score (resample PAIRS)")
pairs = df[["study_hours", "exam_score"]].to_numpy()
boot_r = bootstrap(pairs, lambda d: np.corrcoef(d[:, 0], d[:, 1])[0, 1])
r = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
print(f"  r = {r:.3f}, bootstrap SE {boot_r.std(ddof=1):.3f}, 95% interval "
      f"[{np.percentile(boot_r, 2.5):.3f}, {np.percentile(boot_r, 97.5):.3f}]")
fisher = np.tanh(np.arctanh(r) + np.array([-1, 1]) * 1.96 / np.sqrt(len(pairs) - 3))
print(f"  Fisher-z formula interval [{fisher[0]:.3f}, {fisher[1]:.3f}]")

print("\nWorked example 5: intervals for ML metrics — logistic regression on breast cancer data")
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
prob = model.predict_proba(X_te)[:, 1]
pred = (prob >= 0.5).astype(int)
acc, f1, auc = (pred == y_te).mean(), f1_score(y_te, pred), roc_auc_score(y_te, prob)
idx = rng.integers(0, len(y_te), size=(B, len(y_te)))
boot = np.array([[(pred[i] == y_te[i]).mean(), f1_score(y_te[i], pred[i]), roc_auc_score(y_te[i], prob[i])] for i in idx])
print(f"  test set n = {len(y_te)}")
for name, value, col in [("accuracy", acc, 0), ("F1", f1, 1), ("AUC", auc, 2)]:
    lo, hi = np.percentile(boot[:, col], [2.5, 97.5])
    print(f"  {name:<9} {value:.4f}   95% bootstrap interval [{lo:.4f}, {hi:.4f}]")

print("\nWorked example 6: where the bootstrap fails — the maximum of Uniform(0, 1)")
u = rng.uniform(0, 1, size=50)
boot_max = bootstrap(u, np.max)
print(f"  sample max {u.max():.4f}; share of bootstrap maxima EQUAL to the sample max: {(boot_max == u.max()).mean():.3f}")
print(f"  (theory: 1 - (1 - 1/n)^n ≈ {1 - (1 - 1 / 50) ** 50:.3f}); the bootstrap can never exceed the sample max")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, values, title in [(axes[0], boot_means, "mean exam score"), (axes[1], boot_med, "median commute (lumpy)"),
                          (axes[2], boot_r, "correlation study–score")]:
    ax.hist(values, bins=60, edgecolor="white")
    for q in np.percentile(values, [2.5, 97.5]):
        ax.axvline(q, color="black", ls="--", lw=0.8)
    ax.set_title(f"Bootstrap distribution: {title}")
    ax.set_xlabel("statistic in each resample")
fig.tight_layout()
fig.savefig(FIGURES / "bootstrap_distributions.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "bootstrap_distributions.png")
