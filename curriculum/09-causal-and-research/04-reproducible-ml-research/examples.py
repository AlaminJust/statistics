"""09.04 Reproducible ML research — seed variance, honest comparisons, ablations, aggregate metrics.

Run:  python curriculum/09-causal-and-research/04-reproducible-ml-research/examples.py   (about 1 minute)
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(94)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def mlp(seed, hidden=(32,), alpha=1e-4, scale=True):
    steps = ([StandardScaler()] if scale else []) + [MLPClassifier(hidden, alpha=alpha, max_iter=300, random_state=seed)]
    return make_pipeline(*steps).fit(X_tr, y_tr).score(X_te, y_te)


print("Worked example 1: the same model, 20 random seeds")
seeds_a = np.array([mlp(s) for s in range(20)])
print(f"  test accuracy: min {seeds_a.min():.4f}, max {seeds_a.max():.4f}, mean {seeds_a.mean():.4f}, sd {seeds_a.std(ddof=1):.4f}")
ci = stats.t.interval(0.95, len(seeds_a) - 1, loc=seeds_a.mean(), scale=seeds_a.std(ddof=1) / np.sqrt(len(seeds_a)))
print(f"  report: {seeds_a.mean():.4f} ± {seeds_a.std(ddof=1):.4f} (sd over 20 seeds), 95% CI for the mean [{ci[0]:.4f}, {ci[1]:.4f}]")

print("\nWorked example 2: 'best of 5 seeds' vs an honest comparison")
seeds_b = np.array([mlp(s, hidden=(32,), alpha=1e-3) for s in range(20)])      # a 'new method' with essentially no real gain
print(f"  baseline mean {seeds_a.mean():.4f}; 'new method' mean {seeds_b.mean():.4f}")
print(f"  'new' best-of-first-5 {seeds_b[:5].max():.4f} vs baseline first seed {seeds_a[0]:.4f} -> "
      f"looks like +{100 * (seeds_b[:5].max() - seeds_a[0]):.1f} points")
t_res = stats.ttest_rel(seeds_b, seeds_a)
print(f"  paired over 20 seeds (same seed for both): mean difference {np.mean(seeds_b - seeds_a):+.4f}, p = {t_res.pvalue:.2f}")

print("\nWorked example 3: how many seeds do you need? (detect a 0.5-point gain with seed sd ≈ the observed sd)")
sd = seeds_a.std(ddof=1)
for k in [3, 5, 10, 20, 50]:
    se = sd * np.sqrt(2 / k)
    power = stats.norm.sf(1.96 - 0.005 / se)
    print(f"  {k:>2} seeds per method: SE of the difference {se:.4f}, power to detect +0.005 ≈ {power:.2f}")

print("\nWorked example 4: an ablation study — what does each component contribute?")
configs = {
    "full model (scaling + 64 units + weight decay)": dict(hidden=(64,), alpha=1e-2, scale=True),
    "  - without weight decay": dict(hidden=(64,), alpha=1e-6, scale=True),
    "  - with 16 hidden units": dict(hidden=(16,), alpha=1e-2, scale=True),
    "  - without input scaling": dict(hidden=(64,), alpha=1e-2, scale=False),
}
for name, cfg in configs.items():
    acc = np.array([mlp(s, **cfg) for s in range(8)])
    print(f"  {name:<48} {acc.mean():.4f} ± {acc.std(ddof=1):.4f}")

print("\nWorked example 5: comparing methods across several datasets")
datasets = {"digits": load_digits(return_X_y=True), "breast cancer": load_breast_cancer(return_X_y=True),
            "wine": load_wine(return_X_y=True)}
results = {}
for name, (Xd, yd) in datasets.items():
    a_scores, b_scores = [], []
    for s in range(10):
        Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.3, random_state=s, stratify=yd)
        a_scores.append(make_pipeline(StandardScaler(), MLPClassifier((32,), max_iter=500, random_state=s)).fit(Xa, ya).score(Xb, yb))
        b_scores.append(RandomForestClassifier(200, random_state=s, n_jobs=-1).fit(Xa, ya).score(Xb, yb))
    results[name] = (np.array(a_scores), np.array(b_scores))
    diff = np.array(b_scores) - np.array(a_scores)
    print(f"  {name:<14} MLP {np.mean(a_scores):.4f}  forest {np.mean(b_scores):.4f}  difference {diff.mean():+.4f} "
          f"(middle 95% of the 10 splits: {np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f})")

print("\nWorked example 6: robust aggregates — interquartile mean and a stratified bootstrap (rliable style)")


def iqm(scores):
    s = np.sort(np.asarray(scores).ravel())
    k = len(s) // 4
    return s[k:len(s) - k].mean()


for method, col in [("MLP", 0), ("forest", 1)]:
    runs = np.array([results[d][col] for d in datasets])           # shape (datasets, runs)
    boots = []
    for _ in range(2000):
        sample = np.array([row[rng.integers(0, len(row), len(row))] for row in runs])    # resample runs within each dataset
        boots.append(iqm(sample))
    print(f"  {method:<6} IQM {iqm(runs):.4f} (95% CI {np.percentile(boots, 2.5):.4f} to {np.percentile(boots, 97.5):.4f});"
          f" mean {runs.mean():.4f}; median {np.median(runs):.4f}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(seeds_a, bins=12, alpha=0.6, label="baseline (20 seeds)")
ax.hist(seeds_b, bins=12, alpha=0.6, label="'new method' (20 seeds)")
ax.set_title("Seed-to-seed variation is as large as many claimed gains")
ax.set_xlabel("test accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "seed_variance.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "seed_variance.png")
