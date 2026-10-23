"""05.08 Multiple testing and comparing ML models.

Run:  python curriculum/05-inference/08-multiple-testing-and-model-comparison/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(58)

print("Worked example 1: the family-wise error rate grows with the number of tests")
for m in [1, 5, 10, 20, 100]:
    print(f"  {m:>3} independent tests at α = 0.05: P(at least one false positive) = 1 - 0.95^{m} = {1 - 0.95 ** m:.3f}")

print("\nWorked example 2: 100 tests on pure noise")
p_noise = np.array([stats.ttest_ind(rng.normal(size=30), rng.normal(size=30)).pvalue for _ in range(100)])
print(f"  'significant' at 0.05 with no correction: {(p_noise < 0.05).sum()}")
print(f"  after Bonferroni (p < 0.05/100):           {(p_noise < 0.0005).sum()}")

print("\nWorked example 3: Bonferroni, Holm and Benjamini-Hochberg on 10 p-values")
p = np.array([0.001, 0.008, 0.012, 0.039, 0.041, 0.060, 0.074, 0.205, 0.212, 0.906])
m = len(p)
print("  p-values:", p.tolist())
for method, label in [("bonferroni", "Bonferroni (FWER)"), ("holm", "Holm (FWER)"), ("fdr_bh", "Benjamini-Hochberg (FDR)")]:
    reject, adjusted, _, _ = multipletests(p, alpha=0.05, method=method)
    print(f"  {label:<26} rejects {reject.sum()}: adjusted p = {np.round(adjusted, 3).tolist()}")
order = np.argsort(p)
print("  BH by hand: compare the k-th smallest p with (k/m)·q")
for k, i in enumerate(order, start=1):
    print(f"    k = {k:>2}: p = {p[i]:.3f}  threshold {k / m * 0.05:.4f}  {'<=' if p[i] <= k / m * 0.05 else '> '}")

print("\nWorked example 4: BH controls the FALSE DISCOVERY RATE — 900 nulls + 100 real effects")
fdp_none, fdp_bh, power_none, power_bh = [], [], [], []
for _ in range(200):
    truth = np.r_[np.zeros(900, bool), np.ones(100, bool)]
    z = rng.normal(size=1000) + np.where(truth, 3.0, 0.0)
    pv = 2 * stats.norm.sf(np.abs(z))
    for reject, fdp_list, pow_list in [(pv < 0.05, fdp_none, power_none),
                                       (multipletests(pv, 0.05, "fdr_bh")[0], fdp_bh, power_bh)]:
        fdp_list.append((reject & ~truth).sum() / max(reject.sum(), 1))
        pow_list.append((reject & truth).mean() / truth.mean())
print(f"  no correction: false discovery proportion {np.mean(fdp_none):.3f}, share of real effects found {np.mean(power_none):.3f}")
print(f"  BH at q = 0.05: false discovery proportion {np.mean(fdp_bh):.3f}, share of real effects found {np.mean(power_bh):.3f}")

print("\nWorked example 5: p-hacking by optional stopping — peek after every 10 subjects, stop when p < 0.05")
false_pos = 0
for _ in range(5_000):
    a, b = rng.normal(size=200), rng.normal(size=200)        # NO true effect
    for n in range(10, 201, 10):
        if stats.ttest_ind(a[:n], b[:n]).pvalue < 0.05:
            false_pos += 1
            break
print(f"  false-positive rate with peeking: {false_pos / 5_000:.3f} (nominal 0.05)")

print("\nWorked example 6: comparing two classifiers on the SAME test set")
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
c_lr = lr.predict(X_te) == y_te
c_tree = tree.predict(X_te) == y_te
print(f"  accuracy: logistic {c_lr.mean():.4f}, tree {c_tree.mean():.4f}, n = {len(y_te)}")
table = np.array([[np.sum(c_lr & c_tree), np.sum(c_lr & ~c_tree)],
                  [np.sum(~c_lr & c_tree), np.sum(~c_lr & ~c_tree)]])
print("  2x2 table (rows: logistic right/wrong, columns: tree right/wrong):")
print("   ", table.tolist())
b_, c_ = table[0, 1], table[1, 0]
print(f"  only the {b_ + c_} disagreements matter: logistic-only right {b_}, tree-only right {c_}")
print(f"  McNemar exact test p = {mcnemar(table, exact=True).pvalue:.4f}")
idx = rng.integers(0, len(y_te), size=(10_000, len(y_te)))
diffs = c_lr[idx].mean(axis=1) - c_tree[idx].mean(axis=1)
print(f"  paired bootstrap 95% interval for the accuracy difference: [{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]")

print("\nWorked example 7: why a naive t-test across CV folds is overconfident")
# Two models with IDENTICAL true accuracy. The 10 folds reuse almost the same training data, so every fold's
# difference shares one common random component (how lucky this particular dataset happens to be for model A),
# plus independent per-fold noise. The naive t-test treats the 10 differences as independent.
false_pos = 0
for _ in range(2_000):
    common = rng.normal(0, 0.012)              # one draw shared by all 10 folds
    per_fold = rng.normal(0, 0.02, 10)         # independent fold-to-fold noise
    diff = common + per_fold
    if stats.ttest_1samp(diff, 0).pvalue < 0.05:
        false_pos += 1
print(f"  naive paired t-test over 10 correlated folds rejects a TRUE null {false_pos / 2_000:.3f} of the time")
print("  (use the Nadeau-Bengio corrected resampled t-test, 5x2cv, or repeated CV with a correction)")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ms = np.arange(1, 101)
ax.plot(ms, 1 - 0.95 ** ms)
ax.axhline(0.05, color="grey", ls="--", lw=0.8)
ax.set_title("P(at least one false positive) with m tests at α = 0.05")
ax.set_xlabel("number of independent tests m")
ax.set_ylabel("family-wise error rate")
fig.tight_layout()
fig.savefig(FIGURES / "fwer.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "fwer.png")
