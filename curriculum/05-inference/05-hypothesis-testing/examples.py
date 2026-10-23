"""05.05 Hypothesis testing — the logic, p-values, errors and permutation tests.

Run:  python curriculum/05-inference/05-hypothesis-testing/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(55)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("Worked example 1: is the coin fair? 60 heads in 100 flips")
k, n = 60, 100
z = (k / n - 0.5) / math.sqrt(0.25 / n)
print(f"  z = (0.60 - 0.50) / sqrt(0.25/100) = {z:.2f}")
print(f"  two-sided p-value (normal) = 2 P(Z > {z:.0f}) = {2 * stats.norm.sf(z):.4f}")
print(f"  exact binomial test p-value = {stats.binomtest(k, n, 0.5).pvalue:.4f}")
print(f"  one-sided (H1: p > 0.5) exact p-value = {stats.binomtest(k, n, 0.5, alternative='greater').pvalue:.4f}")
sims = rng.binomial(100, 0.5, size=1_000_000)
print(f"  simulation under H0: P(|heads - 50| >= 10) = {(np.abs(sims - 50) >= 10).mean():.4f}")

print("\nWorked example 2: p-values are uniform when H0 is true")
p_null = np.array([stats.ttest_1samp(rng.normal(0, 1, 20), 0).pvalue for _ in range(20_000)])
for a in [0.01, 0.05, 0.10]:
    print(f"  share of p-values below {a}: {(p_null < a).mean():.4f}")
p_alt = np.array([stats.ttest_1samp(rng.normal(0.5, 1, 20), 0).pvalue for _ in range(20_000)])
print(f"  when the true mean is 0.5 (H0 false): share below 0.05 = {(p_alt < 0.05).mean():.4f}  <- this is the power")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
axes[0].hist(p_null, bins=20, range=(0, 1), edgecolor="white")
axes[0].set_title("H0 true: p-values are uniform")
axes[1].hist(p_alt, bins=20, range=(0, 1), edgecolor="white", color="tab:orange")
axes[1].set_title("H0 false (μ = 0.5, n = 20): p-values pile up near 0")
for ax in axes:
    ax.set_xlabel("p-value")
axes[0].set_ylabel("count of 20,000 experiments")
fig.tight_layout()
fig.savefig(FIGURES / "p_value_distributions.png", dpi=120)
plt.close(fig)

print("\nWorked example 3: a permutation test — do tutored students score higher?")
yes = df.loc[df["tutoring"] == "yes", "exam_score"].to_numpy()
no = df.loc[df["tutoring"] == "no", "exam_score"].to_numpy()
observed = yes.mean() - no.mean()
pooled = np.concatenate([yes, no])
perm = np.empty(20_000)
for i in range(len(perm)):
    shuffled = rng.permutation(pooled)
    perm[i] = shuffled[:len(yes)].mean() - shuffled[len(yes):].mean()
p_perm = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (len(perm) + 1)
print(f"  observed difference {observed:.2f}; under H0 the shuffled differences have sd {perm.std():.2f}")
print(f"  largest shuffled |difference| in 20,000: {np.abs(perm).max():.2f}")
print(f"  permutation p-value = {p_perm:.5f}  (Welch t-test p = {stats.ttest_ind(yes, no, equal_var=False).pvalue:.2e})")
res = stats.permutation_test((yes, no), lambda a, b: a.mean() - b.mean(), n_resamples=20_000, random_state=1)
print(f"  scipy.stats.permutation_test p-value = {res.pvalue:.5f}")

print("\n  The same test where there is NO true effect: program Science vs Arts")
sci = df.loc[df["program"] == "Science", "exam_score"].to_numpy()
art = df.loc[df["program"] == "Arts", "exam_score"].to_numpy()
res2 = stats.permutation_test((sci, art), lambda a, b: a.mean() - b.mean(), n_resamples=20_000, random_state=1)
print(f"  difference {sci.mean() - art.mean():.2f}, permutation p-value = {res2.pvalue:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(perm, bins=60, edgecolor="white")
ax.axvline(observed, color="tab:red", lw=2, label=f"observed {observed:.1f}")
ax.set_title("Permutation distribution of the difference under H0")
ax.set_xlabel("difference in mean score (shuffled labels)")
ax.set_ylabel("count")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "permutation_test.png", dpi=120)
plt.close(fig)

print("\nWorked example 4: confidence intervals and tests agree")
diff_ci = stats.ttest_ind(yes, no, equal_var=False).confidence_interval()
print(f"  95% interval for the difference [{diff_ci.low:.2f}, {diff_ci.high:.2f}] excludes 0  <=>  p < 0.05")

print("\nWorked example 5: Type I and Type II errors by simulation (one-sample t-test, α = 0.05, n = 20)")
for mu_true in [0.0, 0.2, 0.5, 0.8]:
    rejections = np.mean([stats.ttest_1samp(rng.normal(mu_true, 1, 20), 0).pvalue < 0.05 for _ in range(10_000)])
    label = "Type I error rate" if mu_true == 0 else f"power (Type II = {1 - rejections:.3f})"
    print(f"  true mean {mu_true}: P(reject H0) = {rejections:.4f}  <- {label}")
print("\nFigures saved in", FIGURES)
