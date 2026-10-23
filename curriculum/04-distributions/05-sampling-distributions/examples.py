"""04.05 Sampling distributions — standard errors, chi-square, t and F, all checked by simulation.

Run:  python curriculum/04-distributions/05-sampling-distributions/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(45)
mu, sigma, n = 64, 14, 10
samples = rng.normal(mu, sigma, size=(200_000, n))
xbar = samples.mean(axis=1)
s = samples.std(axis=1, ddof=1)

print("1. Sampling distribution of the mean (normal data, n = 10)")
print(f"   sd of x-bar: simulated {xbar.std():.3f}, theory sigma/sqrt(n) = {sigma / math.sqrt(n):.3f}")
print(f"   one sample's estimated standard error s/sqrt(n): first five {np.round(s[:5] / math.sqrt(n), 2)}")

print("\n2. Chi-square: (n-1) s^2 / sigma^2 ~ chi-square with n-1 df")
chi = (n - 1) * s ** 2 / sigma ** 2
print(f"   simulated mean {chi.mean():.3f} (theory df = {n - 1}), variance {chi.var():.3f} (theory 2df = {2 * (n - 1)})")
print(f"   P(chi > 16.92): simulated {(chi > 16.92).mean():.4f}, chi2.sf = {stats.chi2.sf(16.92, n - 1):.4f}")
print(f"   so s^2 has sd {sigma ** 2 * math.sqrt(2 / (n - 1)):.1f} around sigma^2 = {sigma ** 2}: variance estimates are noisy")

print("\n3. Student's t: (x-bar - mu) / (s / sqrt(n)) ~ t with n-1 df")
t_stat = (xbar - mu) / (s / math.sqrt(n))
z_stat = (xbar - mu) / (sigma / math.sqrt(n))
print(f"   P(|stat| > 1.96) using the TRUE sigma: {(np.abs(z_stat) > 1.96).mean():.4f}  (normal says 0.05)")
print(f"   P(|stat| > 1.96) using s instead     : {(np.abs(t_stat) > 1.96).mean():.4f}  (too many! heavier tails)")
t_crit = stats.t.ppf(0.975, n - 1)
print(f"   P(|stat| > t_crit = {t_crit:.3f})        : {(np.abs(t_stat) > t_crit).mean():.4f}")
print("   97.5% critical values of t by df:")
for df_ in [1, 2, 4, 9, 19, 29, 99, 1000]:
    print(f"     df = {df_:>4}: {stats.t.ppf(0.975, df_):.3f}")
print(f"     normal   : {stats.norm.ppf(0.975):.3f}")

print("\n4. F: ratio of two sample variances, (s1^2/sigma1^2) / (s2^2/sigma2^2) ~ F(n1-1, n2-1)")
a = rng.normal(0, 3, size=(200_000, 8)).var(axis=1, ddof=1)
b = rng.normal(0, 3, size=(200_000, 13)).var(axis=1, ddof=1)
ratio = a / b
crit = stats.f.ppf(0.95, 7, 12)
print(f"   equal true variances: P(F > {crit:.3f}) simulated {(ratio > crit).mean():.4f} (theory 0.05)")
print(f"   one sample variance can easily be {np.percentile(ratio, 95):.2f}x another by chance alone (95th percentile)")

print("\n5. Sampling distribution of a proportion: p = 0.3, n = 200")
p_hat = rng.binomial(200, 0.3, size=200_000) / 200
print(f"   sd of p-hat: simulated {p_hat.std():.4f}, theory sqrt(p(1-p)/n) = {math.sqrt(0.3 * 0.7 / 200):.4f}")

print("\n6. Sampling distribution of the median (normal data, n = 101)")
med = np.median(rng.normal(0, 1, size=(50_000, 101)), axis=1)
mean = rng.normal(0, 1, size=(50_000, 101)).mean(axis=1)
print(f"   sd of median {med.std():.4f} vs sd of mean {mean.std():.4f}; ratio {med.std() / mean.std():.3f} "
      f"(theory sqrt(pi/2) = {math.sqrt(math.pi / 2):.3f})")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
grid = np.linspace(0, 30, 300)
axes[0].hist(chi, bins=100, density=True, alpha=0.5)
axes[0].plot(grid, stats.chi2.pdf(grid, n - 1), color="black")
axes[0].set_title("(n−1)s²/σ² vs χ²(9)")
grid = np.linspace(-5, 5, 300)
axes[1].hist(t_stat, bins=200, range=(-5, 5), density=True, alpha=0.5)
axes[1].plot(grid, stats.t.pdf(grid, n - 1), color="black", label="t(9)")
axes[1].plot(grid, stats.norm.pdf(grid), color="tab:red", ls="--", label="N(0, 1)")
axes[1].set_title("t statistic: heavier tails than normal")
axes[1].legend()
grid = np.linspace(0, 5, 300)
axes[2].hist(ratio, bins=200, range=(0, 5), density=True, alpha=0.5)
axes[2].plot(grid, stats.f.pdf(grid, 7, 12), color="black")
axes[2].set_title("Variance ratio vs F(7, 12)")
fig.tight_layout()
fig.savefig(FIGURES / "chi2_t_f.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "chi2_t_f.png")
