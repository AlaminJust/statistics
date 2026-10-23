"""04.04 Law of large numbers and the central limit theorem — simulations.

Run:  python curriculum/04-distributions/04-lln-and-clt/examples.py
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
rng = np.random.default_rng(44)

print("1. Law of large numbers: running mean of exponential(mean 25) draws")
x = rng.exponential(25, size=100_000)
running = np.cumsum(x) / np.arange(1, len(x) + 1)
for n in [10, 100, 1_000, 10_000, 100_000]:
    print(f"   n = {n:>7}: running mean {running[n - 1]:7.3f}   (true mean 25)")

print("\n   Cauchy: no mean exists, so the running mean never settles")
c = rng.standard_cauchy(100_000)
c_running = np.cumsum(c) / np.arange(1, len(c) + 1)
for n in [10, 100, 1_000, 10_000, 100_000]:
    print(f"   n = {n:>7}: running mean {c_running[n - 1]:8.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(np.arange(1, len(x) + 1), running)
axes[0].axhline(25, color="black", ls="--", lw=0.8)
axes[0].set_xscale("log")
axes[0].set_title("Exponential: the running mean converges to 25")
axes[1].plot(np.arange(1, len(c) + 1), c_running, color="tab:red")
axes[1].set_xscale("log")
axes[1].set_title("Cauchy: the running mean never settles")
for ax in axes:
    ax.set_xlabel("number of draws (log scale)")
    ax.set_ylabel("running mean")
fig.tight_layout()
fig.savefig(FIGURES / "lln_running_mean.png", dpi=120)
plt.close(fig)

print("\n2. Central limit theorem: means of n exponential(mean 1) draws, 100,000 times")
print("   theory: mean 1, sd 1/sqrt(n), skewness 2/sqrt(n)")
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
for ax, n in zip(axes, [1, 2, 5, 30]):
    means = rng.exponential(1.0, size=(100_000, n)).mean(axis=1)
    z = (means - 1) / (1 / math.sqrt(n))
    lower, upper = (z < -1.96).mean(), (z > 1.96).mean()
    print(f"   n = {n:>2}: sd {means.std():.3f} (theory {1 / math.sqrt(n):.3f}), "
          f"skew {stats.skew(means):.3f} (theory {2 / math.sqrt(n):.3f}), "
          f"P(Z < -1.96) = {lower:.4f}, P(Z > 1.96) = {upper:.4f} (normal: 0.025 each)")
    ax.hist(means, bins=80, density=True, alpha=0.6)
    grid = np.linspace(means.min(), means.max(), 300)
    ax.plot(grid, stats.norm(1, 1 / math.sqrt(n)).pdf(grid), color="black", lw=1)
    ax.set_title(f"mean of n = {n}")
    ax.set_xlabel("sample mean")
axes[0].set_ylabel("density")
fig.suptitle("Means of skewed data become normal as n grows (black = CLT normal curve)")
fig.tight_layout()
fig.savefig(FIGURES / "clt_exponential.png", dpi=120)
plt.close(fig)

print("\n3. The CLT for a proportion: share of clicks when p = 0.03")
for n in [30, 300, 3000]:
    shares = rng.binomial(n, 0.03, size=100_000) / n
    sd = math.sqrt(0.03 * 0.97 / n)
    print(f"   n = {n:>4}: sd of share {shares.std():.4f} (theory {sd:.4f}), skew {stats.skew(shares):.3f}, "
          f"P(share = 0) = {(shares == 0).mean():.4f}")

print("\n4. Concentration inequalities for an accuracy estimate (each test example correct with p = 0.9)")
n, eps = 1000, 0.03
markov_note = "Markov needs only E[X]; it is too weak here"
chebyshev = (0.9 * 0.1 / n) / eps ** 2
hoeffding = 2 * math.exp(-2 * n * eps ** 2)
acc = rng.binomial(n, 0.9, size=200_000) / n
actual = (np.abs(acc - 0.9) >= eps).mean()
print(f"   P(|accuracy - 0.9| >= 0.03) with n = 1000:")
print(f"   Chebyshev bound {chebyshev:.4f}   Hoeffding bound {hoeffding:.4f}   actual (simulated) {actual:.4f}")
print(f"   ({markov_note})")
eps, delta = 0.02, 0.05
n_needed = math.log(2 / delta) / (2 * eps ** 2)
print(f"   test-set size for ±{eps} with 95% confidence by Hoeffding: n >= ln(2/δ)/(2ε²) = {math.ceil(n_needed)}")
n_normal = (1.96 / eps) ** 2 * 0.25
print(f"   the CLT (worst case p = 0.5) gives n >= (1.96/ε)² × 0.25 = {math.ceil(n_normal)}")
print("\nFigures saved in", FIGURES)
