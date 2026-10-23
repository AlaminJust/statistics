"""04.01 Discrete distributions — worked examples with formulas, scipy.stats and simulation.

Run:  python curriculum/04-distributions/01-discrete-distributions/examples.py
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
rng = np.random.default_rng(41)


def binomial_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


print("Worked example 1: Bernoulli(0.3)")
b = stats.bernoulli(0.3)
print(f"  mean {b.mean():.2f}, variance {b.var():.2f}")

print("\nWorked example 2: binomial — 10 free throws with p = 0.7")
n, p = 10, 0.7
print(f"  P(X = 7) by formula = C(10,7) 0.7^7 0.3^3 = {math.comb(10, 7)} * {0.7 ** 7:.5f} * {0.3 ** 3:.3f} = {binomial_pmf(7, n, p):.4f}")
print(f"  P(X = 7) scipy      = {stats.binom.pmf(7, n, p):.4f}")
print(f"  P(X >= 8)           = {stats.binom.sf(7, n, p):.4f}")
print(f"  P(X <= 5)           = {stats.binom.cdf(5, n, p):.4f}")
print(f"  mean np = {n * p:.1f}, variance np(1-p) = {n * p * (1 - p):.2f}, sd = {math.sqrt(n * p * (1 - p)):.3f}")
sims = rng.binomial(n, p, size=200_000)
print(f"  simulated P(X = 7) = {(sims == 7).mean():.4f}, mean {sims.mean():.3f}")
print("  full PMF:", {k: round(float(stats.binom.pmf(k, n, p)), 4) for k in range(n + 1)})

print("\nWorked example 3: geometric — rolls until the first six")
p6 = 1 / 6
print(f"  P(first six on roll 3) = (5/6)^2 (1/6) = {(5 / 6) ** 2 / 6:.4f}   scipy {stats.geom.pmf(3, p6):.4f}")
print(f"  P(need more than 10 rolls) = (5/6)^10 = {(5 / 6) ** 10:.4f}")
print(f"  mean 1/p = {1 / p6:.0f}, variance (1-p)/p^2 = {(1 - p6) / p6 ** 2:.0f}")

print("\nWorked example 4: Poisson — emails arrive at 4 per hour")
lam = 4
for k in [0, 2, 4, 8]:
    print(f"  P(X = {k}) = 4^{k} e^-4 / {k}! = {poisson_pmf(k, lam):.4f}")
print(f"  P(X <= 2) = {stats.poisson.cdf(2, lam):.4f}   P(X >= 8) = {stats.poisson.sf(7, lam):.4f}")
print(f"  mean = variance = {lam};  in 30 minutes the rate is 2: P(no email) = {math.exp(-2):.4f}")

print("\nWorked example 5: binomial(1000, 0.003) is almost Poisson(3)")
for k in range(6):
    print(f"  k = {k}: binomial {stats.binom.pmf(k, 1000, 0.003):.4f}   Poisson {stats.poisson.pmf(k, 3):.4f}")

print("\nWorked example 6: does real-looking count data fit a Poisson? (goals per match, simulated)")
goals = rng.poisson(2.6, size=380)
lam_hat = goals.mean()
observed = np.bincount(goals, minlength=9)[:9]
expected = [len(goals) * stats.poisson.pmf(k, lam_hat) for k in range(9)]
print(f"  estimated rate = mean = {lam_hat:.3f}; sample variance {goals.var(ddof=1):.3f} (Poisson: mean = variance)")
for k in range(9):
    print(f"  {k} goals: observed {observed[k]:>3}  expected {expected[k]:6.1f}")

print("\nWorked example 7: categorical and multinomial")
probs = [0.5, 0.3, 0.2]
draws = rng.choice(["A", "B", "C"], size=10, p=probs)
print("  10 categorical draws:", "".join(draws))
print(f"  P(5 A, 3 B, 2 C in 10 draws) = {stats.multinomial.pmf([5, 3, 2], n=10, p=probs):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
ks = np.arange(0, 11)
axes[0].bar(ks, stats.binom.pmf(ks, 10, 0.7))
axes[0].set_title("Binomial(10, 0.7): free throws made")
ks = np.arange(1, 21)
axes[1].bar(ks, stats.geom.pmf(ks, 1 / 6))
axes[1].set_title("Geometric(1/6): rolls until first six")
ks = np.arange(0, 13)
axes[2].bar(ks, stats.poisson.pmf(ks, 4))
axes[2].set_title("Poisson(4): emails per hour")
for ax in axes:
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
fig.tight_layout()
fig.savefig(FIGURES / "discrete_pmfs.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "discrete_pmfs.png")
