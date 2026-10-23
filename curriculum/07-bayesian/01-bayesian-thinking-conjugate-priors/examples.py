"""07.01 Bayesian thinking and conjugate priors — exact posteriors you can compute by hand.

Run:  python curriculum/07-bayesian/01-bayesian-thinking-conjugate-priors/examples.py
"""

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
rng = np.random.default_rng(71)

print("Worked example 1: a coin, 7 heads in 10 flips, uniform Beta(1, 1) prior")
a0, b0, k, n = 1, 1, 7, 10
post = stats.beta(a0 + k, b0 + n - k)
print(f"  posterior Beta({a0 + k}, {b0 + n - k}): mean {post.mean():.4f}, mode {(a0 + k - 1) / (a0 + b0 + n - 2):.4f} (= MLE)")
print(f"  95% credible interval [{post.ppf(0.025):.3f}, {post.ppf(0.975):.3f}];  P(p > 0.5 | data) = {post.sf(0.5):.4f}")

print("\nGrid check: posterior ∝ likelihood × prior, normalised numerically")
grid = np.linspace(0, 1, 10_001)
unnorm = grid ** k * (1 - grid) ** (n - k) * stats.beta(a0, b0).pdf(grid)
grid_post = unnorm / np.trapezoid(unnorm, grid)
print(f"  grid posterior mean {np.trapezoid(grid * grid_post, grid):.4f}  (exact {post.mean():.4f})")

print("\nWorked example 2: the prior matters when data are few")
priors = {"flat Beta(1,1)": (1, 1), "sceptical Beta(20,20)": (20, 20), "optimistic Beta(8,2)": (8, 2)}
for label, (a, b) in priors.items():
    for kk, nn in [(7, 10), (70, 100), (700, 1000)]:
        p_ = stats.beta(a + kk, b + nn - kk)
        print(f"  {label:<22} data {kk:>3}/{nn:<4}: posterior mean {p_.mean():.3f}, 95% CI [{p_.ppf(.025):.3f}, {p_.ppf(.975):.3f}]")

print("\nWorked example 3: the posterior mean is a weighted average of prior mean and data")
a, b = 20, 20
w = n / (a + b + n)
print(f"  with Beta(20,20): posterior mean = {w:.2f} × {k / n} + {1 - w:.2f} × 0.5 = {w * k / n + (1 - w) * 0.5:.4f}"
      f"  (direct {(a + k) / (a + b + n):.4f})")

print("\nWorked example 4: sequential updating gives the same answer as all at once")
flips = rng.random(50) < 0.7
a, b = 1, 1
for f in flips:
    a, b = a + f, b + (1 - f)
print(f"  one flip at a time: Beta({a}, {b});  all at once: Beta({1 + flips.sum()}, {1 + 50 - flips.sum()})")

print("\nWorked example 5: posterior predictive — chance the next 5 flips contain at least 4 heads")
draws = post.rvs(200_000, random_state=1)
pred = rng.binomial(5, draws)
plug_in = stats.binom.sf(3, 5, 0.7)
print(f"  posterior predictive P(>= 4 of 5) = {(pred >= 4).mean():.4f};  plug-in p̂ = 0.7 gives {plug_in:.4f}"
      " (ignores the uncertainty in p)")

print("\nWorked example 6: normal mean with known σ — the Normal-Normal update")
df = pd.read_csv(ROOT / "data" / "students.csv")
x = df["exam_score"].to_numpy()
sigma = 14.0
m0, s0 = 50.0, 10.0          # prior belief about the mean score: around 50, give or take 10
prec = 1 / s0 ** 2 + len(x) / sigma ** 2
m_post = (m0 / s0 ** 2 + x.sum() / sigma ** 2) / prec
s_post = prec ** -0.5
print(f"  prior N({m0}, {s0}²); data x̄ = {x.mean():.2f}, n = {len(x)}")
print(f"  posterior N({m_post:.2f}, {s_post:.3f}²): 95% credible interval [{m_post - 1.96 * s_post:.2f}, {m_post + 1.96 * s_post:.2f}]")
print(f"  frequentist z-interval [{x.mean() - 1.96 * sigma / np.sqrt(len(x)):.2f}, {x.mean() + 1.96 * sigma / np.sqrt(len(x)):.2f}]")

print("\nWorked example 7: a Bayesian A/B test on data/ab_test.csv")
ab = pd.read_csv(ROOT / "data" / "ab_test.csv")
summary = ab.groupby("variant")["converted"].agg(["sum", "count"])
print(summary)
pa = stats.beta(1 + summary.loc["A", "sum"], 1 + summary.loc["A", "count"] - summary.loc["A", "sum"]).rvs(200_000, random_state=2)
pb = stats.beta(1 + summary.loc["B", "sum"], 1 + summary.loc["B", "count"] - summary.loc["B", "sum"]).rvs(200_000, random_state=3)
lift = pb - pa
print(f"  P(B converts better than A) = {(lift > 0).mean():.4f}")
print(f"  lift B - A: mean {lift.mean():.4f}, 95% credible interval [{np.percentile(lift, 2.5):.4f}, {np.percentile(lift, 97.5):.4f}]")
print(f"  P(lift > 0.5 percentage points) = {(lift > 0.005).mean():.4f}")
print(f"  expected loss if we ship B: {np.mean(np.maximum(pa - pb, 0)):.2e} in conversion rate"
      f" (vs {np.mean(np.maximum(pb - pa, 0)):.4f} if we keep A)")

print("\nWorked example 8: Gamma-Poisson — goals per match")
goals = rng.poisson(2.6, size=38)
alpha0, beta0 = 2.0, 1.0          # prior: about 2 goals per match, fairly vague
post_g = stats.gamma(alpha0 + goals.sum(), scale=1 / (beta0 + len(goals)))
print(f"  {len(goals)} matches, {goals.sum()} goals: posterior Gamma({alpha0 + goals.sum():.0f}, rate {beta0 + len(goals):.0f}),"
      f" mean {post_g.mean():.3f}, 95% CI [{post_g.ppf(.025):.3f}, {post_g.ppf(.975):.3f}]")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ps = np.linspace(0, 1, 500)
axes[0].plot(ps, stats.beta(1, 1).pdf(ps), ls="--", label="prior Beta(1, 1)")
axes[0].plot(ps, ps ** 7 * (1 - ps) ** 3 / np.trapezoid(ps ** 7 * (1 - ps) ** 3, ps), ls=":", label="likelihood (scaled)")
axes[0].plot(ps, post.pdf(ps), label="posterior Beta(8, 4)")
axes[0].set_title("7 heads in 10: prior × likelihood → posterior")
axes[0].set_xlabel("p = P(heads)")
axes[0].legend()
axes[1].hist(lift, bins=100, density=True)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title(f"Posterior of the A/B lift: P(B > A) = {(lift > 0).mean():.3f}")
axes[1].set_xlabel("conversion rate B − A")
fig.tight_layout()
fig.savefig(FIGURES / "beta_binomial_and_ab.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "beta_binomial_and_ab.png")
