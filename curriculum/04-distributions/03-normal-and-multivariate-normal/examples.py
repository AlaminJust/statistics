"""04.03 The normal and multivariate normal — worked examples.

Run:  python curriculum/04-distributions/03-normal-and-multivariate-normal/examples.py
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
rng = np.random.default_rng(43)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("Worked example 1: IQ ~ N(100, 15^2)")
iq = stats.norm(100, 15)
print(f"  P(IQ > 130): z = (130 - 100)/15 = 2 -> 1 - Φ(2) = {iq.sf(130):.4f}")
print(f"  P(85 < IQ < 115) = Φ(1) - Φ(-1) = {iq.cdf(115) - iq.cdf(85):.4f}")
print(f"  90th percentile = 100 + 15 * {stats.norm.ppf(0.9):.4f} = {iq.ppf(0.9):.2f}")
print(f"  P(IQ < 70) = {iq.cdf(70):.4f}")

print("\nKey standard-normal values")
for z in [1, 1.645, 1.96, 2, 2.576, 3]:
    print(f"  P(-{z} < Z < {z}) = {stats.norm.cdf(z) - stats.norm.cdf(-z):.4f}")

print("\nWorked example 2: sums of independent normals")
# Morning commute: walk ~ N(10, 2^2), train ~ N(25, 4^2)
total = stats.norm(10 + 25, np.sqrt(2 ** 2 + 4 ** 2))
print(f"  total ~ N(35, {total.std():.3f}^2); P(total > 40) = {total.sf(40):.4f}")
sim = rng.normal(10, 2, 500_000) + rng.normal(25, 4, 500_000)
print(f"  simulated P(total > 40) = {(sim > 40).mean():.4f}")
print(f"  mean of 25 IQ scores ~ N(100, (15/5)^2): P(mean > 106) = {stats.norm(100, 3).sf(106):.4f}")

print("\nWorked example 3: normal approximation to the binomial — 100 fair flips, P(X >= 60)")
exact = stats.binom.sf(59, 100, 0.5)
approx_no_cc = stats.norm(50, 5).sf(60)
approx_cc = stats.norm(50, 5).sf(59.5)
print(f"  exact {exact:.4f};  normal without correction {approx_no_cc:.4f};  with continuity correction {approx_cc:.4f}")

print("\nWorked example 4: checking normality — skewness and Q-Q correlation")
for column in ["exam_score", "sleep_hours", "commute_minutes"]:
    x = df[column]
    (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
    print(f"  {column:<16} skew {stats.skew(x):+.2f}   Q-Q correlation {r:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, column in zip(axes, ["exam_score", "sleep_hours", "commute_minutes"]):
    stats.probplot(df[column], dist="norm", plot=ax)
    ax.set_title(f"Q–Q plot: {column}")
fig.tight_layout()
fig.savefig(FIGURES / "qq_plots.png", dpi=120)
plt.close(fig)

print("\nWorked example 5: the bivariate normal — midterm and final scores")
mu = np.array([60, 65])
sd = np.array([10, 12])
rho = 0.6
cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
print("  covariance matrix:\n", cov)
cond_mean = mu[1] + rho * sd[1] / sd[0] * (75 - mu[0])
cond_sd = sd[1] * np.sqrt(1 - rho ** 2)
print(f"  E[final | midterm = 75] = 65 + 0.6*(12/10)*(75 - 60) = {cond_mean:.1f}")
print(f"  sd(final | midterm = 75) = 12 * sqrt(1 - 0.36) = {cond_sd:.2f}  (unconditional sd 12)")
draws = rng.multivariate_normal(mu, cov, size=400_000)
near = np.abs(draws[:, 0] - 75) < 0.5
print(f"  simulated among midterms within 0.5 of 75 (n = {near.sum()}): mean final {draws[near, 1].mean():.2f}, "
      f"sd {draws[near, 1].std():.2f}")
print(f"  regression to the mean: a student 1.5 sd above on the midterm is expected "
      f"{(cond_mean - mu[1]) / sd[1]:.2f} sd above on the final")

L = np.linalg.cholesky(cov)
z = rng.standard_normal((2, 200_000))
by_hand = (mu[:, None] + L @ z).T
print("  sampling by hand with the Cholesky factor L (Σ = L Lᵀ): sample covariance\n",
      np.cov(by_hand, rowvar=False).round(1))

xs, ys = np.meshgrid(np.linspace(25, 95, 200), np.linspace(25, 105, 200))
density = stats.multivariate_normal(mu, cov).pdf(np.dstack([xs, ys]))
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(draws[:2000, 0], draws[:2000, 1], s=4, alpha=0.3)
ax.contour(xs, ys, density, levels=6, colors="black", linewidths=0.8)
line_x = np.linspace(30, 90, 10)
ax.plot(line_x, mu[1] + rho * sd[1] / sd[0] * (line_x - mu[0]), color="tab:red", label="E[final | midterm]")
ax.set_title("Bivariate normal, ρ = 0.6: elliptical contours")
ax.set_xlabel("midterm score")
ax.set_ylabel("final score")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "bivariate_normal.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
