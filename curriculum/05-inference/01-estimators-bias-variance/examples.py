"""05.01 Estimators, bias and variance — every property checked by simulation.

Run:  python curriculum/05-inference/01-estimators-bias-variance/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(51)
REPS = 200_000


def report(name, estimates, truth):
    bias = estimates.mean() - truth
    var = estimates.var()
    mse = np.mean((estimates - truth) ** 2)
    print(f"  {name:<28} mean {estimates.mean():8.3f}  bias {bias:+7.3f}  variance {var:7.3f}  "
          f"MSE {mse:7.3f}  (bias² + var = {bias ** 2 + var:7.3f})")
    return bias, var, mse


print("Worked example 1: estimating the variance σ² = 4 from n = 5 normal observations")
x = rng.normal(0, 2, size=(REPS, 5))
report("s² (divide by n - 1)", x.var(axis=1, ddof=1), 4)
report("σ̂² (divide by n)", x.var(axis=1, ddof=0), 4)
report("divide by n + 1", x.var(axis=1, ddof=0) * 5 / 6, 4)

print("\nWorked example 2: three estimators of the centre of Uniform(0, θ = 10), n = 10")
u = rng.uniform(0, 10, size=(REPS, 10))
for name, est in [("sample mean", u.mean(axis=1)), ("sample median", np.median(u, axis=1)),
                  ("midrange (min + max)/2", (u.min(axis=1) + u.max(axis=1)) / 2)]:
    report(name, est, 5)

print("\nWorked example 3: shrinkage — a biased estimator can have lower MSE")
mu_true, sigma, n = 1.0, 4.0, 10
y = rng.normal(mu_true, sigma, size=(REPS, n)).mean(axis=1)
for c in [1.0, 0.8, 0.6, 0.4]:
    report(f"c · x̄ with c = {c}", c * y, mu_true)
c_best = mu_true ** 2 / (mu_true ** 2 + sigma ** 2 / n)
print(f"  best c = μ² / (μ² + σ²/n) = {c_best:.3f}  (but it depends on the unknown μ!)")

print("\nWorked example 4: consistency — the MSE of x̄ falls like σ²/n")
for n in [5, 20, 80, 320]:
    m = rng.exponential(10, size=(20_000, n)).mean(axis=1)
    print(f"  n = {n:>3}: MSE {np.mean((m - 10) ** 2):7.3f}   σ²/n = {100 / n:7.3f}")

print("\nWorked example 5: a consistent but biased estimator — the sample maximum for θ in Uniform(0, θ)")
for n in [5, 20, 100]:
    mx = rng.uniform(0, 10, size=(50_000, n)).max(axis=1)
    corrected = mx * (n + 1) / n
    print(f"  n = {n:>3}: E[max] = {mx.mean():.3f} (theory {10 * n / (n + 1):.3f})   "
          f"(n+1)/n · max has mean {corrected.mean():.3f}")

# Figure: bias-variance picture for the variance estimators
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for ddof, label in [(1, "s² (÷ n−1): unbiased"), (0, "σ̂² (÷ n): biased, lower variance")]:
    ax.hist(x.var(axis=1, ddof=ddof), bins=150, range=(0, 16), alpha=0.5, density=True, label=label)
ax.axvline(4, color="black", ls="--", label="true σ² = 4")
ax.set_title("Two estimators of σ² from samples of 5")
ax.set_xlabel("estimate")
ax.set_ylabel("density")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "variance_estimators.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "variance_estimators.png")
