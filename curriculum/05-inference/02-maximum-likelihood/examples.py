"""05.02 Maximum likelihood — closed forms, numerical optimisation and standard errors.

Run:  python curriculum/05-inference/02-maximum-likelihood/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(52)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("Worked example 1: a coin lands heads 7 times in 10 flips")
k, n = 7, 10
for p in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    L = p ** k * (1 - p) ** (n - k)
    print(f"  p = {p}: likelihood p^7 (1-p)^3 = {L:.6f}   log-likelihood = {math.log(L):.4f}")
print(f"  MLE = k/n = {k / n}")

print("\nWorked example 2: Poisson MLE = the sample mean (simulated goals data)")
goals = rng.poisson(2.6, size=380)
lam_grid = np.linspace(1.5, 4, 2501)
loglik = [stats.poisson.logpmf(goals, lam).sum() for lam in lam_grid]
print(f"  grid maximum at λ = {lam_grid[int(np.argmax(loglik))]:.3f};  sample mean = {goals.mean():.3f}")

print("\nWorked example 3: normal MLE for exam_score")
x = df["exam_score"].to_numpy(dtype=float)
mu_hat = x.mean()
sigma2_hat = ((x - mu_hat) ** 2).mean()
print(f"  μ̂ = x̄ = {mu_hat:.3f};  σ̂² = (1/n)Σ(x - x̄)² = {sigma2_hat:.3f} (vs s² = {x.var(ddof=1):.3f})")
fit = stats.norm.fit(x)
print(f"  scipy.stats.norm.fit: loc {fit[0]:.3f}, scale² {fit[1] ** 2:.3f}  (scipy returns the MLE)")

print("\nWorked example 4: numerical MLE for a gamma model of study_hours (true shape 4, scale 2.5)")
h = df["study_hours"].to_numpy()


def negative_log_likelihood(params):
    log_shape, log_scale = params          # optimise on the log scale so both stay positive
    return -stats.gamma.logpdf(h, a=np.exp(log_shape), scale=np.exp(log_scale)).sum()


result = optimize.minimize(negative_log_likelihood, x0=[0.0, 0.0], method="Nelder-Mead")
shape_hat, scale_hat = np.exp(result.x)
print(f"  optimiser converged: {result.success};  shape {shape_hat:.3f}, scale {scale_hat:.3f}, "
      f"mean shape*scale = {shape_hat * scale_hat:.2f} (sample mean {h.mean():.2f})")
a, loc, sc = stats.gamma.fit(h, floc=0)
print(f"  scipy.stats.gamma.fit(floc=0): shape {a:.3f}, scale {sc:.3f}")
print(f"  NLL at the MLE {result.fun:.2f};  NLL at the truth (4, 2.5) {negative_log_likelihood(np.log([4, 2.5])):.2f}")

print("\nWorked example 5: standard error from the Fisher information (Bernoulli)")
p_hat, n = 0.7, 10
info = n / (p_hat * (1 - p_hat))
print(f"  I(p̂) = n / (p̂(1 - p̂)) = {info:.2f};  SE = 1/sqrt(I) = {1 / math.sqrt(info):.4f} = sqrt(p̂(1-p̂)/n)")
h_ = 1e-4
ll = lambda p: k * math.log(p) + (n - k) * math.log(1 - p)
curvature = -(ll(p_hat + h_) - 2 * ll(p_hat) + ll(p_hat - h_)) / h_ ** 2
print(f"  numerical curvature of the log-likelihood at p̂ = {curvature:.2f} (same as I)")
repeat_mles = rng.binomial(10, 0.7, size=100_000) / 10
print(f"  sampling check: MLEs from 100,000 repeats with true p = 0.7, n = 10 have sd {repeat_mles.std():.4f}")

print("\nWorked example 6: least squares = normal MLE (a line through students.csv)")
X = np.column_stack([np.ones(len(df)), df["study_hours"]])
y = df["exam_score"].to_numpy(dtype=float)


def nll_line(params):
    b0, b1, log_sigma = params
    return -stats.norm.logpdf(y, loc=b0 + b1 * X[:, 1], scale=np.exp(log_sigma)).sum()


mle = optimize.minimize(nll_line, x0=[50, 1, 2], method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
ols = np.linalg.solve(X.T @ X, X.T @ y)
print(f"  MLE  intercept {mle.x[0]:.3f}, slope {mle.x[1]:.4f}")
print(f"  OLS  intercept {ols[0]:.3f}, slope {ols[1]:.4f}  (identical: minimising squared error = maximising normal likelihood)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ps = np.linspace(0.01, 0.99, 300)
axes[0].plot(ps, ps ** 7 * (1 - ps) ** 3)
axes[0].axvline(0.7, color="grey", ls="--")
axes[0].set_title("Likelihood of p after 7 heads in 10")
axes[0].set_xlabel("p")
axes[0].set_ylabel("L(p)")
for n_, k_ in [(10, 7), (100, 70)]:
    lp = k_ * np.log(ps) + (n_ - k_) * np.log(1 - ps)
    axes[1].plot(ps, lp - lp.max(), label=f"{k_} of {n_}")
axes[1].set_ylim(-6, 0.3)
axes[1].axvline(0.7, color="grey", ls="--")
axes[1].set_title("More data → sharper log-likelihood → smaller SE")
axes[1].set_xlabel("p")
axes[1].set_ylabel("log L(p) − max")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "likelihood_curves.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "likelihood_curves.png")
