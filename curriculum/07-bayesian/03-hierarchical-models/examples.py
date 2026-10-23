"""07.03 Hierarchical models — the Eight Schools, three kinds of pooling, centred vs non-centred.

Run:  python curriculum/07-bayesian/03-hierarchical-models/examples.py   (about 1 minute)
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("pymc").setLevel(logging.ERROR)
import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)

# Rubin (1981): estimated effects of SAT coaching programmes in 8 schools, with their standard errors.
schools = list("ABCDEFGH")
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

print("The data: estimated coaching effect (points) ± standard error")
for s, yi, si in zip(schools, y, sigma):
    print(f"  school {s}: {yi:5.0f} ± {si:.0f}")

print("\n1. No pooling: each school on its own")
print(f"  school A's 95% interval: [{y[0] - 1.96 * sigma[0]:.1f}, {y[0] + 1.96 * sigma[0]:.1f}]")

print("\n2. Complete pooling: one common effect (precision-weighted mean)")
w = 1 / sigma ** 2
pooled = (w * y).sum() / w.sum()
pooled_se = w.sum() ** -0.5
print(f"  common effect {pooled:.2f} ± {pooled_se:.2f}")
chi2 = ((y - pooled) ** 2 * w).sum()
print(f"  test of homogeneity: chi2 = {chi2:.2f} on 7 df (p = {1 - __import__('scipy').stats.chi2.cdf(chi2, 7):.2f})"
      " — the data can't rule out identical effects")

print("\n3. Partial pooling: a hierarchical model")
with pm.Model() as centred:
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta = pm.Normal("theta", mu, tau, shape=8)
    pm.Normal("y", theta, sigma, observed=y)
    idata_c = pm.sample(1000, tune=1000, chains=4, target_accept=0.8, random_seed=1, progressbar=False)

with pm.Model() as noncentred:
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    z = pm.Normal("z", 0, 1, shape=8)
    theta = pm.Deterministic("theta", mu + tau * z)       # same model, different parameterisation
    pm.Normal("y", theta, sigma, observed=y)
    idata_nc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=1, progressbar=False)

for name, idata in [("centred", idata_c), ("non-centred", idata_nc)]:
    div = int(idata.sample_stats["diverging"].to_numpy().sum())
    s = az.summary(idata, var_names=["mu", "tau"], ci_prob=0.95, round_to="none")
    print(f"  {name:<12} divergences {div:>4};  mu {float(s.loc['mu', 'mean']):.2f}, tau {float(s.loc['tau', 'mean']):.2f}, "
          f"ESS(tau) {float(s.loc['tau', 'ess_bulk']):.0f}, R-hat(tau) {float(s.loc['tau', 'r_hat']):.3f}")

theta = idata_nc.posterior["theta"].to_numpy().reshape(-1, 8)
tau_draws = idata_nc.posterior["tau"].to_numpy().ravel()
mu_draws = idata_nc.posterior["mu"].to_numpy().ravel()
print("\n  school effects: raw estimate -> hierarchical posterior mean [95% interval]")
for i, s in enumerate(schools):
    lo, hi = np.percentile(theta[:, i], [2.5, 97.5])
    print(f"    {s}: {y[i]:5.0f} -> {theta[:, i].mean():5.1f}  [{lo:5.1f}, {hi:5.1f}]")
print(f"  overall mean mu {mu_draws.mean():.2f}; between-school sd tau: median {np.median(tau_draws):.2f}, "
      f"95% [{np.percentile(tau_draws, 2.5):.2f}, {np.percentile(tau_draws, 97.5):.2f}]")
print(f"  P(school A's true effect > 20) = {(theta[:, 0] > 20).mean():.3f}   (no pooling would say "
      f"{1 - __import__('scipy').stats.norm.cdf(20, 28, 15):.3f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order = np.arange(8)
axes[0].errorbar(order - 0.1, y, yerr=1.96 * sigma, fmt="o", label="no pooling (raw ± 1.96 SE)")
means = theta.mean(axis=0)
lo, hi = np.percentile(theta, [2.5, 97.5], axis=0)
axes[0].errorbar(order + 0.1, means, yerr=[means - lo, hi - means], fmt="s", label="partial pooling (posterior)")
axes[0].axhline(pooled, color="black", ls="--", lw=0.8, label=f"complete pooling ({pooled:.1f})")
axes[0].set_xticks(order, schools)
axes[0].set_title("Eight Schools: estimates shrink towards the common mean")
axes[0].set_ylabel("coaching effect")
axes[0].legend(fontsize=8)
for name, idata, color in [("centred", idata_c, "tab:red"), ("non-centred", idata_nc, "tab:blue")]:
    t = idata.posterior["tau"].to_numpy().ravel()
    th = idata.posterior["theta"].to_numpy().reshape(-1, 8)[:, 0]
    axes[1].scatter(np.log(t), th, s=3, alpha=0.3, color=color, label=name)
div_mask = idata_c.sample_stats["diverging"].to_numpy().ravel().astype(bool)
axes[1].scatter(np.log(idata_c.posterior["tau"].to_numpy().ravel()[div_mask]),
                idata_c.posterior["theta"].to_numpy().reshape(-1, 8)[div_mask, 0], s=12, color="black", label="divergences")
axes[1].set_title("The funnel: small τ is hard to explore (centred)")
axes[1].set_xlabel("log τ")
axes[1].set_ylabel("θ for school A")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "eight_schools.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "eight_schools.png")
