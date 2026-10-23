"""07.04 Model checking and comparison — prior and posterior predictive checks, LOO, prior sensitivity.

Run:  python curriculum/07-bayesian/04-model-checking-and-comparison/examples.py   (about 1 minute)
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("pymc").setLevel(logging.ERROR)
import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(74)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# Overdispersed counts: purchases per customer in a month (customers differ in their underlying rates).
n = 400
rates = rng.gamma(shape=1.5, scale=2.0, size=n)
y = rng.poisson(rates)
print(f"Data: {n} customers, mean purchases {y.mean():.2f}, variance {y.var(ddof=1):.2f}, zeros {np.mean(y == 0):.1%}")

print("\nWorked example 1: prior predictive check — do the priors produce sensible data?")
with pm.Model() as vague:
    lam = pm.Exponential("lam", 0.001)                    # 'uninformative': mean 1,000 purchases a month!
    pm.Poisson("y", lam, observed=y)
    prior_vague = pm.sample_prior_predictive(1000, random_seed=1)
with pm.Model() as sensible:
    lam = pm.Gamma("lam", 2, 0.5)                          # mean 4 purchases, rarely above 15
    pm.Poisson("y", lam, observed=y)
    prior_sensible = pm.sample_prior_predictive(1000, random_seed=1)
for name, pp in [("Exponential(0.001)", prior_vague), ("Gamma(2, 0.5)", prior_sensible)]:
    sims = pp.prior_predictive["y"].to_numpy().ravel()
    print(f"  prior {name:<19}: simulated purchases median {np.median(sims):.0f}, 99th percentile {np.percentile(sims, 99):.0f}")

print("\nFit two models: Poisson and negative binomial")
with pm.Model() as poisson_model:
    lam = pm.Gamma("lam", 2, 0.5)
    pm.Poisson("y", lam, observed=y)
    idata_p = pm.sample(1000, tune=1000, chains=4, random_seed=2, progressbar=False)
    pm.sample_posterior_predictive(idata_p, extend_inferencedata=True, random_seed=3, progressbar=False)
    pm.compute_log_likelihood(idata_p, progressbar=False)
with pm.Model() as nb_model:
    mu = pm.Gamma("mu", 2, 0.5)
    alpha = pm.Gamma("alpha", 2, 1)                        # dispersion: smaller = more overdispersed
    pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y)
    idata_nb = pm.sample(1000, tune=1000, chains=4, random_seed=2, progressbar=False)
    pm.sample_posterior_predictive(idata_nb, extend_inferencedata=True, random_seed=3, progressbar=False)
    pm.compute_log_likelihood(idata_nb, progressbar=False)
print(az.summary(idata_nb, ci_prob=0.95))

print("\nWorked example 2: posterior predictive checks — can each model reproduce the data?")
stats_to_check = {"variance": lambda d: d.var(axis=-1, ddof=1),
                  "share of zeros": lambda d: (d == 0).mean(axis=-1),
                  "maximum": lambda d: d.max(axis=-1)}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for row, (name, idata) in enumerate([("Poisson", idata_p), ("negative binomial", idata_nb)]):
    reps = idata.posterior_predictive["y"].to_numpy().reshape(-1, n)
    for col, (stat_name, f) in enumerate(stats_to_check.items()):
        rep_values = f(reps)
        observed = f(y[None, :])[0]
        p_value = np.mean(rep_values >= observed)
        print(f"  {name:<18} {stat_name:<15}: observed {observed:7.3f}, replicated 95% range "
              f"[{np.percentile(rep_values, 2.5):7.3f}, {np.percentile(rep_values, 97.5):7.3f}], "
              f"posterior predictive p = {p_value:.3f}")
        axes[row, col].hist(rep_values, bins=40, alpha=0.7)
        axes[row, col].axvline(observed, color="black", lw=2)
        axes[row, col].set_title(f"{name}: {stat_name}", fontsize=10)
fig.suptitle("Posterior predictive checks (black line = observed data)")
fig.tight_layout()
fig.savefig(FIGURES / "posterior_predictive_checks.png", dpi=110)
plt.close(fig)

print("\nWorked example 3: comparing models with PSIS-LOO cross-validation")
print(az.loo(idata_p))
print(az.loo(idata_nb))
print(az.compare({"poisson": idata_p, "negative binomial": idata_nb}))

print("\nWorked example 4: prior sensitivity for a small dataset (5 customers)")
small = y[:5]
for label, (a, b) in {"Gamma(2, 0.5)": (2, 0.5), "Gamma(1, 0.1)": (1, 0.1), "Gamma(20, 10)": (20, 10)}.items():
    post_a, post_b = a + small.sum(), b + len(small)
    print(f"  prior {label:<14} (prior mean {a / b:5.1f}): posterior mean {post_a / post_b:.2f}")
big_post = [(a + y.sum()) / (b + n) for a, b in [(2, 0.5), (1, 0.1), (20, 10)]]
print(f"  with all {n} customers the three posterior means are {np.round(big_post, 3).tolist()}")
print("\nFigure saved:", FIGURES / "posterior_predictive_checks.png")
