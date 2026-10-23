"""07.02 MCMC and PyMC — Metropolis from scratch, diagnostics, then the same models in PyMC.

Run:  python curriculum/07-bayesian/02-mcmc-and-pymc/examples.py   (about 1 minute; PyMC compiles models)
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.getLogger("pymc").setLevel(logging.ERROR)
import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(72)
df = pd.read_csv(ROOT / "data" / "students.csv")


def metropolis(log_post, x0, steps, proposal_sd, rng):
    x = np.atleast_1d(np.asarray(x0, float))
    lp = log_post(x)
    chain = np.empty((steps, x.size))
    accepted = 0
    for t in range(steps):
        proposal = x + rng.normal(0, proposal_sd, x.size)
        lp_new = log_post(proposal)
        if np.log(rng.random()) < lp_new - lp:          # accept with probability min(1, ratio)
            x, lp = proposal, lp_new
            accepted += 1
        chain[t] = x
    return chain, accepted / steps


def ess(chain):
    """Effective sample size from the initial positive autocorrelations."""
    x = chain - chain.mean()
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (x @ x)
    s = 0.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        s += acf[k]
    return n / (1 + 2 * s)


def rhat(chains):
    """Classic Gelman-Rubin R-hat for an array of shape (m chains, n draws)."""
    m, n = chains.shape
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


print("Worked example 1: Monte Carlo integration — E[p²] under Beta(8, 4)")
draws = stats.beta(8, 4).rvs(100_000, random_state=1)
print(f"  Monte Carlo {np.mean(draws ** 2):.4f};  exact a(a+1)/((a+b)(a+b+1)) = {8 * 9 / (12 * 13):.4f}")

print("\nWorked example 2: Metropolis on the coin posterior (7 heads in 10, flat prior)")


def log_post_coin(p):
    p = p[0]
    if not 0 < p < 1:
        return -np.inf
    return 7 * np.log(p) + 3 * np.log(1 - p)


chain, acc = metropolis(log_post_coin, [0.5], 20_000, 0.15, rng)
kept = chain[2_000:, 0]                                   # discard warm-up (burn-in)
print(f"  acceptance rate {acc:.2f}; posterior mean {kept.mean():.4f} (exact 0.6667); "
      f"95% interval [{np.percentile(kept, 2.5):.3f}, {np.percentile(kept, 97.5):.3f}] (exact [0.390, 0.891])")
print(f"  {len(kept)} draws but effective sample size only {ess(kept):.0f} (draws are autocorrelated)")

print("\nWorked example 3: the proposal step size matters")
for sd in [0.01, 0.15, 2.0]:
    c, a = metropolis(log_post_coin, [0.5], 20_000, sd, np.random.default_rng(1))
    print(f"  proposal sd {sd:<5}: acceptance {a:.2f}, ESS {ess(c[2_000:, 0]):7.0f}, mean {c[2_000:, 0].mean():.4f}")

print("\nWorked example 4: two parameters — mean and sd of exam_score, four chains from scattered starts")
y = df["exam_score"].to_numpy(dtype=float)


def log_post_normal(theta):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    log_prior = stats.norm.logpdf(mu, 60, 20) + stats.norm.logpdf(log_sigma, np.log(10), 1)
    return log_prior + stats.norm.logpdf(y, mu, sigma).sum()


starts = [[30, 1], [90, 4], [60, 2.6], [50, 3.5]]
step_sizes = np.array([1.5, 0.08])                 # one step size per parameter: μ moves in points, log σ in small units
chains = np.array([metropolis(log_post_normal, s, 10_000, step_sizes, np.random.default_rng(i))[0] for i, s in enumerate(starts)])
post_mu = chains[:, 2_000:, 0]
post_sigma = np.exp(chains[:, 2_000:, 1])
print(f"  μ: mean {post_mu.mean():.2f}, 95% [{np.percentile(post_mu, 2.5):.2f}, {np.percentile(post_mu, 97.5):.2f}], "
      f"R-hat {rhat(post_mu):.3f}")
print(f"  σ: mean {post_sigma.mean():.2f}, 95% [{np.percentile(post_sigma, 2.5):.2f}, {np.percentile(post_sigma, 97.5):.2f}], "
      f"R-hat {rhat(post_sigma):.3f}")
print(f"  R-hat using only the first 100 draws (before the chains meet): {rhat(chains[:, :100, 0]):.2f}")
print(f"  ESS of μ (chain 1, after warm-up): {ess(post_mu[0]):.0f} of {post_mu.shape[1]} draws")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for i in range(4):
    axes[0].plot(chains[i, :1500, 0], lw=0.7, label=f"chain {i + 1}")
axes[0].set_title("Trace of μ: chains from different starts converge")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("μ")
axes[0].legend(fontsize=7)
acf = [np.corrcoef(kept[:-k], kept[k:])[0, 1] for k in range(1, 60)]
axes[1].bar(range(1, 60), acf)
axes[1].set_title("Autocorrelation of the coin chain (why ESS < n)")
axes[1].set_xlabel("lag")
fig.tight_layout()
fig.savefig(FIGURES / "trace_and_acf.png", dpi=120)
plt.close(fig)

print("\nWorked example 5: the same normal model in PyMC (NUTS sampler)")
with pm.Model() as normal_model:
    mu = pm.Normal("mu", 60, 20)
    sigma = pm.LogNormal("sigma", np.log(10), 1)
    pm.Normal("score", mu, sigma, observed=y)
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=7, progressbar=False)
print(az.summary(idata, ci_prob=0.95))

print("\nWorked example 6: Bayesian logistic regression in PyMC — P(score >= 70) on study hours")
high = (df["exam_score"] >= 70).astype(int).to_numpy()
hours = df["study_hours"].to_numpy()
with pm.Model() as logit_model:
    b0 = pm.Normal("b0", 0, 5)
    b1 = pm.Normal("b1", 0, 1)
    pm.Bernoulli("high", logit_p=b0 + b1 * hours, observed=high)
    idata_logit = pm.sample(1000, tune=1000, chains=4, random_seed=8, progressbar=False)
summ = az.summary(idata_logit, ci_prob=0.95)
print(summ)
b1_draws = idata_logit.posterior["b1"].to_numpy().ravel()
print(f"  odds ratio per study hour: posterior mean {np.exp(b1_draws).mean():.3f}, "
      f"95% [{np.exp(np.percentile(b1_draws, 2.5)):.3f}, {np.exp(np.percentile(b1_draws, 97.5)):.3f}]")
divergences = int(idata_logit.sample_stats["diverging"].to_numpy().sum())
print(f"  divergent transitions: {divergences}")
print("\nFigure saved:", FIGURES / "trace_and_acf.png")
