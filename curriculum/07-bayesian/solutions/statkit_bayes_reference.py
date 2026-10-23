"""Reference implementation of statkit/bayes.py. Compare with yours AFTER your tests pass."""

import numpy as np
from scipy import stats


def beta_binomial_update(alpha, beta, successes, failures):
    return alpha + successes, beta + failures


def normal_normal_update(prior_mean, prior_sd, data, sigma):
    data = np.asarray(data, float)
    precision = 1 / prior_sd ** 2 + len(data) / sigma ** 2
    mean = (prior_mean / prior_sd ** 2 + data.sum() / sigma ** 2) / precision
    return mean, precision ** -0.5


def metropolis(log_post, x0, steps, proposal_sd, rng):
    x = np.atleast_1d(np.asarray(x0, float))
    lp = log_post(x)
    draws = np.empty((steps, x.size))
    accepted = 0
    for t in range(steps):
        proposal = x + rng.normal(0, proposal_sd, x.size)
        lp_new = log_post(proposal)
        if np.log(rng.random()) < lp_new - lp:
            x, lp = proposal, lp_new
            accepted += 1
        draws[t] = x
    return draws, accepted / steps


def effective_sample_size(chain):
    x = np.asarray(chain, float) - np.mean(chain)
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (x @ x)
    total = 0.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        total += acf[k]
    return n / (1 + 2 * total)


def rhat(chains):
    chains = np.asarray(chains, float)
    m, n = chains.shape
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def credible_interval(samples, level=0.95):
    alpha = (1 - level) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def posterior_predictive_pvalue(replicated_stats, observed_stat):
    return float(np.mean(np.asarray(replicated_stats) >= observed_stat))


def em_gaussian_mixture(x, k=2, iterations=500, rng=None, n_init=5):
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(x, float)
    best = None
    for _ in range(n_init):
        w = np.full(k, 1 / k)
        m = rng.choice(x, k, replace=False)
        s = np.full(k, x.std())
        previous = -np.inf
        for _ in range(iterations):
            dens = np.vstack([w[j] * stats.norm.pdf(x, m[j], s[j]) for j in range(k)])
            total = dens.sum(axis=0)
            ll = np.log(total).sum()
            r = dens / total
            nk = r.sum(axis=1)
            w = nk / len(x)
            m = (r @ x) / nk
            s = np.sqrt((r * (x - m[:, None]) ** 2).sum(axis=1) / nk)
            if abs(ll - previous) < 1e-10:
                break
            previous = ll
        if best is None or ll > best[3]:
            order = np.argsort(m)
            best = (w[order], m[order], s[order], ll)
    return best
