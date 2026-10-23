"""Module 07 — Bayesian computation from scratch. NumPy and scipy.stats densities are allowed; PyMC is not.

Run: make test M=bayes
"""

import numpy as np  # noqa: F401


# ---- 07.01 Conjugate updates -------------------------------------------------------------------

def beta_binomial_update(alpha, beta, successes, failures):
    """Posterior Beta parameters (alpha + successes, beta + failures)."""
    raise NotImplementedError


def normal_normal_update(prior_mean, prior_sd, data, sigma):
    """(posterior mean, posterior sd) for a normal mean with known data sd sigma."""
    raise NotImplementedError


# ---- 07.02 MCMC --------------------------------------------------------------------------------

def metropolis(log_post, x0, steps, proposal_sd, rng):
    """Random-walk Metropolis. Returns (draws of shape (steps, dim), acceptance rate)."""
    raise NotImplementedError


def effective_sample_size(chain):
    """ESS = n / (1 + 2 Σ ρ_k), summing autocorrelations until the first negative one."""
    raise NotImplementedError


def rhat(chains):
    """Classic Gelman–Rubin R-hat for an array of shape (m chains, n draws)."""
    raise NotImplementedError


# ---- 07.04 Model checking ----------------------------------------------------------------------

def credible_interval(samples, level=0.95):
    """Equal-tailed interval (low, high) from posterior draws."""
    raise NotImplementedError


def posterior_predictive_pvalue(replicated_stats, observed_stat):
    """Share of replicated statistics >= the observed one."""
    raise NotImplementedError


# ---- 07.05 EM ----------------------------------------------------------------------------------

def em_gaussian_mixture(x, k=2, iterations=500, rng=None, n_init=5):
    """Fit a k-component 1-D Gaussian mixture by EM from n_init random starts; return the best
    (weights, means, sds, log_likelihood), with components sorted by mean."""
    raise NotImplementedError
