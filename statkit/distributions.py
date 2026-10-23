"""Module 04 — distributions, implemented from scratch with `math` (no scipy).

Run: make test M=distributions
"""

import math  # noqa: F401


# ---- 04.01 Discrete distributions --------------------------------------------------------------

def binomial_pmf(k, n, p):
    """P(X = k) for X ~ Binomial(n, p). Return 0 for k outside 0..n."""
    raise NotImplementedError


def poisson_pmf(k, lam):
    """P(X = k) for X ~ Poisson(lam). Return 0 for negative k."""
    raise NotImplementedError


def geometric_pmf(k, p):
    """P(first success on trial k), k = 1, 2, ... Return 0 for k < 1."""
    raise NotImplementedError


# ---- 04.02 Continuous distributions ------------------------------------------------------------

def exponential_cdf(x, rate):
    """P(X <= x) for X ~ Exponential(rate): 1 - e^(-rate x) for x >= 0, else 0."""
    raise NotImplementedError


def sample_inverse_cdf(inverse_cdf, n, rng):
    """Draw n values by applying inverse_cdf to n uniforms from rng.uniform(). Return a list."""
    raise NotImplementedError


# ---- 04.03 The normal distribution -------------------------------------------------------------

def normal_pdf(x, mu=0.0, sigma=1.0):
    """Density of N(mu, sigma²) at x."""
    raise NotImplementedError


def normal_cdf(x, mu=0.0, sigma=1.0):
    """P(X <= x) for N(mu, sigma²), using math.erf."""
    raise NotImplementedError


def normal_ppf(q, mu=0.0, sigma=1.0):
    """Quantile of N(mu, sigma²) by bisection on normal_cdf (accurate to 1e-8). Raise ValueError unless 0 < q < 1."""
    raise NotImplementedError


# ---- 04.06 Information theory ------------------------------------------------------------------

def entropy(probs, base=2):
    """-Σ p log p, skipping p = 0."""
    raise NotImplementedError


def cross_entropy(p, q, base=2):
    """-Σ p log q, skipping p = 0; math.inf if q = 0 where p > 0."""
    raise NotImplementedError


def kl_divergence(p, q, base=2):
    """Σ p log(p/q), skipping p = 0; math.inf if q = 0 where p > 0."""
    raise NotImplementedError
