"""Reference implementation of statkit/distributions.py. Compare with yours AFTER your tests pass."""

import math


def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def poisson_pmf(k, lam):
    if k < 0:
        return 0.0
    # Work in logs so large k doesn't overflow: log(λ^k e^-λ / k!) = k log λ - λ - log k!
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def geometric_pmf(k, p):
    if k < 1:
        return 0.0
    return (1 - p) ** (k - 1) * p


def exponential_cdf(x, rate):
    return 1 - math.exp(-rate * x) if x >= 0 else 0.0


def sample_inverse_cdf(inverse_cdf, n, rng):
    return [inverse_cdf(u) for u in rng.uniform(size=n)]


def normal_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return math.exp(-z * z / 2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def normal_ppf(q, mu=0.0, sigma=1.0):
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        if normal_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return mu + sigma * (lo + hi) / 2


def entropy(probs, base=2):
    return -sum(p * math.log(p, base) for p in probs if p > 0)


def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return math.inf
        total -= pi * math.log(qi, base)
    return total


def kl_divergence(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return math.inf
        total += pi * math.log(pi / qi, base)
    return total
