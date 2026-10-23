"""Module 05 — statistical inference, implemented from scratch.

You may import `scipy.stats` here ONLY for distribution quantiles and tail probabilities (t.ppf, t.sf, norm.ppf,
norm.sf). Everything else — means, SEs, intervals, resampling, corrections — is yours. Reuse statkit.descriptive.
Run: make test M=inference
"""

import math  # noqa: F401


# ---- 05.01 Estimators --------------------------------------------------------------------------

def standard_error(values):
    """Estimated standard error of the mean: s / sqrt(n)."""
    raise NotImplementedError


# ---- 05.02 Maximum likelihood ------------------------------------------------------------------

def mle_bernoulli(data):
    """MLE of p from 0/1 data."""
    raise NotImplementedError


def mle_poisson(data):
    """MLE of λ from count data."""
    raise NotImplementedError


def mle_normal(data):
    """(μ̂, σ̂²) — note σ̂² divides by n."""
    raise NotImplementedError


# ---- 05.03 Confidence intervals ----------------------------------------------------------------

def t_interval(values, level=0.95):
    """(low, high): x̄ ± t_{(1+level)/2, n-1} · s/√n."""
    raise NotImplementedError


def wilson_interval(successes, n, level=0.95):
    """(low, high): the Wilson score interval for a proportion."""
    raise NotImplementedError


# ---- 05.04 Bootstrap ---------------------------------------------------------------------------

def bootstrap_ci(data, statistic, n_boot=10_000, level=0.95, rng=None):
    """Percentile bootstrap interval (low, high) for statistic(data). rng: a NumPy Generator."""
    raise NotImplementedError


# ---- 05.05 Hypothesis testing ------------------------------------------------------------------

def permutation_test(x, y, n_perm=10_000, rng=None):
    """Two-sided permutation p-value for the difference in means: (count + 1) / (n_perm + 1)."""
    raise NotImplementedError


# ---- 05.06 Common tests ------------------------------------------------------------------------

def welch_t_test(x, y):
    """(t, df, two-sided p) for Welch's unequal-variance t-test."""
    raise NotImplementedError


# ---- 05.07 Power and effect size ---------------------------------------------------------------

def cohens_d(x, y):
    """(mean(x) - mean(y)) / pooled sd."""
    raise NotImplementedError


def sample_size_two_means(delta, sd, alpha=0.05, power=0.8):
    """n per group (rounded up) from n = 2 (z_{1-α/2} + z_{power})² sd² / delta²."""
    raise NotImplementedError


# ---- 05.08 Multiple testing --------------------------------------------------------------------

def bonferroni(pvalues, alpha=0.05):
    """List of booleans (reject?) in the original order."""
    raise NotImplementedError


def benjamini_hochberg(pvalues, q=0.05):
    """List of booleans (reject?) in the original order, controlling the FDR at q."""
    raise NotImplementedError
