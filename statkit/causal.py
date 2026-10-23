"""Module 09 — experiments, causal inference and research methods, from scratch.

NumPy allowed; scipy.stats only for quantiles and tail probabilities.
Run: make test M=causal
"""

import numpy as np  # noqa: F401


# ---- 09.01 Experimental design -----------------------------------------------------------------

def sample_size_two_proportions(p1, p2, alpha=0.05, power=0.8):
    """Users per arm, rounded up:
    n = [z_{1-α/2} sqrt(2 p̄(1-p̄)) + z_power sqrt(p1(1-p1) + p2(1-p2))]² / (p2 - p1)², with p̄ = (p1 + p2)/2."""
    raise NotImplementedError


def srm_test(n_a, n_b, expected_ratio=0.5):
    """χ² goodness-of-fit p-value for a sample ratio mismatch (expected share of A = expected_ratio)."""
    raise NotImplementedError


def cuped_adjust(y, x):
    """y - θ (x - mean(x)), with θ = Cov(x, y) / Var(x)."""
    raise NotImplementedError


# ---- 09.02 Potential outcomes ------------------------------------------------------------------

def ate_difference_in_means(y, t):
    """(ATE estimate, Welch standard error) for a randomised experiment; t is 0/1."""
    raise NotImplementedError


# ---- 09.03 Observational methods ---------------------------------------------------------------

def ipw_ate(y, t, propensity):
    """ATE by inverse probability weighting with normalised (Hájek) weights."""
    raise NotImplementedError


def standardized_mean_difference(x, t, weights=None):
    """(weighted mean_treated - weighted mean_control) / sqrt((var_treated + var_control) / 2)."""
    raise NotImplementedError


def difference_in_differences(pre_treated, post_treated, pre_control, post_control):
    """(mean(post_treated) - mean(pre_treated)) - (mean(post_control) - mean(pre_control))."""
    raise NotImplementedError


# ---- 09.04 Reproducible research ---------------------------------------------------------------

def seed_summary(scores, level=0.95):
    """(mean, sd, ci_low, ci_high) with a t-interval for the mean over runs."""
    raise NotImplementedError


def interquartile_mean(scores):
    """Mean of the middle 50% of all values (drop the lowest and highest quarter, len // 4 from each end)."""
    raise NotImplementedError
