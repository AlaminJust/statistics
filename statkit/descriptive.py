"""Module 02 — descriptive statistics, implemented from scratch.

Plain Python and `math` only (no NumPy, no statistics module). Reuse your own functions: `variance` should call
`mean`, `iqr` should call `percentile`, and so on. Each function says which lesson introduces it.
Run: make test M=descriptive
"""

import math  # noqa: F401


# ---- 02.01 Data and variables ------------------------------------------------------------------

def frequency_table(values):
    """Dict mapping each distinct value to its count. frequency_table(["a", "b", "a"]) -> {"a": 2, "b": 1}."""
    raise NotImplementedError


def relative_frequencies(values):
    """Dict mapping each distinct value to count / n. The proportions sum to 1."""
    raise NotImplementedError


# ---- 02.02 Measures of centre ------------------------------------------------------------------

def mean(values):
    """Arithmetic mean. Raise ValueError for empty data."""
    raise NotImplementedError


def median(values):
    """Middle of the sorted data; the mean of the two middle values when n is even."""
    raise NotImplementedError


def mode(values):
    """Sorted list of every value tied for the highest count. mode([1, 2, 2, 3, 3]) -> [2, 3]."""
    raise NotImplementedError


def weighted_mean(values, weights):
    """Σ w·x / Σ w. Raise ValueError if lengths differ or the weights sum to 0."""
    raise NotImplementedError


def trimmed_mean(values, proportion):
    """Sort, drop int(proportion * n) values from EACH end, average the rest (matches scipy.stats.trim_mean)."""
    raise NotImplementedError


def geometric_mean(values):
    """exp(mean(log x)). Raise ValueError if any value is <= 0."""
    raise NotImplementedError


# ---- 02.03 Measures of spread ------------------------------------------------------------------

def data_range(values):
    """max - min."""
    raise NotImplementedError


def variance(values, ddof=1):
    """Σ(x - x̄)² / (n - ddof). ddof=1: sample variance; ddof=0: population. Raise ValueError if n <= ddof."""
    raise NotImplementedError


def std(values, ddof=1):
    """Square root of variance(values, ddof)."""
    raise NotImplementedError


def median_absolute_deviation(values):
    """median(|x - median(x)|), unscaled (matches scipy.stats.median_abs_deviation default)."""
    raise NotImplementedError


def coefficient_of_variation(values):
    """Sample standard deviation divided by the mean."""
    raise NotImplementedError


# ---- 02.04 Position, shape and outliers --------------------------------------------------------

def percentile(values, p):
    """p-th percentile (0 <= p <= 100) by linear interpolation at position (n - 1) * p / 100 (NumPy's default)."""
    raise NotImplementedError


def quartiles(values):
    """(Q1, Q2, Q3) using `percentile`."""
    raise NotImplementedError


def iqr(values):
    """Q3 - Q1."""
    raise NotImplementedError


def five_number_summary(values):
    """(min, Q1, median, Q3, max)."""
    raise NotImplementedError


def zscores(values):
    """List of (x - mean) / sample std for every value."""
    raise NotImplementedError


def skewness(values):
    """g1 = mean of ((x - mean) / population std)³ (matches scipy.stats.skew default)."""
    raise NotImplementedError


def iqr_outliers(values, k=1.5):
    """Sorted list of values below Q1 - k·IQR or above Q3 + k·IQR."""
    raise NotImplementedError


# ---- 02.05 Visualising data --------------------------------------------------------------------

def ecdf(values):
    """(sorted values, [1/n, 2/n, ..., 1]) — the points of the empirical CDF."""
    raise NotImplementedError


def histogram_counts(values, bins, low, high):
    """Counts in `bins` equal-width bins from low to high; the last bin includes `high` (like np.histogram).
    Values outside [low, high] are ignored."""
    raise NotImplementedError


# ---- 02.06 Two variables and correlation -------------------------------------------------------

def covariance(x, y, ddof=1):
    """Σ(x - x̄)(y - ȳ) / (n - ddof). Raise ValueError if lengths differ."""
    raise NotImplementedError


def correlation(x, y):
    """Pearson's r = cov(x, y) / (s_x s_y)."""
    raise NotImplementedError


def ranks(values):
    """Ranks starting at 1; tied values share their average rank. ranks([10, 20, 20, 40]) -> [1, 2.5, 2.5, 4]."""
    raise NotImplementedError


def spearman(x, y):
    """Spearman's rank correlation: Pearson's r computed on ranks(x) and ranks(y)."""
    raise NotImplementedError
