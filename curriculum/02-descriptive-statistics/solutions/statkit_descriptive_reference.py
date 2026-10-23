"""Reference implementation of statkit/descriptive.py. Compare with yours AFTER your tests pass."""

import math


# 02.01
def frequency_table(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def relative_frequencies(values):
    counts = frequency_table(values)
    n = sum(counts.values())
    return {key: count / n for key, count in counts.items()}


# 02.02
def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of empty data")
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty data")
    middle = n // 2
    if n % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def mode(values):
    counts = frequency_table(values)
    top = max(counts.values())
    return sorted(value for value, count in counts.items() if count == top)


def weighted_mean(values, weights):
    if len(values) != len(weights):
        raise ValueError("values and weights have different lengths")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("weights sum to zero")
    return sum(w * x for x, w in zip(values, weights)) / total_weight


def trimmed_mean(values, proportion):
    ordered = sorted(values)
    cut = int(proportion * len(ordered))
    kept = ordered[cut:len(ordered) - cut]
    return mean(kept)


def geometric_mean(values):
    if any(v <= 0 for v in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(mean([math.log(v) for v in values]))


# 02.03
def data_range(values):
    return max(values) - min(values)


def variance(values, ddof=1):
    values = list(values)
    n = len(values)
    if n <= ddof:
        raise ValueError(f"need more than {ddof} values")
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (n - ddof)


def std(values, ddof=1):
    return math.sqrt(variance(values, ddof))


def median_absolute_deviation(values):
    m = median(values)
    return median([abs(x - m) for x in values])


def coefficient_of_variation(values):
    return std(values) / mean(values)


# 02.04
def percentile(values, p):
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    h = (len(ordered) - 1) * p / 100
    lower = math.floor(h)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])


def quartiles(values):
    return percentile(values, 25), percentile(values, 50), percentile(values, 75)


def iqr(values):
    q1, _, q3 = quartiles(values)
    return q3 - q1


def five_number_summary(values):
    q1, q2, q3 = quartiles(values)
    return min(values), q1, q2, q3, max(values)


def zscores(values):
    m, s = mean(values), std(values)
    return [(x - m) / s for x in values]


def skewness(values):
    m = mean(values)
    sigma = std(values, ddof=0)
    return mean([((x - m) / sigma) ** 3 for x in values])


def iqr_outliers(values, k=1.5):
    q1, _, q3 = quartiles(values)
    spread = q3 - q1
    low, high = q1 - k * spread, q3 + k * spread
    return sorted(x for x in values if x < low or x > high)


# 02.05
def ecdf(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered, [(i + 1) / n for i in range(n)]


def histogram_counts(values, bins, low, high):
    width = (high - low) / bins
    counts = [0] * bins
    for x in values:
        if x < low or x > high:
            continue
        index = min(int((x - low) / width), bins - 1)    # x == high goes in the last bin
        counts[index] += 1
    return counts


# 02.06
def covariance(x, y, ddof=1):
    if len(x) != len(y):
        raise ValueError("x and y have different lengths")
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - ddof)


def correlation(x, y):
    return covariance(x, y) / (std(x) * std(y))


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1          # positions i..j (0-based) share ranks i+1..j+1
        for k in range(i, j + 1):
            result[order[k]] = average_rank
        i = j + 1
    return result


def spearman(x, y):
    return correlation(ranks(x), ranks(y))
