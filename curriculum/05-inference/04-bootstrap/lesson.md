# 05.04 — The bootstrap

**Time:** about 4 hours · **Prerequisites:** 05.01, 05.03

## Why this matters

Formulas for standard errors exist only for simple statistics: means, proportions, regression coefficients. What's the
SE of a median? Of a correlation? Of a model's F1 score or AUC? The **bootstrap** (Efron, 1979) answers all of these
with one idea: **resample your own data, recompute the statistic each time, and look at how much it varies.** It is
probably the single most useful technique in applied statistics and ML evaluation, and it's about ten lines of code.

## Learning objectives

- Explain the bootstrap principle and draw resamples with replacement.
- Compute bootstrap standard errors and percentile intervals.
- Use BCa intervals (`scipy.stats.bootstrap`) for skewed statistics.
- Bootstrap correctly for paired data and for ML test-set metrics.
- Recognise when the bootstrap fails.

## Before you start

You have one sample of 200 commute times and want a 95% interval for their *median*. There's no simple formula. How
could you find out how much the median would vary from sample to sample, using only the data you have?

---

## 1. The idea

If we could draw many samples from the population, we'd compute the statistic on each and look at its spread. We
can't, but our sample is our best picture of the population. So we **treat the sample as the population** and draw
"new samples" from it:

1. From your n observations, draw n **with replacement**: a *bootstrap resample*. Some points appear twice or more,
   and some not at all.
2. Compute the statistic on the resample.
3. Repeat B times (B = 2,000–10,000).
4. The spread of the B values estimates the sampling variability.

**Worked example 1.** Data (3, 7, 8, 12, 20). Three resamples, sorted: (7, 7, 7, 8, 8), (3, 3, 8, 12, 20),
(7, 7, 8, 12, 12). Each resample has the same size as the original. On average a resample contains about 63% of the
distinct original points.

```python
def bootstrap(data, statistic, n_boot=10_000, rng=rng):
    n = len(data)
    idx = rng.integers(0, n, size=(n_boot, n))       # n_boot resamples of n row indices
    return np.array([statistic(data[i]) for i in idx])
```

## 2. Bootstrap standard errors and percentile intervals

- **Bootstrap SE** = the standard deviation of the B bootstrap statistics.
- **Percentile interval** = the 2.5th and 97.5th percentiles of the B bootstrap statistics.

**Worked example 2: check against a known formula.** Mean exam score:

| Method | SE | 95% interval |
| --- | --- | --- |
| formula s/√n and t | 0.978 | [62.00, 65.86] |
| bootstrap, B = 10,000 | 0.966 | [62.02, 65.75] |

They agree, so the bootstrap passes the sanity check. Its real value is for statistics with no formula.

**Worked example 3: median commute** (answering "Before you start"). Median 18 minutes, bootstrap SE **1.32**,
percentile interval **[15, 20]**. Only 19 distinct medians appear across 10,000 resamples, because the data are whole
minutes. That's fine, but it means the interval endpoints jump in steps.

## 3. Better intervals for skewed statistics: BCa

The percentile interval can be off when the bootstrap distribution is skewed or biased. The **BCa** (bias-corrected and
accelerated) interval adjusts the percentiles for both:

```python
res = stats.bootstrap((data,), np.mean, n_resamples=10_000, method="BCa", random_state=1)
res.confidence_interval
```

**Worked example 3b: mean commute** (right-skewed data):

| Method | 95% interval |
| --- | --- |
| t-interval | [20.67, 26.37], forced to be symmetric |
| percentile | [20.75, 26.39] |
| BCa | **[20.86, 26.57]**, shifted right |

With right-skewed data, the uncertainty is larger on the high side, and BCa captures that. BCa can fail for
degenerate cases (it needs a jackknife step, which breaks down for medians of heavily tied data); then fall back to
the percentile interval.

## 4. Resample the right unit

The bootstrap must copy how the data were *generated*: resample **independent units**.

- **Correlation, or any statistic of pairs: resample rows (pairs), never each variable separately.** Shuffling the
  variables independently would destroy the relationship you're measuring.
  **Worked example 4.** r(study, score) = 0.728, bootstrap SE 0.033, interval **[0.658, 0.786]**. The Fisher-z formula
  gives [0.655, 0.787].
- **Clustered data** (several rows per student, patient or school): resample whole clusters.
- **Time series**: resample blocks of consecutive observations (the *block bootstrap*).
- **Two independent groups**: resample within each group separately.

## 5. The bootstrap for ML metrics

The test set is a sample, so test metrics are statistics with sampling error. **Worked example 5:** logistic regression
on scikit-learn's breast-cancer data, test set n = 171. Resample the *test examples* (keeping each prediction with its
label):

| Metric | Value | 95% bootstrap interval |
| --- | --- | --- |
| accuracy | 0.959 | [0.924, 0.988] |
| F1 | 0.967 | [0.940, 0.990] |
| AUC | 0.996 | [0.989, 1.000] |

An accuracy of "95.9%" on 171 examples is honestly "somewhere between about 92% and 99%". A competing model at 96.5%
is not shown to be better by these numbers. Lesson 05.08 compares two models properly: by bootstrapping the
*difference* on the same resamples (a paired bootstrap).

This interval covers only **test-set sampling** noise. It ignores the variability from training (random seeds,
different training data), which Module 09 addresses.

## 6. When the bootstrap fails

| Situation | Why |
| --- | --- |
| extremes (max, min) | Resamples can never go beyond the sample's extreme. **Worked example 6:** 63.6% of bootstrap maxima *equal* the sample maximum (theory 1 − (1 − 1/n)ⁿ ≈ 0.636) |
| very small n (< about 15) | The sample is a poor picture of the population |
| dependent data resampled as if independent | Underestimates the SE |
| heavy tails with infinite variance | Like the CLT, it relies on finite variance |
| a statistic that isn't smooth in the data | e.g. the number of distinct values |

The bootstrap is not magic: it can't create information. It tells you how much your statistic would vary *if your
sample were representative*.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Resampling without replacement | With replacement, same size n |
| Resampling variables of a pair separately | Resample rows |
| B = 100 | Use B ≥ 2,000 for intervals (10,000 is cheap) |
| Treating the bootstrap distribution's centre as a better estimate | Report the original statistic; the bootstrap measures spread |
| Bootstrapping the max | Use theory or a parametric model |
| Forgetting the seed | Results should be reproducible |

## Summary

- Resample n observations with replacement; recompute; repeat B times.
- Bootstrap SE = sd of the replicates; percentile interval = their 2.5% and 97.5% quantiles; BCa for skewed statistics.
- Resample the independent unit: pairs, clusters, blocks.
- Use it for medians, correlations, ML metrics, and anything without a formula.
- It fails for extremes, tiny samples and dependent data treated as independent.

## Self-test

1. Why must bootstrap resamples be drawn with replacement?
2. How would you bootstrap a 95% interval for the difference in median exam score between tutoring groups?
3. Why does bootstrapping the sample maximum fail?
4. Which source of variability does a test-set bootstrap *not* capture?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
