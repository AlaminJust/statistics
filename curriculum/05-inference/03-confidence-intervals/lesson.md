# 05.03 — Confidence intervals

**Time:** about 4 hours · **Prerequisites:** 04.04, 04.05, 05.01

## Why this matters

A single number ("mean score 63.9", "accuracy 91.2%") hides how uncertain it is. A **confidence interval** gives a
range of plausible values for the parameter, and its width tells the reader how much to trust the estimate. Reviewers
of research papers and ML papers increasingly reject results reported without intervals. This lesson shows how to
build intervals, how to interpret them correctly (most people get this wrong), and which intervals to avoid.

## Learning objectives

- Build z- and t-intervals for a mean, and Welch intervals for a difference of means.
- Build Wald and Wilson intervals for a proportion, and explain why Wilson is better.
- Interpret "95% confidence" correctly, and check coverage by simulation.
- Explain how the level and n affect the width, and plan a sample size for a target margin of error.

## Before you start

A 95% confidence interval for a mean is [62.0, 65.9]. True or false: "There is a 95% probability that the true mean is
between 62.0 and 65.9." Write down your answer and your reasoning.

---

## 1. The general recipe

Most intervals have the same shape:

$$\text{estimate} \;\pm\; \text{critical value} \times \text{standard error}.$$

The critical value comes from the sampling distribution of the estimate (Lesson 04.05): 1.96 for a 95% normal
interval, and t_{0.975, n−1} when σ is estimated. The **margin of error** is critical value × SE.

## 2. The t-interval for a mean

$$\bar{x} \pm t_{1-\alpha/2,\;n-1} \cdot \frac{s}{\sqrt{n}}.$$

**Worked example 1: mean exam score.** x̄ = 63.93, s = 13.84, n = 200, so SE = 0.978. t₀.₉₇₅,₁₉₉ = 1.972.

63.93 ± 1.972 × 0.978 = 63.93 ± 1.93 = **[62.00, 65.86]**.

In code: `stats.t.interval(0.95, n - 1, loc=x.mean(), scale=se)`.

| Level | Interval | Width |
| --- | --- | --- |
| 80% | [62.67, 65.19] | 2.52 |
| 90% | [62.31, 65.55] | 3.23 |
| 95% | [62.00, 65.86] | 3.86 |
| 99% | [61.39, 66.47] | 5.09 |

More confidence requires a wider interval. To make it narrower at the same level, you need more data: the width shrinks
like 1/√n.

**Assumptions:** independent observations, and either roughly normal data or a large enough n for the CLT (Lesson
04.04). For strongly skewed data, use the bootstrap (Lesson 05.04).

## 3. What "95% confidence" means

The **procedure** captures the true parameter in 95% of repeated samples. Any *particular* interval either contains μ
or doesn't. We just don't know which.

**Worked example 2.** Draw 10,000 samples of 15 from N(64, 14²) and build a t-interval from each. **94.6%** contain
μ = 64, close to 95% (`coverage.png` shows 50 of them; the red ones miss). Using 1.96 instead of t₁₄ = 2.145 gives
only **92.8%** coverage: for small n the t-correction matters.

Answer to "Before you start": strictly, **false**, in the frequentist framework. μ is a fixed number, not random, so
"P(μ in [62.0, 65.9]) = 0.95" has no frequentist meaning. Say instead: "We are 95% confident…", meaning the method
works 95% of the time. A Bayesian *credible* interval (Module 07) does support the probability statement, at the price
of specifying a prior. With lots of data and weak priors the two usually agree numerically.

Other wrong readings to avoid:

- "95% of the *data* lie in the interval." No: the interval is about the mean, and it's much narrower than the data's
  spread.
- "A repeat study's mean will fall in this interval 95% of the time." No: that probability is lower, about 83% on
  average, because both studies have noise.

## 4. Intervals for a proportion: avoid the Wald interval

The textbook **Wald** interval p̂ ± z√(p̂(1 − p̂)/n) behaves badly for small n or p near 0 or 1.

**Worked example 3.** 2 successes in 20 (p̂ = 0.10):

| Method | Interval | Problem |
| --- | --- | --- |
| Wald | [−0.032, 0.232] | goes **below 0** |
| **Wilson** | [0.028, 0.301] | stays in [0, 1], asymmetric as it should be |

With 0 successes in 20, the Wald interval is [0, 0], claiming certainty that p = 0. Wilson gives [0, 0.161].

**Coverage** when the true p = 0.1 and n = 20 (20,000 simulations): Wald **87.8%**, Wilson **95.7%**. The "95%" Wald
interval isn't a 95% interval at all.

The **Wilson** interval inverts the score test (Lesson 05.05):

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}.$$

Use it (`statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`) for every proportion, including
model accuracies. For tutoring in `students.csv`, 60/200 = 0.30 gives the Wilson interval **[0.241, 0.367]**.

## 5. The difference between two means: the Welch interval

For independent groups,

$$(\bar{x}_1 - \bar{x}_2) \pm t_{\nu} \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}},$$

where ν is the Welch–Satterthwaite degrees of freedom, which `scipy` computes for you. Welch's version doesn't assume
equal variances; use it by default.

**Worked example 4: tutoring vs not.** Difference 69.68 − 61.46 = 8.22, SE 2.058, ν = 111.9:
**[4.14, 12.30]** points. The interval excludes 0, so the data are inconsistent with "no difference in means" (Lesson
05.05 makes this formal). It still says nothing about *why* the groups differ (Lesson 02.06).

**Overlapping intervals are not a test.** Two separate 95% intervals can overlap while the interval for their
*difference* excludes 0. Always build the interval for the quantity you care about, the difference itself.

## 6. Planning: how big a sample?

For a target margin of error E with a guessed σ: **n = (z σ / E)²**.

| Margin for exam scores (σ ≈ 14) | n |
| --- | --- |
| ± 5 points | 31 |
| ± 2 points | 189 |
| ± 1 point | 753 |

For a proportion, the worst case p = 0.5 gives n = (z × 0.5 / E)². A poll with a ±3-point margin needs about **1,068**
people. That's why national polls survey roughly 1,000 people.

Halving the margin requires **four times** the data.

## 7. Intervals in ML papers

Report every headline metric with an interval:

- accuracy on a test set: a Wilson interval (n = number of test examples);
- AUC, F1 and other complicated metrics: the bootstrap (Lesson 05.04);
- performance across random seeds: a t-interval over the seeds (Module 09).

A 1-point accuracy improvement measured on 500 test examples is usually well inside the noise: the 95% interval for
an accuracy near 90% on 500 examples is about ±2.6 points.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "95% probability μ is in this interval" | 95% of such intervals contain μ |
| 1.96 for small samples with estimated s | t critical value |
| Wald interval for proportions | Wilson |
| Comparing two intervals by overlap | Build the interval for the difference |
| Pooled-variance t when variances differ | Welch by default |

## Summary

- Interval = estimate ± critical value × SE.
- Mean: x̄ ± t s/√n. Difference of means: Welch. Proportion: Wilson.
- "95%" is a property of the procedure: 95% coverage over repeated samples.
- Width ∝ 1/√n; n = (zσ/E)² for a target margin E.
- Report every key result with an interval.

## Self-test

1. A sample of 25 has x̄ = 50 and s = 10. Give the 95% t-interval.
2. 45 of 50 test images are classified correctly. Give a 95% Wilson interval for the accuracy.
3. Why does a 99% interval have to be wider than a 95% one?
4. What sample size gives a ±0.5-point margin for exam scores?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
