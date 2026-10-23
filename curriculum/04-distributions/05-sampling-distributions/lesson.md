# 04.05 — Sampling distributions: standard errors, χ², t and F

**Time:** about 4 hours · **Prerequisites:** 04.03, 04.04

## Why this matters

A statistic computed from a sample (a mean, a variance, a proportion, a median) would come out differently with a
different sample. Its distribution over all possible samples is its **sampling distribution**, and the sd of that
distribution is its **standard error**. Every confidence interval and test in Module 05 compares a statistic with its
sampling distribution. Three distributions, derived from the normal, appear again and again: **χ²** (for variances),
**t** (for means when σ is unknown) and **F** (for ratios of variances, as in ANOVA and regression).

## Learning objectives

- Explain sampling distributions and standard errors.
- Compute standard errors for a mean, a proportion and a median.
- Know where the χ², t and F distributions come from and what they're used for.
- Explain why using s instead of σ requires the t distribution, and look up t critical values.

## Before you start

You compute (x̄ − μ)/(s/√n) from a sample of 10, replacing the unknown σ with the sample sd s, and compare it with ±1.96.
How often will it fall outside that range when the null hypothesis is true: 5% of the time, more, or less?

---

## 1. Sampling distributions and standard errors

Imagine taking every possible sample of size n and computing a statistic for each one. The histogram of those values
is the statistic's **sampling distribution**. Its sd is the **standard error** (SE). The SE measures how much the
statistic would vary from sample to sample, i.e. the statistic's *precision*.

| Statistic | Standard error | Estimated in practice by |
| --- | --- | --- |
| mean x̄ | σ/√n | s/√n |
| proportion p̂ | √(p(1 − p)/n) | √(p̂(1 − p̂)/n) |
| difference of two independent means | √(σ₁²/n₁ + σ₂²/n₂) | √(s₁²/n₁ + s₂²/n₂) |
| median (normal data) | ≈ 1.253 σ/√n | (or the bootstrap, Module 05) |

**Simulation** (normal data with μ = 64, σ = 14, n = 10, 200,000 samples): the sd of x̄ is 4.422, against σ/√10 = 4.427.
Each *single* sample's *estimated* SE varies: 2.61, 3.13, 3.98, 5.02, 4.84 for the first five. That variability is
exactly why we'll need the t distribution.

**The median is less precise than the mean** for normal data: its SE is √(π/2) ≈ 1.253 times larger (simulated: 1.254).
The mean is the more *efficient* estimator here. For heavy-tailed data the order reverses (Lesson 02.02, exercise 11).

## 2. The χ² (chi-square) distribution

If Z₁, …, Z_k are independent standard normals, then Z₁² + … + Z_k² follows a **χ² distribution with k degrees of
freedom**: mean k, variance 2k, positive and right-skewed.

**Use 1: the sampling distribution of s².** For normal data,

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}.$$

Simulated with n = 10: mean 9.00 (theory 9) and variance 18.0 (theory 18). P(χ² > 16.92) = 0.0500, matching
`stats.chi2.sf(16.92, 9)`. The n − 1 degrees of freedom are the same n − 1 as in Bessel's correction (Lesson 02.03).

**Variance estimates are noisy.** With n = 10, s² has sd σ²√(2/9) ≈ 92 around the true σ² = 196. You need much more
data to pin down a variance than a mean.

**Use 2:** χ² tests for counts in contingency tables and goodness of fit (Module 05).

## 3. Student's t distribution

When σ is known, (x̄ − μ)/(σ/√n) ~ N(0, 1). In practice σ is unknown and we plug in s. That adds a second source of
randomness, the noise in s, so the statistic

$$T = \frac{\bar{X} - \mu}{s/\sqrt{n}} \sim t_{n-1}$$

follows a **t distribution with n − 1 degrees of freedom**: symmetric and bell-shaped, but with **heavier tails** than
the normal. (It was published in 1908 by W. S. Gosset, a statistician at the Guinness brewery, under the pen name
"Student".)

**Answer to "Before you start"** (n = 10, simulated under the null):

| Statistic and cut-off | P(outside) |
| --- | --- |
| using the true σ, ±1.96 | 0.0499 ✓ |
| using s, ±1.96 | **0.0807**: far too often |
| using s, ±t₉ critical value 2.262 | 0.0490 ✓ |

Using 1.96 with a small sample would make "95%" intervals that cover the truth only about 92% of the time.

**Critical values t_{0.975, df}** (for 95% two-sided):

| df | 1 | 2 | 4 | 9 | 19 | 29 | 99 | ∞ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t | 12.71 | 4.30 | 2.78 | 2.26 | 2.09 | 2.05 | 1.98 | 1.96 |

As n grows, s becomes a precise estimate of σ, and t approaches the normal. The difference matters below about 30
observations. The t distribution's derivation assumes normal data, but it's fairly robust to mild non-normality,
thanks to the CLT.

## 4. The F distribution

The ratio of two independent χ² variables, each divided by its degrees of freedom, follows an **F distribution**
F(d₁, d₂). For two normal samples with equal true variances,

$$\frac{s_1^2}{s_2^2} \sim F(n_1 - 1,\; n_2 - 1).$$

**Simulation:** with n₁ = 8 and n₂ = 13 from the *same* population, the ratio exceeds F₀.₉₅(7, 12) = 2.913 in 4.97% of
samples. One sample variance can easily be nearly three times another by chance alone.

Uses: ANOVA (comparing several group means, Module 05), the overall F-test in regression (Module 06), and tests of
equal variances. Also, t²_df = F(1, df).

## 5. Summary map

```text
                Z ~ N(0, 1)
   ┌──────────────┼───────────────────────┐
 sum of k Z²      Z / sqrt(χ²_k / k)       (χ²_a / a) / (χ²_b / b)
   │              │                        │
  χ²_k           t_k                      F(a, b)
 (variances,     (means with unknown σ)   (ratios of variances,
  count tables)                             ANOVA, regression)
```

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Using 1.96 with small samples and an estimated s | Use the t critical value with n − 1 df |
| Confusing sd (spread of the data) with SE (precision of a statistic) | SE = sd/√n for a mean |
| Trusting a variance from 10 observations | Its relative error is about √(2/(n − 1)) ≈ 47% |
| Using these exact distributions for strongly non-normal small samples | Use the bootstrap or non-parametric methods (Module 05) |

## Summary

- Sampling distribution: the distribution of a statistic across samples; its sd is the standard error.
- SE of a mean σ/√n; of a proportion √(p(1 − p)/n); of a median about 1.253σ/√n for normal data.
- (n − 1)s²/σ² ~ χ²_{n−1}; (x̄ − μ)/(s/√n) ~ t_{n−1}; s₁²/s₂² ~ F(n₁ − 1, n₂ − 1).
- The t has heavier tails than the normal; it matters for n < 30.

## Self-test

1. A sample of 16 has s = 12. What is the estimated SE of the mean? Which critical value gives a 95% interval?
2. What are the mean and variance of a χ² with 20 df?
3. Why does replacing σ by s make the tails heavier?
4. Two samples of sizes 10 and 10 have variances 50 and 20. Is that ratio unusual if the true variances are equal?
   (Use `stats.f.sf`.)

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
