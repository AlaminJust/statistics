# 05.05 — Hypothesis testing

**Time:** about 4 hours · **Prerequisites:** 04.04, 04.05, 05.03

## Why this matters

"Is the new model really better, or did it just get lucky on this test set?" "Does the drug work, or is the difference
noise?" A **hypothesis test** answers: *if there were no real effect, how surprising would data like ours be?* Tests
appear in every empirical paper. They are also misunderstood more often than any other tool in statistics, and that
misunderstanding drives the "replication crisis". This lesson teaches the logic, the correct reading of a p-value, the
two kinds of error, and the permutation test, which needs no formulas at all.

## Learning objectives

- State null and alternative hypotheses and choose a test statistic.
- Compute and correctly interpret a p-value, one- and two-sided.
- Explain significance levels, Type I and Type II errors, and power.
- Run a permutation test from scratch.
- Connect tests and confidence intervals.
- Say what a p-value is **not**.

## Before you start

A coin lands heads 60 times in 100 flips. Your friend says: "The p-value is 0.057, so there's a 5.7% chance the coin is
fair." Is that right?

---

## 1. The logic: proof by (statistical) contradiction

1. **State a null hypothesis H₀**, usually "no effect": the coin is fair (p = 0.5); the two groups have equal means.
2. **State the alternative H₁**: p ≠ 0.5 (two-sided), or p > 0.5 (one-sided, only if decided *before* seeing the data).
3. **Choose a test statistic** that measures how far the data are from H₀, e.g. z = (p̂ − 0.5)/SE.
4. **Find its distribution assuming H₀ is true** (the null distribution): from theory, or by simulation or permutation.
5. **The p-value** = P(a statistic at least as extreme as the observed one | H₀ true).
6. **Small p** means the data would be surprising if H₀ were true, which is evidence against H₀.

Like a court: H₀ is "innocent"; we reject it only with strong evidence. Failing to reject is "not proven", **not**
"proven innocent".

## 2. Worked example 1: the coin

H₀: p = 0.5. H₁: p ≠ 0.5. Under H₀, p̂ ≈ N(0.5, 0.25/100), so the SE is 0.05.

z = (0.60 − 0.50)/0.05 = **2.0**. Two-sided p-value = 2 P(Z > 2) = **0.0455** (normal approximation).

| Method | p-value |
| --- | --- |
| normal approximation | 0.0455 |
| exact binomial test (`stats.binomtest(60, 100, 0.5)`) | **0.0569** |
| simulation: P(\|heads − 50\| ≥ 10) in 10⁶ fair-coin experiments | 0.0568 |
| one-sided exact (H₁: p > 0.5) | 0.0284 |

The normal approximation lacks a continuity correction, so it's a bit small here.

**Answer to "Before you start": no.** The p-value is P(data this extreme | coin fair), **not** P(coin fair | data). Those
are different conditional probabilities: the prosecutor's fallacy again (Lesson 03.04). Getting P(H₀ | data) requires
a prior (Module 07).

## 3. Significance level and decisions

Choose a **significance level α** in advance (conventionally 0.05). Reject H₀ if p < α.

| | H₀ true | H₀ false |
| --- | --- | --- |
| **Reject H₀** | **Type I error** (false positive), probability α | correct: **power** = 1 − β |
| **Don't reject** | correct | **Type II error** (false negative), probability β |

**Worked example 5** (one-sample t-test, n = 20, α = 0.05, 10,000 simulations each):

| True mean (in sd units) | P(reject H₀) | Meaning |
| --- | --- | --- |
| 0 | 0.051 | Type I error rate ≈ α ✓ |
| 0.2 | 0.136 | power: a small effect is usually missed (Type II = 0.86) |
| 0.5 | 0.576 | coin-flip power |
| 0.8 | 0.926 | high power |

Lesson 05.07 shows how to choose n so that power is high enough *before* you collect data.

### p-values are uniform under H₀

**Worked example 2.** For 20,000 experiments where H₀ is true, the p-values are spread evenly over [0, 1]: 0.91% fall
below 0.01, 4.83% below 0.05, 9.50% below 0.10. **That is the definition of a valid test**: with α = 0.05 you'll reject a
true H₀ 5% of the time. When H₀ is false (true mean 0.5, n = 20), p-values pile up near 0, and 56% fall below 0.05
(`p_value_distributions.png`).

Consequence: if you run 20 tests where nothing is going on, expect about one "significant" result (Lesson 05.08).

## 4. The permutation test: a test with no formulas

If H₀ says "the group label doesn't matter", then shuffling the labels shouldn't change anything. So:

1. Compute the observed statistic (e.g. the difference in means).
2. Shuffle the group labels many times, recomputing the statistic each time. This builds the null distribution
   directly from your data.
3. p-value = share of shuffled statistics at least as extreme as the observed one.

**Worked example 3: tutoring.** The observed difference is 8.22 points. In 20,000 shuffles the differences have sd 2.13,
and the largest is 9.09. Permutation p-value = **0.00015** (Welch t-test: 0.00012). The difference is very unlikely to be
chance. (Remember from 02.06 that this is still observational: the test says the gap is real, not that tutoring
*causes* it.)

**A case with no real effect:** Science vs Arts differ by −0.97 points, permutation p = **0.659**, entirely ordinary.
(In `make_students.py` program has no effect on scores.)

Permutation tests assume only *exchangeability* under H₀ (the labels are interchangeable), not normality. They work for
any statistic: medians, correlations, AUC differences. `scipy.stats.permutation_test` does it in one line. Add 1 to the
numerator and denominator ((count + 1)/(B + 1)) so the p-value is never exactly 0.

## 5. Tests and confidence intervals are two views of the same thing

A two-sided test at level α rejects H₀: θ = θ₀ **exactly when** θ₀ lies outside the (1 − α) confidence interval.

**Worked example 4.** The 95% interval for the tutoring difference, [4.14, 12.30], excludes 0, so p < 0.05. The interval
tells you more than the test: it shows the plausible *size* of the effect, not just whether it's zero. **Prefer
reporting intervals; report p-values alongside them.**

## 6. What a p-value is NOT

The American Statistical Association's 2016 statement lists six principles. In short:

| Misreading | Truth |
| --- | --- |
| "p = 0.03 means a 3% chance H₀ is true" | p is computed *assuming* H₀; it isn't a probability of H₀ |
| "p = 0.03 means a 97% chance the effect is real" | same mistake |
| "p > 0.05 means no effect" | absence of evidence ≠ evidence of absence; maybe low power |
| "a smaller p means a bigger or more important effect" | p mixes effect size and sample size; huge n makes tiny effects "significant" |
| "p = 0.049 and p = 0.051 are different conclusions" | the 0.05 line is a convention; report the actual p and the interval |
| "if p < 0.05, the result will replicate" | many significant results from small studies don't |

## 7. One-sided or two-sided?

Use **two-sided** by default. A one-sided test halves the p-value, so it's only legitimate if the direction was chosen
*before* seeing the data and an effect in the other direction would genuinely be treated the same as no effect.
Switching to one-sided after looking is a form of p-hacking (Lesson 05.08).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| p = P(H₀ \| data) | p = P(data this extreme \| H₀) |
| "Accept H₀" | "Fail to reject H₀" |
| Choosing one-sided after seeing the data | Decide the direction in advance |
| Reporting only "p < 0.05" | Report the effect size, interval and exact p |
| Many tests, one α | Correct for multiple testing (05.08) |

## Summary

- H₀ (no effect) vs H₁; statistic; null distribution; p = P(as extreme | H₀).
- Reject if p < α. Type I error = α; power = P(reject | H₀ false).
- Under H₀, p-values are uniform.
- Permutation test: shuffle the labels to build the null distribution; no normality needed.
- Test at α ⇔ the (1 − α) interval excludes θ₀. Prefer intervals.

## Self-test

1. 65 heads in 100 flips: compute z and the two-sided p-value.
2. Define Type I and Type II errors in the context of a spam filter where H₀ = "not spam".
3. Why are p-values uniform when H₀ is true?
4. A study reports p = 0.20. What can and can't you conclude?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
