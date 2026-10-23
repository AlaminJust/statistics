# 02.04 — Position, shape and outliers

**Time:** about 4 hours · **Prerequisites:** 02.02, 02.03

## Why this matters

"Is a score of 90 good?" depends on everyone else's scores. **Position** measures (percentiles and z-scores) put a
value in context. **Shape** measures (skewness) tell you whether the mean and standard deviation can be trusted as a
summary. **Outlier** rules flag values that might be errors, fraud, or the most interesting cases in the data. In ML,
z-scores are the standard way to scale features, percentiles define robust scalers and clipping thresholds, and outlier
detection is a whole field (anomaly detection).

## Learning objectives

- Compute percentiles and quartiles by hand (linear interpolation) and in code.
- Compute the IQR and the five-number summary, and read a box plot.
- Flag outliers with the 1.5 × IQR rule and with z-scores, and know when each works.
- Compute and interpret z-scores; standardise a variable.
- Measure skewness and use the 68–95–99.7 rule and Chebyshev's inequality.

## Before you start

You scored 90 on an exam where the mean was 64 and the standard deviation 14. Your friend scored 80 on a different exam
with mean 60 and standard deviation 8. Who did better relative to their class?

---

## 1. Percentiles and quartiles

The **p-th percentile** is the value below which p% of the data fall. Special cases:

| Name | Percentile | Notation |
| --- | --- | --- |
| first quartile | 25th | Q1 |
| median | 50th | Q2 |
| third quartile | 75th | Q3 |

There are several conventions for computing percentiles. This course uses **linear interpolation**, the default in
NumPy and pandas:

1. Sort the data: x₍₀₎ ≤ x₍₁₎ ≤ … ≤ x₍ₙ₋₁₎ (positions counted from 0, as in Python).
2. Compute the position h = (n − 1) · p, with p as a proportion (0.25 for Q1).
3. If h is a whole number, the answer is x₍ₕ₎. Otherwise, go the fractional part of the way from x₍⌊h⌋₎ to the next value.

**Worked example 1.** Data (n = 10, already sorted):

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value | 2 | 4 | 4 | 5 | 7 | 9 | 10 | 12 | 15 | 30 |

- **Q1:** h = 9 × 0.25 = 2.25 → between position 2 (value 4) and 3 (value 5), a quarter of the way: 4 + 0.25 × (5 − 4) = **4.25**.
- **Median:** h = 4.5 → halfway between 7 and 9 = **8**.
- **Q3:** h = 6.75 → between 10 and 12: 10 + 0.75 × 2 = **11.5**.
- **90th percentile:** h = 8.1 → 15 + 0.1 × (30 − 15) = **16.5**.

Another common textbook method takes the medians of the lower and upper halves (Q1 = median of 2, 4, 4, 5, 7 = 4 and
Q3 = 12). Small differences between methods are normal. In a report, state which software or method you used.

**Percentile rank** is the reverse: the share of values below a given value. 12 is above 7 of the 10 values, so its
percentile rank is about 70%.

## 2. IQR, five-number summary and box plots

**Interquartile range:** IQR = Q3 − Q1 = 11.5 − 4.25 = **7.25**. It is the spread of the middle 50% of the data, and it's
robust (it ignores the top and bottom quarters).

**Five-number summary:** min, Q1, median, Q3, max = **2, 4.25, 8, 11.5, 30**.

A **box plot** draws it:

```text
          ┌──────┬────────┐
  •───────┤      │        ├──────────•                          ○
          └──────┴────────┘
  2     4.25     8      11.5        15                          30
 min     Q1    median    Q3   whisker end                 outlier
```

- The box spans Q1 to Q3, with a line at the median.
- The whiskers extend to the most extreme values **inside the fences** (next section).
- Points beyond the fences are drawn individually as possible outliers.

## 3. Outliers: the 1.5 × IQR rule

$$\text{lower fence} = Q_1 - 1.5\,\text{IQR}, \qquad \text{upper fence} = Q_3 + 1.5\,\text{IQR}.$$

**Worked example 2.** 1.5 × 7.25 = 10.875. Lower fence = 4.25 − 10.875 = −6.625. Upper fence = 11.5 + 10.875 = **22.375**.
30 > 22.375, so **30 is flagged**. Everything else is inside, so the upper whisker ends at 15.

In `students.csv`, `commute_minutes` has Q1 = 9, Q3 = 33.25, IQR = 24.25 and upper fence 69.6, which flags **11 students**.
That's normal for a right-skewed variable: long tails produce many "outliers" by this rule.

**What to do with an outlier: investigate, don't delete.**

| Cause | Action |
| --- | --- |
| Data-entry error (a height of 1750 cm) | Fix it if possible, otherwise remove it and say so |
| Impossible value (negative age) | Treat as missing |
| Real but extreme (a 108-minute commute) | Keep it. Use robust summaries, or analyse with and without it |
| A different population (a teacher in a student dataset) | Remove it, and document the rule you used |

## 4. z-scores: position in standard-deviation units

$$z_i = \frac{x_i - \bar{x}}{s}.$$

A z-score says how many standard deviations a value is above (+) or below (−) the mean. It has no units, so it lets
you compare values from different scales.

**Worked example 3 (Before you start).** You: z = (90 − 64)/14 = **1.86**. Your friend: z = (80 − 60)/8 = **2.5**.
Your friend did better relative to their class, even with the lower raw score.

**Standardising** a variable means converting every value to its z-score. The result always has mean 0 and standard
deviation 1. Many ML algorithms (k-nearest neighbours, SVMs, PCA, regularised regression, neural networks) work much
better on standardised features. Without it, a feature measured in thousands swamps one measured in fractions. You saw
this in the m01 mini-project: gradient descent converged about 200 times faster on the standardised feature.

**z-score outlier rule:** flag |z| > 3. In worked example 1, x̄ = 9.8 and s = 8.16, so 30 has z = 2.47 and is *not*
flagged. The outlier inflates s, which makes itself look less extreme. That's why the IQR rule, or z-scores based on the
median and MAD, are safer for spotting outliers. A **robust z-score**, (x − median) / (1.4826 × MAD), gives 30 a value of
**3.71**, which is flagged.

## 5. Shape: skewness

**Skewness** measures asymmetry: the average cubed z-score.

$$g_1 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{\hat\sigma}\right)^3, \qquad \hat\sigma = \sqrt{\tfrac{1}{n}\textstyle\sum (x_i - \bar{x})^2}.$$

Cubing keeps the sign, and big deviations dominate. A long right tail makes g₁ positive.

| g₁ | Shape | Mean vs median |
| --- | --- | --- |
| about 0 (−0.5 to 0.5) | roughly symmetric | about equal |
| > 0.5 | right-skewed | mean > median |
| < −0.5 | left-skewed | mean < median |

Worked example 1 has g₁ = **1.59**: strongly right-skewed, driven by the 30.
In `students.csv`: exam_score 0.05 (symmetric), study_hours 0.76 (moderately right-skewed), commute_minutes 1.37
(strongly right-skewed).

`scipy.stats.skew(x)` computes g₁. pandas' `.skew()` applies a small-sample correction and gives slightly larger
values. **Kurtosis** (the average fourth power of z) measures how heavy the tails are; you'll meet it again in Module 04.

## 6. How much data lies within k standard deviations?

**The 68–95–99.7 rule** (for roughly bell-shaped, *normal* data):

| Within | Share of data |
| --- | --- |
| x̄ ± 1s | about 68% |
| x̄ ± 2s | about 95% |
| x̄ ± 3s | about 99.7% |

Check on `students.csv`:

| Variable | Within 1s | Within 2s | Within 3s | Shape |
| --- | --- | --- | --- | --- |
| sleep_hours | 68.0% | 96.0% | 99.5% | symmetric ✓ |
| exam_score | 69.0% | 94.0% | 100% | symmetric ✓ |
| commute_minutes | 76.0% | 93.5% | 99.5% | skewed: the rule fits less well |

**Chebyshev's inequality** holds for *any* distribution: at least 1 − 1/k² of the data lie within k standard
deviations (k > 1). For k = 2, at least 75%; for k = 3, at least 88.9%. It's a weak guarantee, but it never fails.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Deleting outliers automatically | Investigate; document any removal |
| z-score rule on small or skewed data | The outlier inflates s; use the IQR rule or a median/MAD version |
| Expecting identical quartiles from every tool | Methods differ slightly; state yours |
| Applying 68–95–99.7 to skewed data | It's for bell-shaped data; Chebyshev always holds |
| Standardising test data with test-set statistics | In ML, compute x̄ and s on the training data only (Module 08: leakage) |

## Summary

- Percentile: position h = (n − 1)p, interpolate. Q1, median, Q3; IQR = Q3 − Q1.
- Five-number summary → box plot. Outliers lie beyond Q1 − 1.5 IQR or Q3 + 1.5 IQR.
- z = (x − x̄)/s compares values across scales; standardised data have mean 0, sd 1.
- Skewness g₁ = mean cubed z-score; positive means a right tail.
- 68–95–99.7 for normal data; Chebyshev's ≥ 1 − 1/k² for any data.

## Self-test

1. Find Q1, Q3 and the IQR of 3, 5, 5, 6, 8, 9, 30 (linear method). Is 30 an outlier?
2. A value has z = −1.5. What does that mean in words?
3. Why can the z-score rule miss an outlier that the IQR rule catches?
4. Which way is a distribution skewed if its mean is 50 and its median 42?
5. What does Chebyshev guarantee for k = 4?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
