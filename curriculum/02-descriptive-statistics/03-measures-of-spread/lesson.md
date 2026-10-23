# 02.03 — Measures of spread

**Time:** about 4 hours · **Prerequisites:** 01.01, 02.02

## Why this matters

Two classes can have the same average score and be completely different: in one everyone scored about 75, in the other
half failed and half excelled. A centre without a spread is half a summary. Spread also drives everything later in the
course:

- the standard deviation sets the width of confidence intervals and the size of test statistics (Module 05);
- variance is the "V" in the bias–variance trade-off (Module 08);
- standardising features by their standard deviation is a standard ML preprocessing step;
- risk in finance is the standard deviation of returns.

## Learning objectives

- Compute the range, variance, standard deviation, median absolute deviation and coefficient of variation.
- Explain why variance squares the deviations.
- Explain why the sample variance divides by n − 1, and show it by simulation.
- Choose a robust or non-robust measure of spread.
- Interpret the standard deviation in the units of the data.

## Before you start

Class A scored 70, 72, 75, 78, 80. Class B scored 50, 60, 75, 90, 100. Both have mean 75. Invent a single number that
captures how different they are. Then compare it with the standard deviation in section 3.

---

## 1. The range

Range = max − min. Class A: 80 − 70 = **10**. Class B: 100 − 50 = **50**.

Easy, but it uses only two values, so **one outlier changes it completely**, and it tends to grow with sample size (a
bigger sample is more likely to contain an extreme value).

## 2. Why deviations must be squared (or made positive)

The natural idea, "average distance from the mean", fails if you use signed deviations, because they **always sum to
zero** (Lesson 01.01). You must remove the signs:

| Option | Formula | Measure |
| --- | --- | --- |
| Square the deviations | (xᵢ − x̄)² | variance, standard deviation |
| Take absolute values | \|xᵢ − x̄\| | mean absolute deviation |
| Absolute deviations from the median, then take their median | median \|xᵢ − median\| | median absolute deviation (MAD) |

Squaring dominates statistics because it gives clean mathematics: smooth derivatives (Lesson 01.03), the link to the
normal distribution, and variances of independent quantities that add up (Module 03).

## 3. Variance and standard deviation

**Sample variance:**

$$s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2.$$

**Sample standard deviation:** s = √s². It's back in the **original units** (points, hours, dollars), so it's the one
you report and interpret. Variance is in squared units ("points²"), which nobody can picture.

**Worked example 1, step by step: class A.** x̄ = 75.

| xᵢ | xᵢ − x̄ | (xᵢ − x̄)² |
| --- | --- | --- |
| 70 | −5 | 25 |
| 72 | −3 | 9 |
| 75 | 0 | 0 |
| 78 | 3 | 9 |
| 80 | 5 | 25 |
| **sum** | **0** ✓ | **68** |

s² = 68 / (5 − 1) = **17**, s = √17 ≈ **4.12 points**.

**Class B:** deviations −25, −15, 0, 15, 25 → squares 625, 225, 0, 225, 625 → sum 1700.
s² = 1700 / 4 = **425**, s ≈ **20.62 points**.

Class B's standard deviation is five times class A's: that's the number "Before you start" asked for.

**Reading a standard deviation.** "Scores typically sit about 4 points from the average" (A) vs "about 21 points" (B).
For roughly bell-shaped data, about 68% of values fall within 1 s of the mean and 95% within 2 s (Lesson 02.04).

**The shortcut formula.** From Lesson 01.01, Σ(xᵢ − x̄)² = Σxᵢ² − n x̄². For the data 4, 7, 1, 8: Σxᵢ² = 130, n x̄² = 100,
so the sum of squares is 30, s² = 30/3 = **10**, s ≈ **3.162**.
(In code, prefer the two-pass formula: the shortcut can lose accuracy with large numbers.)

## 4. Why n − 1? (Bessel's correction)

There are two versions:

| | Divides by | Use when | NumPy | pandas default |
| --- | --- | --- | --- | --- |
| Population variance σ² | n | you have the **whole population** | `np.var(x)` (ddof=0) | |
| Sample variance s² | n − 1 | you have a **sample** and want to estimate σ² | `np.var(x, ddof=1)` | `df.var()` (ddof=1) |

`ddof` means "delta degrees of freedom": the divisor is n − ddof.

**Watch out:** NumPy defaults to n, pandas defaults to n − 1. The same data give different answers. For 4, 7, 1, 8:
`np.var` = 30/4 = **7.5** and `pd.Series.var` = 30/3 = **10**.

**Why n − 1 is right for samples.** We measure deviations from x̄, not from the unknown true mean μ. Since x̄ is
computed from the same data, it sits closer to the data than μ does. In fact x̄ is the number that *minimises*
Σ(xᵢ − m)² (Lesson 01.03). So Σ(xᵢ − x̄)² is systematically **too small**, and dividing by n underestimates σ².
Dividing by the slightly smaller n − 1 corrects this exactly.

**See it by simulation** (`examples.py`): draw 100,000 samples of size 5 from a population with σ² = 4.

| Divisor | Average of the 100,000 variance estimates | True σ² |
| --- | --- | --- |
| n = 5 | **3.21** | 4 |
| n − 1 = 4 | **4.01** | 4 |

Dividing by n is wrong by the factor (n − 1)/n = 4/5 on average: 4 × 0.8 = 3.2 ✓. An estimator whose average equals the
true value is **unbiased** (Module 05). The difference matters for small n and vanishes for large n.

**Degrees of freedom intuition.** Once you know x̄, only n − 1 deviations are free: the last one is forced, because
they must sum to zero. So there are n − 1 independent pieces of information about spread.

## 5. Robust spread: the median absolute deviation (MAD)

$$\text{MAD} = \text{median}\big(|x_i - \text{median}(x)|\big).$$

**Worked example 2.** For 4, 7, 1, 8: median = (4 + 7)/2 = 5.5. Absolute deviations: 1.5, 1.5, 4.5, 2.5. Sorted:
1.5, 1.5, 2.5, 4.5 → MAD = (1.5 + 2.5)/2 = **2.0**.

Like the median, the MAD ignores outliers. For normal data, 1.4826 × MAD estimates σ (that's what
`scipy.stats.median_abs_deviation(x, scale="normal")` reports). Use it to detect outliers without letting the outliers
inflate the yardstick.

| Data | s | MAD |
| --- | --- | --- |
| 32, 35, 38, 40, 41, 45 | 4.59 | 3.0 |
| same + 250 | 80.05 | 5.0 |

One outlier multiplies the standard deviation by about 17, while the MAD only moves from 3 to 5.

## 6. Coefficient of variation: comparing spreads on different scales

$$\text{CV} = \frac{s}{\bar{x}}.$$

A unitless ratio, used to compare variability of things measured in different units or at different levels.

**Worked example 3.** In `students.csv`, study hours have s = 4.92 and mean 10.18, so CV = **0.48**. Sleep hours have
s ≈ 1.0 and mean 6.96, so CV ≈ **0.14**. Relative to their typical values, study time varies about three times as much as
sleep.

Use CV only for **ratio-scale, positive** data. For temperatures in °C (mean can be 0 or negative) it's meaningless.

## 7. How spread responds to transformations

| Transformation | Mean | Standard deviation | Variance |
| --- | --- | --- | --- |
| Add c to every value | + c | unchanged | unchanged |
| Multiply every value by a | × a | × \|a\| | × a² |

Shifting data doesn't change how spread out they are; rescaling does. Converting minutes to hours divides the standard
deviation by 60 and the variance by 3600.

## 8. Which spread should I use?

| Situation | Use |
| --- | --- |
| Roughly symmetric data, no wild outliers | standard deviation (with the mean) |
| Skewed data or outliers | IQR (Lesson 02.04) or MAD (with the median) |
| Comparing variability across different scales | coefficient of variation |
| A quick look | range (with care) |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Mixing `np.var` (÷ n) and `pd.var` (÷ n − 1) | Always set `ddof` explicitly |
| Reporting variance as "the spread" | Report the standard deviation, which has the data's units |
| Mean ± s for very skewed data | Use median and IQR |
| sd(X + Y) = sd(X) + sd(Y) | Variances add for *independent* variables, not sds (Module 03) |
| CV for data with mean near 0 | CV only for positive ratio data |

## Summary

- Range: max − min; fragile.
- s² = Σ(xᵢ − x̄)² / (n − 1); s = √s² in the data's units.
- n − 1 corrects the underestimate caused by measuring from x̄; dividing by n is correct only for a full population.
- MAD: robust spread. CV = s / x̄: unitless comparison.
- Adding a constant doesn't change spread; multiplying by a scales the sd by |a|.

## Self-test

1. Compute s for 2, 4, 4, 4, 5, 5, 7, 9.
2. Why does Σ(xᵢ − x̄) not work as a measure of spread?
3. In one sentence, why do we divide by n − 1?
4. Temperatures have s = 5 °C. What is s in °F (°F = 1.8 °C + 32)?
5. Which of s and MAD would you use to describe the spread of house prices?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
