# 02.02 — Measures of centre

**Time:** about 3 hours · **Prerequisites:** 01.01, 01.03, 02.01

## Why this matters

"What is a typical value?" is the first question about any variable. The answer is not always the mean. Average
income is dominated by billionaires, an average review score hides whether people love or hate a product, and an
average growth rate can be misleading. Choosing the right centre (and knowing *why* it's right) prevents wrong
conclusions. In ML, the same choice appears as the choice of loss: squared error targets the mean, absolute error
targets the median.

## Learning objectives

- Compute the mean, median and mode by hand and in code.
- Explain how outliers and skew affect the mean and median differently.
- Compute weighted, trimmed and geometric means and say when each is right.
- Choose a measure of centre for a given variable and situation.

## Before you start

Seven employees earn (in thousands per year) 32, 35, 38, 40, 41, 45 and 250 (the founder). The company advertises an
"average salary of 68.7k". Is that honest? Write a better summary.

---

## 1. The mean

$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i.$$

**Worked example 1.** Salaries: Σxᵢ = 32 + 35 + 38 + 40 + 41 + 45 + 250 = 481, n = 7, x̄ = 481 / 7 = **68.71**.

Properties:

- **Balance point.** The deviations from the mean sum to zero (Lesson 01.01): the mean is where a see-saw holding the
  data would balance.
- **Minimises squared error.** No other number m makes Σ(xᵢ − m)² smaller (Lesson 01.03).
- **Uses every value**, so it's efficient, but **one extreme value can drag it a long way**. Six of seven employees
  earn less than the "average".

## 2. The median

The **median** is the middle value of the *sorted* data: half the values are at or below it, half at or above.

1. Sort the data.
2. If n is odd, the median is the value at position (n + 1)/2.
3. If n is even, it's the mean of the two middle values, at positions n/2 and n/2 + 1.

**Worked example 2.** Salaries sorted: 32, 35, 38, **40**, 41, 45, 250. n = 7 is odd, position (7 + 1)/2 = 4,
so median = **40**.

Remove the founder (n = 6): 32, 35, **38, 40**, 41, 45 → median = (38 + 40)/2 = **39**, mean = 231/6 = **38.5**.

| | With founder | Without founder | Change |
| --- | --- | --- | --- |
| Mean | 68.71 | 38.50 | −30.2 |
| Median | 40 | 39 | −1 |

The median barely moves: it is **robust** to outliers. It minimises Σ|xᵢ − m| (absolute error) instead of squared
error. Answer to "Before you start": "median salary 40k; one founder earns 250k" is the honest summary.

## 3. The mode

The **mode** is the most frequent value. It's the only measure of centre for nominal data (the most common
`program` is Science).

**Worked example 3.** 2, 3, 3, 5, 7, 7, 7, 9 → mode **7** (appears 3 times).

- Data can have two modes (*bimodal*) or several. A bimodal distribution, such as review scores clustered at 1 and 5
  stars, often means two different groups are mixed together. The mean (about 3 stars) then describes nobody.
- For continuous data every value is often unique, so the mode of the raw values is useless. Use the peak of a
  histogram instead. `commute_minutes` in `students.csv` has several tied modes (2, 4, 6, …), so its mode tells you
  little.

## 4. Mean vs median and the shape of the data

| Shape | Picture | Relationship |
| --- | --- | --- |
| Symmetric | bell, both tails equal | mean ≈ median |
| Right-skewed (positive skew) | long tail to the right | mean > median |
| Left-skewed (negative skew) | long tail to the left | mean < median |

The mean is pulled towards the long tail. In `students.csv`:

| Variable | Mean | Median | Shape |
| --- | --- | --- | --- |
| exam_score | 63.93 | 64.0 | roughly symmetric |
| sleep_hours | 6.96 | 7.0 | roughly symmetric |
| study_hours | 10.18 | 9.6 | right-skewed |
| commute_minutes | 23.52 | 18.0 | strongly right-skewed |

Rule of thumb: **report the median for skewed data or data with outliers (income, house prices, waiting times);
report the mean for roughly symmetric data, or when totals matter** (total cost = mean × n; a median can't give you
that).

## 5. The weighted mean

When values matter unequally:

$$\bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}.$$

**Worked example 4: grade point average.** Course marks 80 (3 credits), 70 (4 credits), 95 (1 credit).

x̄_w = (3·80 + 4·70 + 1·95) / (3 + 4 + 1) = (240 + 280 + 95) / 8 = 615 / 8 = **76.875**.
The unweighted mean would be (80 + 70 + 95)/3 = 81.67, overstating the result because the 95 came from a 1-credit
course.

Other uses: combining group means (weight by group size), survey weights that correct for over-sampled groups, and
averaging a model's accuracy across classes of different sizes.

**Combining groups.** Science (n = 79) averages 63.11, Arts (72) 64.08, Commerce (49) 65.02. Overall mean =
(79 × 63.11 + 72 × 64.08 + 49 × 65.02) / 200 ≈ 63.93. Averaging the three group means without weights gives 64.07,
which is wrong.

## 6. The trimmed mean

A compromise between mean and median: sort, drop a fixed proportion from **each** end, and average the rest.

**Worked example 5.** 20% trimmed mean of the salaries: 20% of 7 is 1.4, rounded down to 1 value cut from each end.
Remaining: 35, 38, 40, 41, 45 → mean = 199/5 = **39.8**.

Used in judging (drop the highest and lowest scores), in robust statistics, and whenever a few outliers are expected
but you still want to use most of the data. `scipy.stats.trim_mean(x, 0.2)` computes it.

## 7. The geometric mean: for rates and ratios

$$\text{GM} = \left(\prod_{i=1}^n x_i\right)^{1/n} = \exp\!\left(\frac{1}{n}\sum \ln x_i\right).$$

The second form (the mean on the log scale, transformed back) is how you compute it without overflow, using
Lesson 01.02.

**Worked example 6.** An investment rises 50% in year 1, then falls 50% in year 2. The arithmetic mean of the rates
is 0%, suggesting you broke even. But 100 → 150 → 75: you lost 25%.
Growth **factors** are 1.5 and 0.5. GM = √(1.5 × 0.5) = √0.75 = **0.866**, i.e. **−13.4% per year**. Check:
100 × 0.866² = 75 ✓.

Use the geometric mean for growth rates, ratios and multiplicative quantities. It requires all values to be positive.

## 8. Which centre should I use?

| Situation | Use |
| --- | --- |
| Nominal categories | mode |
| Ordinal data | median (and mode) |
| Symmetric numeric data, no extreme outliers | mean |
| Skewed data, outliers, incomes, prices, times | median (or trimmed mean) |
| Values of unequal importance | weighted mean |
| Growth rates, ratios | geometric mean |
| You need a total | mean (total = n × mean) |

**Report more than one** when they differ a lot. The gap between mean and median is itself information about skew.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Finding the median of unsorted data | Sort first |
| Averaging group means without weights | Weight by group size |
| Averaging percentage changes | Use the geometric mean of growth factors |
| Reporting only the mean of skewed data | Report the median too |
| "The average customer" for bimodal data | Describe the two groups separately |

## Summary

- Mean: balance point; minimises squared error; sensitive to outliers.
- Median: middle of sorted data; minimises absolute error; robust.
- Mode: most frequent; the only centre for nominal data.
- Right skew: mean > median. Left skew: mean < median.
- Weighted mean for unequal importance; trimmed mean for robustness; geometric mean for growth.

## Self-test

1. Mean and median of 3, 5, 5, 6, 8, 9, 30. Which better describes a typical value?
2. A class of 20 averages 70 and a class of 30 averages 80. What is the overall average?
3. A price rises 10%, then rises 20%, then falls 30%. What's the average yearly change?
4. In a right-skewed distribution, which is larger, the mean or the median? Why?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
