# 06.01 — Simple linear regression

**Time:** about 4 hours · **Prerequisites:** 01.03, 02.06, 04.03, 05.02, 05.03

## Why this matters

Regression is the bridge from statistics to machine learning. It predicts a numeric outcome from inputs, it estimates
how much the outcome changes with each input, and it quantifies the uncertainty of both. Linear regression is the first
baseline for any prediction problem, the last layer of most neural networks, and the basis of most causal-effect
estimates in research. This lesson does the one-predictor case thoroughly; Lesson 06.02 generalises it.

## Learning objectives

- Fit a least-squares line by hand and with statsmodels, and interpret its slope and intercept.
- Explain residuals, R² and the residual standard error.
- Distinguish a confidence interval for the mean from a prediction interval for a new observation.
- Explain why regressing y on x differs from regressing x on y.
- Recognise regression to the mean and the dangers of extrapolation.

## Before you start

The top 10% of students on a first test take a second, equally hard test. With no tutoring or changes at all, do you
expect their average to go up, stay the same, or go down?

---

## 1. The model

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \text{ independent}, \; E[\varepsilon_i] = 0, \; \operatorname{Var}(\varepsilon_i) = \sigma^2.$$

- β₀ (intercept): the mean of y when x = 0.
- β₁ (slope): how much the **mean** of y changes for a one-unit increase in x.
- εᵢ (error): everything about yᵢ not explained by xᵢ.

The fitted line ŷ = b₀ + b₁x estimates the population line E[y | x] = β₀ + β₁x. For bivariate normal data this
conditional mean is exactly linear (Lesson 04.03).

## 2. Least squares

Choose b₀ and b₁ to minimise the sum of squared residuals, SSE = Σ(yᵢ − b₀ − b₁xᵢ)². Setting both partial derivatives
to zero (Lesson 01.03) gives:

$$b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = r\,\frac{s_y}{s_x}, \qquad b_0 = \bar{y} - b_1\bar{x}.$$

The line always passes through (x̄, ȳ). With normal errors, least squares is the maximum-likelihood fit (Lesson 05.02).

**Worked example 1: five points.** x = (2, 4, 6, 8, 10), y = (50, 60, 65, 80, 85), x̄ = 6, ȳ = 68.

- Sxy = Σ(x − x̄)(y − ȳ) = 180 (computed in Lesson 02.06) and Sxx = Σ(x − x̄)² = 40.
- b₁ = 180/40 = **4.5** points per hour; b₀ = 68 − 4.5 × 6 = **41**.
- Fitted values 50, 59, 68, 77, 86; residuals 0, 1, −3, 3, −1. **Residuals always sum to zero.**

## 3. How good is the fit?

| Quantity | Formula | Meaning |
| --- | --- | --- |
| SST | Σ(yᵢ − ȳ)² | total variation in y |
| SSE | Σ(yᵢ − ŷᵢ)² | variation left unexplained |
| **R²** | 1 − SSE/SST | share of the variance in y explained by the line; equals r² in simple regression |
| residual standard error | √(SSE/(n − 2)) | typical size of a prediction error, in y's units |

Worked example 1: SSE = 20, SST = 830, so R² = **0.976** = r² (r = 0.988).

**Worked example 2: students.csv.**

| | Estimate | SE | 95% CI |
| --- | --- | --- | --- |
| intercept | 43.09 | 1.55 | [40.04, 46.15] |
| study_hours | **2.048** | 0.137 | **[1.78, 2.32]** |

R² = **0.53**; residual standard error **9.5 points**.

Interpretation: *each additional weekly study hour is associated with about 2.0 more exam points on average (95% CI 1.8
to 2.3).* The true value used to generate the data, 2.2, is inside the interval. Study hours explain about half the
variation in scores; individual predictions are typically off by about 9.5 points.

The intercept, 43 points for zero study hours, is an extrapolation: the minimum observed is 1.8 hours. Intercepts are
often not meaningful on their own.

## 4. Two kinds of interval

At x = 15 hours, the predicted score is **73.8**.

| Interval | Question | 95% interval |
| --- | --- | --- |
| **confidence interval for the mean** | What's the *average* score of all students who study 15 hours? | [71.9, 75.7] |
| **prediction interval** | What will *one* new student who studies 15 hours score? | **[55.0, 92.7]** |

The prediction interval adds the individual noise σ² to the uncertainty about the line, so it is always much wider and it
doesn't shrink to zero as n grows. Use it whenever you predict individual outcomes.

**Extrapolation is dangerous.** The line predicts 165.9 points for 60 study hours, which is impossible on a 100-point
exam. A model is only supported by data inside the range it was fitted on.

## 5. y on x is not x on y

| Regression | Slope |
| --- | --- |
| score on hours | 2.048 points per hour |
| hours on score | **0.259** hours per point, not 1/2.048 = 0.488 |

Each regression minimises vertical errors in its own outcome, and both slopes are shrunk by r:
b(y on x) = r s_y/s_x and b(x on y) = r s_x/s_y. Only when r = ±1 are the two lines the same. **Choose the outcome you
want to predict as y.**

## 6. Regression to the mean

Since the standardised slope is r, with |r| < 1, a value 2 sd above average on x predicts only 2r sd above average on y.

**Worked example 5 (Before you start).** Simulate test scores = ability + noise, twice. The top 10% on test 1 averaged
**83.0**, but on test 2 the same students averaged **73.7**. The bottom 10% rose from 38.0 to **46.8**. Nobody changed;
the extreme scores were partly luck, and luck doesn't repeat.

This fools people constantly: the "Sports Illustrated cover jinx", "punishment works better than praise" (in pilot
training, praise after a great landing is followed by a worse one either way), and treatments given to the
worst-performing patients or schools that "work" with no effect at all. A randomised control group is the only reliable
protection (Module 09).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "Each hour *causes* 2 more points" from observational data | "is associated with" (Module 09 for causation) |
| Using the confidence interval to predict an individual | Use the prediction interval |
| Predicting far outside the data range | Don't extrapolate |
| Inverting the y-on-x line to predict x | Fit x on y |
| A high R² means a good model for any purpose | Check residuals (06.03) and whether the relationship is causal |

## Summary

- ŷ = b₀ + b₁x with b₁ = r s_y/s_x and b₀ = ȳ − b₁x̄; least squares = normal MLE.
- R² = 1 − SSE/SST (= r²); residual SE √(SSE/(n − 2)).
- CI for the mean response vs a (wider) prediction interval for a new observation.
- Regressions of y on x and x on y differ; regression to the mean is everywhere.

## Self-test

1. For x̄ = 10, ȳ = 50, r = 0.6, s_x = 2 and s_y = 8, write the regression line of y on x.
2. Why do residuals from least squares always sum to zero?
3. A model has R² = 0.53. What does the remaining 47% consist of?
4. Explain regression to the mean to someone who thinks a new coach improved the worst team.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
