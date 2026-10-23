# 06.03 — Diagnostics and assumptions

**Time:** about 5 hours · **Prerequisites:** 06.01, 06.02, 04.03

## Why this matters

Regression software always produces numbers, even when the model is badly wrong. Diagnostics are how you find out
whether to trust them: whether the relationship is really linear, whether the standard errors are honest, whether one
strange data point is driving everything, and whether your predictors are too entangled to separate. Reviewers ask for
these checks, and in ML the same plots reveal data bugs and model misfit before deployment.

## Learning objectives

- State the regression assumptions and which conclusions depend on each.
- Read the four standard diagnostic plots.
- Detect and handle non-linearity, heteroscedasticity, influential points and multicollinearity.
- Use robust (HC3) standard errors, nested F-tests and VIFs.

## Before you start

You add a near-duplicate of a predictor (study time in minutes, as reported by students) to a model that already has
study hours. What happens to the coefficient and standard error of study hours? And to the model's predictions?

---

## 1. The assumptions, and what breaks when each fails

| Assumption | Needed for | If it fails |
| --- | --- | --- |
| **Linearity**: E[y \| x] is linear in the coefficients | unbiased coefficients, good predictions | systematic errors; fix with transformations, polynomials, splines (06.06) |
| **Independence** of errors | valid SEs, tests, intervals | SEs too small; use clustered SEs or mixed models (06.06) |
| **Constant variance** (homoscedasticity) | valid SEs | SEs wrong; use robust SEs or model the variance |
| **Normality** of errors | exact small-sample t and F tests, prediction intervals | usually minor for large n (CLT); prediction intervals can be off |
| no perfect **multicollinearity** | unique coefficients | XᵀX can't be inverted; near-collinearity inflates SEs |
| **Exogeneity**: errors unrelated to the predictors (no omitted confounders) | causal interpretation | biased coefficients (06.02, Module 09) |

Linearity and exogeneity are about whether the coefficients are *right*. Independence and constant variance are about
whether the *uncertainty* is right. Normality matters least.

## 2. The four diagnostic plots

`diagnostic_plots.png` shows them for `exam_score ~ study_hours + sleep_hours + tutored`:

| Plot | What a good model shows | Warning signs |
| --- | --- | --- |
| residuals vs fitted | a structureless band around 0 | a curve (non-linearity); a funnel (changing variance) |
| normal Q–Q of residuals | points on the line | S-shape (heavy tails), curved ends (skew) |
| scale–location: √\|standardised residual\| vs fitted | flat | upward trend = variance grows with the mean |
| residuals vs leverage | no points far right *and* far from 0 | high-leverage outliers (influential points) |

For the students model everything looks fine: residual skew −0.14, Breusch–Pagan test p = 0.16, the largest Cook's
distance only 0.045.

## 3. Non-linearity

**Simulation.** y = 5 + 0.8x² + noise, fitted with a straight line. R² = 0.886 looks great, but the mean residual by
x-range is **+6.5, −4.5, −6.4, −2.2, +6.5**: a U-shape. The line overpredicts in the middle and underpredicts at both
ends. Adding x² raises R² to 0.952 and removes the pattern.

**A high R² doesn't mean the functional form is right.** Look at the residual plot. Fixes: transform x or y (logs for
multiplicative effects, Lesson 01.02), add polynomial terms, or use splines (Lesson 06.06).

## 4. Heteroscedasticity and robust standard errors

**Simulation.** Spending vs income where the noise grows with income (sd ∝ income²), as is typical of money variables.
The Breusch–Pagan test strongly rejects constant variance (p ≈ 10⁻¹⁹).

| SE type | slope SE | Coverage of the "95%" interval (2,000 simulations) |
| --- | --- | --- |
| classic OLS | 0.028 | **89.1%** |
| robust (HC3) | 0.033 | **93.6%** |

The coefficient is still unbiased, but classic SEs are too small, so intervals are too narrow and tests too eager. 
**Heteroscedasticity-consistent (robust) SEs** fix the uncertainty without changing the estimate:

```python
smf.ols("spend ~ income", data).fit(cov_type="HC3")
```

Many applied researchers use robust SEs by default. Alternatives: model log(spend) (which often stabilises the
variance), or weighted least squares if you know how the variance changes.

## 5. Outliers, leverage and influence

| Concept | Meaning | Measure |
| --- | --- | --- |
| **outlier** | a large residual: y is unusual given x | standardised (studentised) residual, \|r\| > 3 |
| **leverage** | an unusual x value, far from the others | hᵢᵢ, the diagonal of the hat matrix X(XᵀX)⁻¹Xᵀ; average is (p + 1)/n |
| **influence** | removing the point changes the fit a lot | **Cook's distance**; values near or above 1 deserve a look |

A point needs *both* high leverage and a large residual to be very influential.

**Worked example 4.** Add one data-entry error, a student with 60 study hours and a score of 20, to the 200 students:

| | Slope |
| --- | --- |
| without the point | 2.048 |
| with it | **1.054**: halved by one row out of 201 |

The point's leverage is 0.34 (average 0.01) and its Cook's distance is **22.7** (the next largest is 0.05). As in
Lesson 02.04: investigate influential points, fix errors, and if the point is real, report results with and without it.

## 6. Multicollinearity

When predictors are highly correlated, the data can't tell their effects apart. Coefficients become unstable and their
SEs explode, **even though predictions stay just as good**.

The **variance inflation factor** VIF_j = 1/(1 − R_j²), where R_j² comes from regressing predictor j on the others,
says how much predictor j's coefficient variance is inflated. Rules of thumb: VIF > 5 deserves attention, > 10 is
serious.

**Worked example 5 (Before you start).** Add `study_minutes_reported` = 60 × hours + noise:

| | VIF | study_hours coefficient (SE) |
| --- | --- | --- |
| without the duplicate | 1.0 | 2.048 (0.125) |
| with the duplicate | **101** | **3.278 (1.332)**: SE more than 10 times larger |

R² barely moves (0.566 vs 0.565): predictions are fine, but individual coefficients become meaningless. Fixes: drop or
combine redundant predictors, use subject knowledge to choose, or use ridge regression (Lesson 06.05). For pure
prediction, multicollinearity often doesn't matter.

## 7. Testing coefficients and groups of coefficients

- **t-test for one coefficient** (H₀: βⱼ = 0): t = β̂ⱼ/SE(β̂ⱼ), with n − p − 1 df, printed by `summary()`.
- **Overall F-test** (H₀: all slopes are 0): F = 104.3, p ≈ 10⁻⁴⁰ for the students model.
- **Nested F-test**: does a *group* of predictors add anything? Compare the full model with a reduced one:

$$F = \frac{(\text{SSE}_{\text{reduced}} - \text{SSE}_{\text{full}})/q}{\text{SSE}_{\text{full}}/(n - p - 1)},$$

  where q is the number of extra coefficients. Do sleep and tutoring add to study hours? F = 21.6, p = 3 × 10⁻⁹:
  yes. (`full.compare_f_test(reduced)`; it's the way to test a categorical predictor with several dummies as a whole.)

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Checking only R² | Always plot residuals vs fitted |
| Testing the normality of y itself | Check the *residuals* |
| Deleting points with big residuals | Investigate; report with and without; use robust methods |
| Classic SEs with a funnel-shaped residual plot | Robust (HC3) SEs |
| Interpreting coefficients with VIF ≈ 100 | Remove or combine collinear predictors |

## Summary

- Linearity and exogeneity → correct coefficients; independence and constant variance → correct SEs; normality → exact small-sample tests.
- Four plots: residuals vs fitted, Q–Q, scale–location, residuals vs leverage.
- Curves → transform or add terms; funnels → robust SEs; influential points → Cook's distance and investigation.
- VIF = 1/(1 − R_j²) measures multicollinearity: unstable coefficients, unchanged predictions.
- Nested F-tests compare groups of predictors.

## Self-test

1. A residual-vs-fitted plot shows a funnel opening to the right. What's wrong, and what's the quickest fix for inference?
2. Why can a point with a small residual still be influential?
3. A predictor has VIF = 20. What does that mean for its SE?
4. When does non-normality of the residuals matter?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
