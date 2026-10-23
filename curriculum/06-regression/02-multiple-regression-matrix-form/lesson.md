# 06.02 — Multiple regression in matrix form

**Time:** about 5 hours · **Prerequisites:** 01.05, 06.01

## Why this matters

Real outcomes depend on many things at once. Multiple regression estimates the effect of each input **holding the
others fixed**. That's how researchers adjust for confounders, and it's the starting point of every linear model in ML
(a linear layer is exactly X**β**). Matrix notation makes the whole theory fit in one line, **β̂** = (XᵀX)⁻¹Xᵀ**y**,
and shows where standard errors, dummy variables and interactions come from.

## Learning objectives

- Write a regression as **y** = X**β** + **ε**, and solve the normal equations.
- Interpret coefficients as "holding the other predictors constant".
- Explain omitted-variable bias and why adjustment changes coefficients.
- Encode categorical predictors with dummy variables, and interpret interactions.
- Compare models with adjusted R², AIC and BIC, and know their limits.

## Before you start

Tutored students score 8.2 points higher on average. If you also account for how much they study and sleep, do you
expect the tutoring gap to grow, shrink, or stay the same?

---

## 1. The model in matrix form

With n observations and p predictors:

$$\mathbf{y} = X\boldsymbol\beta + \boldsymbol\varepsilon, \qquad X = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ \vdots & & & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}.$$

X is n × (p + 1): one row per observation, a first column of 1s for the intercept. Minimising ‖**y** − X**β**‖² (set
the gradient −2Xᵀ(**y** − X**β**) to zero) gives the **normal equations** (Lesson 01.05):

$$X^\top X\,\hat{\boldsymbol\beta} = X^\top \mathbf{y} \quad\Longrightarrow\quad \hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top \mathbf{y}.$$

Under the usual assumptions (Lesson 06.03), the standard errors are the square roots of the diagonal of

$$\widehat{\operatorname{Var}}(\hat{\boldsymbol\beta}) = \hat\sigma^2 (X^\top X)^{-1}, \qquad \hat\sigma^2 = \frac{\|\mathbf{y} - X\hat{\boldsymbol\beta}\|^2}{n - p - 1}.$$

In code use `np.linalg.solve(X.T @ X, X.T @ y)` or `np.linalg.lstsq`, never an explicit inverse.

## 2. Worked example 1: three predictors

exam_score on study_hours, sleep_hours and tutored (1 if yes):

| Term | Estimate | SE | 95% CI | True value |
| --- | --- | --- | --- | --- |
| intercept | 23.21 | 4.62 | [14.1, 32.3] | 26 |
| study_hours | **2.048** | 0.125 | [1.80, 2.30] | 2.2 |
| sleep_hours | **2.566** | 0.617 | [1.35, 3.78] | 2.0 |
| tutored | **6.764** | 1.338 | [4.13, 9.40] | 5.0 |

R² = 0.615, adjusted R² = 0.609, residual SE 8.65. Every true value lies inside its 95% interval.

**Interpretation, holding the others fixed:** "Among students with the same study and sleep hours, tutored students
scored about 6.8 points higher (95% CI 4.1 to 9.4)." Each coefficient is the expected change in y for a one-unit change
in *that* predictor, with all the other predictors held constant. It's a comparison *within* the data, not a claim
about what would happen if you intervened (that needs the causal reasoning of Module 09).

Notice that sleep is now clearly significant (p < 0.001), while in Lesson 06.01, exercise 4, alone it wasn't.
Including study hours removed noise from the residuals, so sleep's effect became visible. Adding predictors that
explain the outcome makes the other estimates **more precise**.

## 3. Omitted-variable bias

**Before you start.** The tutoring coefficient shrinks from **8.22** (tutoring alone) to **6.76** (adjusting for study
and sleep), because tutored students also studied slightly more, and part of the raw gap was really study time.

If a variable Z affects y **and** is correlated with a predictor X, leaving Z out biases X's coefficient: X takes credit
for Z's effect. Bias = (Z's effect on y) × (the slope of Z on X).

**Simulation with real confounding.** Motivation drives both study hours and tutoring. True tutoring effect = 3:

| Model | Tutoring coefficient |
| --- | --- |
| score ~ tutor | **10.14**, more than 3 times too big |
| score ~ tutor + hours | **2.95**, close to the truth |

Here adjusting for hours works because hours carry motivation's whole effect on the score. If motivation also affected
scores directly, you'd need to measure it. **Regression adjusts only for the variables you include.** Module 09
explains which ones to include (and which you must *not* include, such as colliders).

## 4. Categorical predictors: dummy variables

A categorical variable with k levels becomes k − 1 **dummy** (0/1) columns; the left-out level is the **baseline**.

**Worked example 4: program** (baseline Arts):

| Term | Estimate | Meaning |
| --- | --- | --- |
| intercept | 64.08 | mean score in Arts |
| Commerce | +0.94 | Commerce mean − Arts mean (65.02 − 64.08) |
| Science | −0.97 | Science mean − Arts mean (63.11 − 64.08) |

Changing the baseline to Science changes the coefficients (Arts +0.97, Commerce +1.91) but not the model's
predictions. The overall F-test, F = 0.29 with p = 0.75, is exactly the one-way ANOVA of Lesson 05.06. **ANOVA is
regression with dummy variables.**

Don't include all k dummies plus an intercept: the columns would then sum to the intercept column, XᵀX would be singular
(Lesson 01.05, exercise 7), and nothing could be estimated (the "dummy variable trap"). Formula interfaces such as
`C(program)` handle this automatically.

## 5. Interactions: when one effect depends on another

`exam_score ~ study_hours * tutored` fits

$$\text{score} = \beta_0 + \beta_1\,\text{study} + \beta_2\,\text{tutored} + \beta_3\,(\text{study} \times \text{tutored}).$$

| Group | Slope of study hours |
| --- | --- |
| not tutored | β₁ = **2.15** |
| tutored | β₁ + β₃ = 2.15 − 0.36 = **1.79** |

The interaction β₃ = −0.36 has p = 0.17: no evidence that the slopes differ. (In the generating script they're equal.)
With an interaction in the model, β₂ (10.57) is the tutoring gap at study = 0 hours, an extrapolation. Centre continuous
variables before forming interactions so that the main effects are interpretable at typical values.

**To claim "the effect differs between groups", test the interaction.** Comparing separate subgroup p-values is wrong
(Lesson 05.08, exercise 6).

## 6. Which predictor matters most? Standardised coefficients

Raw coefficients depend on units. Standardising every variable (z-scores) gives effects in "sd of y per sd of x":
study **0.74**, sleep **0.19**, commute **−0.003**. Study time is by far the strongest predictor *in this data*. That's
partly because it varies a lot between students, not only because its per-hour effect is large.

## 7. Comparing models: R², adjusted R², AIC and BIC

R² never decreases when you add a predictor, even pure noise. Better criteria penalise complexity:

| Criterion | Formula | Prefers |
| --- | --- | --- |
| adjusted R² | 1 − (1 − R²)(n − 1)/(n − p − 1) | higher |
| AIC | 2k − 2 log L̂ | lower |
| BIC | k log n − 2 log L̂ | lower (penalises more for n > 7) |

**Worked example 7.** Adding three columns of pure random noise:

| Model | R² | adjusted R² | AIC | BIC |
| --- | --- | --- | --- | --- |
| 3 real predictors | 0.6147 | 0.6088 | 1434.8 | **1448.0** |
| + 3 noise columns | 0.6346 | **0.6232** | **1430.2** | 1453.3 |

This is an honest and instructive outcome: by chance, the noise columns happened to fit some of this sample's
residuals, so adjusted R² and AIC both *preferred* the model with noise. Only BIC, with its heavier penalty, picked the
right model. **No in-sample criterion is safe on its own.** For prediction, compare models on data they haven't seen
(cross-validation, Module 08). For explanation, choose predictors from subject knowledge, not by searching.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Interpreting a coefficient without "holding others fixed" | Each coefficient is conditional on the others |
| Assuming adjustment removes all confounding | Only measured, included confounders are adjusted for |
| All k dummies plus an intercept | k − 1 dummies; the formula interface does it |
| Interpreting main effects with an interaction as overall effects | They're effects at 0 of the other variable; centre first |
| Choosing a model by R² | Adjusted R², AIC/BIC, and best of all cross-validation |

## Summary

- **y** = X**β** + **ε**; **β̂** = (XᵀX)⁻¹Xᵀ**y**; Var(**β̂**) = σ²(XᵀX)⁻¹.
- Coefficients are effects with the other predictors held fixed.
- Omitting a confounder biases the coefficients of correlated predictors.
- Categorical variables → k − 1 dummies (ANOVA = regression); interactions let slopes differ.
- R² always rises; use adjusted R², AIC/BIC or cross-validation.

## Self-test

1. What are the shapes of X, **β** and **y** for 200 observations and 4 predictors plus an intercept?
2. Why did the tutoring coefficient shrink when study hours were added?
3. In `score ~ C(program)`, what does the Commerce coefficient mean?
4. How would you test whether the effect of sleep is different for tutored students?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
