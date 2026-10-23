# 06.02 Exercises — Multiple regression

`make lesson L=06-regression/02-multiple-regression-matrix-form`

## ★ Warm-up

1. **(M)** For 200 observations, 4 predictors and an intercept, give the shapes of X, **β**, **y**, XᵀX and Xᵀ**y**.
2. **(C)** Interpret the sleep coefficient 2.57 in the lesson's three-predictor model in one careful sentence.
3. **(C)** What goes wrong if you include dummies for all three programs *and* an intercept?

## ★★ Core

4. **(P)** Fit `exam_score ~ study_hours + sleep_hours + tutored + C(program) + year + commute_minutes`. Which predictors
   have p < 0.05? Compare with how `make_students.py` generated the data. Are there any false positives or false
   negatives?
5. **(P)** Predict the score of a 2nd-year Science student who studies 12 hours, sleeps 8, is tutored and commutes 20
   minutes, using the model from exercise 4. Give the 95% prediction interval.
6. **(P)** Test whether the effect of sleep differs for tutored students (an interaction model). Report and interpret.
7. **(P)** Compare the SE of the study-hours coefficient with and without sleep_hours in the model. Why does adding a
   predictor that is *uncorrelated* with study hours make the study estimate more precise?
8. **(P)** Add `ols_standard_errors(X, y)` to `statkit/regression.py`: return √diag(σ̂²(XᵀX)⁻¹) with
   σ̂² = SSE/(n − p − 1), where p + 1 is the number of columns of X. Check against statsmodels.
9. **(M)** **Omitted-variable bias formula.** True model y = β₀ + β₁x + β₂z + ε. You fit y on x only. Show that the
   expected slope is β₁ + β₂ δ, where δ is the slope from regressing z on x. Check the formula with the lesson's
   motivation simulation.

## ★★★ Challenge

10. **(P)** **Frisch–Waugh–Lovell.** Regress exam_score on sleep and tutored and keep the residuals; regress study_hours
    on sleep and tutored and keep those residuals. Regress the first residuals on the second. Show the slope equals the
    study coefficient in the full model. Explain in words what "holding the others constant" means in light of this.
11. **(M)** Show that the residual vector **e** = **y** − X**β̂** is orthogonal to every column of X (Xᵀ**e** = **0**).
    What does this mean geometrically?
12. **(P)** Simulate 1,000 datasets of n = 100 with 10 candidate predictors, none related to y. In each, keep the
    predictor with the smallest p-value and refit with only it. What share of these "best" single-predictor models has
    p < 0.05? Connect to Lesson 05.08.

Solutions: [solutions/solutions.md](solutions/solutions.md)
