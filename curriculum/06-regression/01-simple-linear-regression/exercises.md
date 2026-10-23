# 06.01 Exercises — Simple linear regression

`make lesson L=06-regression/01-simple-linear-regression`

## ★ Warm-up

1. **(M)** x̄ = 10, ȳ = 50, r = 0.6, s_x = 2, s_y = 8. Write the regression line of y on x, and predict y at x = 13.
2. **(M)** By hand, fit the least-squares line to (1, 2), (2, 4), (3, 5), (4, 4), (5, 5). Give the residuals and R².
3. **(C)** Why do least-squares residuals always sum to zero? (Hint: the derivative with respect to b₀.)

## ★★ Core

4. **(P)** Regress `exam_score` on `sleep_hours`. Report the slope with its 95% CI, and R². Interpret both in one
   sentence each. Is sleep "unimportant"? (Compare with how the data were generated.)
5. **(P)** Convert study hours to minutes and refit. What happens to the slope, intercept, R² and t-statistic? Why?
6. **(P)** Standardise both `study_hours` and `exam_score` (z-scores) and refit. Show that the slope equals r and the
   intercept is 0.
7. **(P)** Give the 95% confidence and prediction intervals for students studying 5, 10 and 25 hours. Which predictions
   are least reliable, and why?
8. **(P)** Implement `fit_ols(X, y)` (normal equations with `np.linalg.solve`; X includes a column of 1s), `predict(X, beta)`,
   `residuals(X, y, beta)` and `r_squared(X, y, beta)` in `statkit/regression.py`. NumPy is allowed in this module.
9. **(P)** Simulate regression to the mean: 100,000 "patients" with true blood pressure ~ N(140, 15²) and two measurements
   each, with measurement noise sd 10. Select those with a first reading above 160. What is their average second reading,
   with no treatment at all? Predict it from the regression formula ρ = Var(true)/Var(measured).

## ★★★ Challenge

10. **(M)** Derive b₀ and b₁ by setting ∂SSE/∂b₀ = 0 and ∂SSE/∂b₁ = 0.
11. **(M)** Show that in simple regression R² = r². (Hint: SST − SSE = b₁² Sxx.)
12. **(P)** **Anscombe again.** Fit a regression line to each of Anscombe's four datasets (Lesson 02.06) with statsmodels.
    Show that the coefficients, SEs and R² are all nearly identical, then plot the residuals of each. Which datasets
    violate the model's assumptions, and how?

Solutions: [solutions/solutions.md](solutions/solutions.md)
