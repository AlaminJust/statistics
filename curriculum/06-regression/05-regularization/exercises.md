# 06.05 Exercises — Regularisation

`make lesson L=06-regression/05-regularization`

## ★ Warm-up

1. **(C)** Why is the training error always smallest at λ = 0? Why does that make CV necessary?
2. **(M)** Sketch, in two dimensions, the lasso constraint |β₁| + |β₂| ≤ t and the ridge constraint β₁² + β₂² ≤ t, with
   elliptical error contours. Mark where they first touch, and explain the zeros.
3. **(C)** n = 1,000 and p = 5. Would you expect regularisation to improve test error much? Why?

## ★★ Core

4. **(M)** For a single standardised predictor with no intercept, show that the ridge estimate is
   β̂ = Σxᵢyᵢ/(Σxᵢ² + λ), and so is the OLS estimate shrunk by the factor n/(n + λ) (since Σxᵢ² = n).
5. **(P)** Implement `fit_ridge(X, y, lam)` in `statkit/regression.py` using the closed form, penalising every column
   **except the first** (the intercept). Check against `sklearn.linear_model.Ridge` on the standardised diabetes data.
6. **(P)** Using `LassoCV` on the standardised diabetes data, find the CV-optimal α and the α chosen by the
   one-standard-error rule. How many features does each keep?
7. **(P)** **Stability of selection.** Bootstrap the diabetes data 200 times and fit a lasso with α = 5 each time. For
   each feature, report the share of bootstraps in which it's selected. Which selections are stable?
8. **(P)** Regularised logistic regression: fit `LogisticRegression(penalty="l1", solver="liblinear", C=...)` on the
   breast-cancer data for C = 0.01, 0.1, 1 and 10. Plot the number of non-zero coefficients and the 5-fold CV accuracy
   against C.
9. **(P)** **When ridge wins.** Repeat worked example 4 with a *dense* truth: all 200 coefficients drawn from N(0, 0.3²).
   Compare the test RMSE of ridge, lasso and predicting the mean. Explain the result.

## ★★★ Challenge

10. **(M)** Show that the ridge estimate is the posterior mode under the model y ~ N(X**β**, σ²I) with prior
    β ~ N(**0**, τ²I), and find λ in terms of σ² and τ².
11. **(P)** Implement the lasso for standardised X by **coordinate descent**: cycle through j, updating
    βⱼ ← S(xⱼᵀ(**y** − X₋ⱼ**β**₋ⱼ)/n, α), where S(z, α) = sign(z) max(|z| − α, 0) is the soft-threshold function. Compare
    with scikit-learn's `Lasso` (which minimises (1/2n)‖**y** − X**β**‖² + α‖**β**‖₁).
12. **(C)** A paper reports "lasso selected 7 genes, all with p < 0.001 in the refitted OLS model". Explain why those
    p-values are misleading, and suggest a valid alternative.

Solutions: [solutions/solutions.md](solutions/solutions.md)
