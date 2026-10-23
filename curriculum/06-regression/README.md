# Module 06 — Regression and GLMs

**Weeks 16–19 · about 32 hours**

Regression is the bridge between classical statistics and machine learning. A neural network's last layer is a
generalised linear model, weight decay is ridge regression, and logistic regression is the baseline every classifier
must beat. This module builds regression from the normal equations of Lesson 01.05 up to regularised, non-linear and
multilevel models.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 06.01 | [Simple linear regression](01-simple-linear-regression/lesson.md) | b₁ = r s_y/s_x; prediction vs confidence intervals; regression to the mean | `fit_ols`, `predict`, `residuals`, `r_squared` |
| 06.02 | [Multiple regression in matrix form](02-multiple-regression-matrix-form/lesson.md) | β̂ = (XᵀX)⁻¹Xᵀy; "holding others fixed"; omitted-variable bias | `ols_standard_errors` |
| 06.03 | [Diagnostics and assumptions](03-diagnostics-and-assumptions/lesson.md) | Residual plots, robust SEs, influence, VIF | `vif` |
| 06.04 | [Logistic regression and GLMs](04-logistic-regression-and-glms/lesson.md) | Odds ratios, IRLS, Poisson with offsets, overdispersion | `fit_logistic_gd` |
| 06.05 | [Regularisation](05-regularization/lesson.md) | Ridge shrinks, lasso selects; choose λ by CV | `fit_ridge` |
| 06.06 | [Splines and mixed models](06-splines-and-mixed-models/lesson.md) | Local curves; partial pooling of groups | — |

Run `make test M=regression` as you implement each function.

## Mini-project: a regression modelling report

Brief in [projects/m06-regression-report/README.md](../../projects/m06-regression-report/README.md).

## Checkpoint (without notes)

1. Derive the normal equations by setting the gradient of the squared error to zero.
2. Interpret a coefficient of 2.1 on `study_hours` in a model that also contains `sleep_hours`.
3. Name two assumptions that matter for unbiased coefficients and two that matter only for valid standard errors.
4. Explain an odds ratio of 3.6, and why it isn't "3.6 times as likely".
5. Explain why lasso can set coefficients exactly to zero and ridge cannot.
6. Explain partial pooling with the formula for wⱼ.
7. `make test M=regression` passes.

Pass all seven → `git tag -a m06-done -m "Module 06 complete"`.

## Resources

[An Introduction to Statistical Learning (Python edition)](https://www.statlearning.com/) chapters 3, 4, 6, 7 ·
[Regression and Other Stories](https://avehtari.github.io/ROS-Examples/) ·
[statsmodels documentation](https://www.statsmodels.org/)
