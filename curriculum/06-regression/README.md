# Module 06 — Regression and GLMs

**Weeks 16–19 · about 32 hours** · Status: **outline**

Regression is the bridge between classical statistics and machine learning. A neural network's last layer is a
generalised linear model, weight decay is ridge regression, and logistic regression is the baseline every classifier
must beat. This module builds regression from the normal equations you met in Lesson 01.05 up to regularised and mixed
models.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 06.01 | Simple linear regression | Fit and interpret a line | least squares, slope = r · s_y/s_x, residuals, R², prediction vs explanation, regression to the mean |
| 06.02 | Multiple regression in matrix form | Fit models with many predictors | design matrix, normal equations, the hat matrix, categorical predictors (dummy variables), interactions, interpreting "holding others constant" |
| 06.03 | Diagnostics and assumptions | Check whether a model can be trusted | the Gauss–Markov assumptions, residual plots, Q–Q plots, heteroscedasticity, leverage and influence, multicollinearity and VIF, robust standard errors, inference for coefficients |
| 06.04 | Logistic regression and GLMs | Model binary and count outcomes | the logit link, odds ratios, fitting by gradient descent and by IRLS, Poisson regression, deviance, overdispersion |
| 06.05 | Regularisation | Control overfitting and select features | ridge, lasso, elastic net, coefficient paths, choosing λ by cross-validation, the Bayesian view (priors) |
| 06.06 | Splines and mixed models | Handle curves and grouped data | polynomial and spline bases, GAMs, random intercepts and slopes, partial pooling |

## statkit functions (`statkit/regression.py`)

`fit_ols(X, y)`, `predict(X, beta)`, `r_squared`, `residuals`, `standard_errors(X, y, beta)`, `vif(X)`,
`fit_logistic_gd(X, y, lr, steps)`, `fit_ridge(X, y, lam)`.

## Mini-project: regression modelling report

Model `exam_score` in `students.csv`, whose true generating equation you know. Then model disease progression in the
scikit-learn diabetes dataset (continuing the m02 EDA). Report coefficients with intervals, diagnostics, a comparison of
OLS, ridge and lasso by cross-validated error, and an honest section on what the coefficients do and don't mean.

## Checkpoint (without notes)

1. Derive the normal equations by setting the gradient of the squared error to zero.
2. Interpret a coefficient of 2.1 on `study_hours` in a model that also contains `sleep_hours`.
3. Name two assumptions that matter for unbiased coefficients and two that matter only for valid standard errors.
4. Explain why lasso can set coefficients exactly to zero and ridge cannot.
5. `make test M=regression` passes.

## Resources

[An Introduction to Statistical Learning (Python edition)](https://www.statlearning.com/) chapters 3, 4, 6, 7 ·
[Regression and Other Stories](https://avehtari.github.io/ROS-Examples/) ·
[statsmodels documentation](https://www.statsmodels.org/)
