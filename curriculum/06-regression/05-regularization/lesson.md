# 06.05 — Regularisation: ridge, lasso and elastic net

**Time:** about 5 hours · **Prerequisites:** 05.01 (shrinkage), 06.02, 06.03

## Why this matters

With many predictors, or highly correlated ones, least squares overfits: it chases noise, its coefficients become
unstable, and its predictions on new data suffer. **Regularisation** adds a penalty on coefficient size, trading a little
bias for a large drop in variance (the shrinkage idea of Lesson 05.01). Ridge regression is "weight decay" in neural
networks. The lasso does automatic feature selection. Choosing the penalty strength by cross-validation is the template
for all hyperparameter tuning in ML.

## Learning objectives

- Write down the ridge, lasso and elastic-net objectives, and the ridge closed form.
- Explain geometrically why the lasso sets coefficients exactly to zero and ridge doesn't.
- Read coefficient paths; choose λ by cross-validation.
- Explain why features must be standardised first.
- Know when regularisation helps (p ≈ n or p > n, collinearity) and when it doesn't.
- Interpret ridge and lasso as Bayesian MAP estimates.

## Before you start

You have 60 observations and 200 candidate predictors, of which only 5 truly matter. Can ordinary least squares even be
fitted? What would happen to its predictions on new data?

---

## 1. The penalised objectives

With standardised predictors (mean 0, sd 1) and the intercept left unpenalised:

| Method | Minimise | Penalty |
| --- | --- | --- |
| OLS | ‖**y** − X**β**‖² | none |
| **Ridge** (L2) | ‖**y** − X**β**‖² + λ Σβⱼ² | squared size |
| **Lasso** (L1) | (1/2n)‖**y** − X**β**‖² + α Σ\|βⱼ\| | absolute size |
| **Elastic net** | mix of both | α[ρ Σ\|βⱼ\| + (1 − ρ)/2 Σβⱼ²] |

λ (or α) controls the strength: 0 gives OLS, and ∞ shrinks every coefficient to 0. (scikit-learn calls it `alpha` for
all three, with the scalings shown; values aren't comparable across methods.)

**Ridge has a closed form:**

$$\hat{\boldsymbol\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}.$$

Adding λI makes the matrix invertible even when XᵀX isn't (collinearity, p > n). That's why ridge "fixes"
multicollinearity (Lesson 06.03).

**Worked example 1: diabetes data, standardised.**

| λ | ‖**β**‖ | s1 coefficient |
| --- | --- | --- |
| 0 (OLS) | 65.5 | −37.7 |
| 10 | 42.6 | −11.3 |
| 100 | 34.8 | −2.1 |
| 1000 | 17.4 | +1.1 |

The closed form matches scikit-learn exactly. The collinear s1 (VIF 59) has a wild OLS coefficient that ridge reins in
quickly.

## 2. Why the lasso selects variables

The lasso can be written as: minimise the squared error subject to Σ|βⱼ| ≤ t. That constraint region is a **diamond**
with corners on the axes. Ridge's region, Σβⱼ² ≤ t, is a **circle**. The elliptical contours of the squared error
usually first touch the diamond at a **corner**, where some coefficients are exactly 0. A circle has no corners, so ridge
shrinks everything but rarely zeroes anything.

`coefficient_paths.png` shows this: ridge coefficients glide smoothly towards 0, while lasso coefficients drop out one by
one. **Worked example 2: lasso on diabetes.**

| α | Non-zero coefficients |
| --- | --- |
| 0.1 | 9 |
| 1 | 7 |
| 5 | 5: sex, bmi, bp, s3, s5 |
| 20 | **3: bmi, bp, s5** |

The last survivors are the strongest predictors, matching the correlations from the m02 EDA project.

**Caution:** with correlated predictors the lasso picks one more or less arbitrarily and drops the others. Its selection
is unstable, and it's not evidence that the dropped variables "don't matter". The **elastic net** keeps groups of
correlated predictors together.

## 3. Choosing λ by cross-validation

λ is a **hyperparameter**: it can't be estimated by least squares, because the training error is always smallest at
λ = 0. Instead:

1. Split the training data into k folds.
2. For each λ on a grid, fit on k − 1 folds and measure the error on the held-out fold; average over folds.
3. Choose the λ with the lowest CV error (or the "one-standard-error rule": the largest λ within 1 SE of the minimum, for
   a simpler model).
4. Refit on all training data, and evaluate **once** on the untouched test set.

`RidgeCV`, `LassoCV` and `ElasticNetCV` do steps 1–3. Wrap them in a pipeline with the scaler so that standardisation is
learned inside each fold; otherwise information leaks from the validation fold (Module 08).

**Worked example 3: diabetes test-set RMSE** (30% held out):

| Model | Test RMSE | Notes |
| --- | --- | --- |
| OLS | 55.65 | |
| ridge (CV) | 55.45 | λ = 23.1 |
| lasso (CV) | 55.84 | α = 0.86; 8 of 10 features kept |
| elastic net (CV) | 55.45 | |

**Honest result: they're all about the same.** With n = 442 and only 10 predictors, OLS isn't overfitting much, so
there's little variance to remove. Regularisation pays off when p is large relative to n.

## 4. Worked example 4: p > n (Before you start)

n = 60 observations, p = 200 predictors, 5 truly non-zero coefficients.

| Model | Training RMSE | Test RMSE (1,000 new points) |
| --- | --- | --- |
| OLS (minimum-norm least squares) | **0.000**: a perfect fit | 3.57 |
| ridge (CV) | | 3.66 |
| **lasso (CV)** | | **1.32** |

With p > n, XᵀX can't be inverted and OLS has infinitely many perfect solutions; any of them interpolates noise and
predicts badly. The lasso keeps 24 features, including **all 5 true ones**, and cuts the test error by almost two
thirds. Ridge doesn't help here because the truth is **sparse** (a few big effects). Ridge works better when there are
**many small effects** (exercise 9). Rule of thumb: no single penalty is always best, so let cross-validation choose,
or use the elastic net.

## 5. Standardise first

The penalty treats all coefficients alike, but a coefficient's size depends on its feature's units. Without
standardising, features measured in big units (cholesterol in mg/dL) have small coefficients and are barely penalised,
while features in small units (sex coded 1/2) are penalised heavily. **Worked example 5:** ridge on raw diabetes
features gives the sex coefficient −10.6; on standardised features (converted back to raw units) it's −16.9. Always
standardise inside a pipeline, and leave the intercept unpenalised.

## 6. The Bayesian view

Ridge is the **MAP estimate** (posterior mode) under independent normal priors βⱼ ~ N(0, τ²), with λ = σ²/τ². The lasso
is the MAP estimate under Laplace (double-exponential) priors, whose sharp peak at 0 produces exact zeros. Regularisation
is prior knowledge that "most effects are small". Module 07 makes this precise, and adds the uncertainty around the
estimates, which penalised regression doesn't give you.

**Inference after regularisation is hard.** Standard errors and p-values from a lasso-selected model are not valid
(selection used the same data). Use regularisation for prediction. For inference, use pre-specified models, sample
splitting, or specialised post-selection methods.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Regularising unstandardised features | Standardise in a pipeline |
| Choosing λ by training error | Cross-validation |
| Scaling on the full data before CV | Fit the scaler inside each fold (pipeline) |
| "The lasso dropped X, so X is irrelevant" | Selection among correlated predictors is unstable |
| Reporting p-values after lasso selection | Not valid; use selection-aware methods |

## Summary

- Ridge: + λΣβ²; closed form (XᵀX + λI)⁻¹Xᵀy; smooth shrinkage; handles collinearity.
- Lasso: + αΣ|β|; exact zeros (feature selection); best for sparse truths.
- Elastic net: both; keeps correlated groups together.
- Choose λ by CV inside a pipeline; evaluate once on a test set.
- They help most when p is large relative to n; with n ≫ p, OLS is often just as good.
- Ridge = normal prior MAP; lasso = Laplace prior MAP.

## Self-test

1. Why is the training error smallest at λ = 0?
2. Draw the lasso and ridge constraint regions in 2-D, and explain why the lasso gives zeros.
3. Your data have n = 1,000 and p = 5. Should you expect regularisation to help much?
4. What prior corresponds to ridge regression?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
