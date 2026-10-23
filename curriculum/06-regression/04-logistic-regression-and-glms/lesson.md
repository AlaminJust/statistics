# 06.04 — Logistic regression and generalised linear models

**Time:** about 5 hours · **Prerequisites:** 01.02, 04.01, 05.02, 06.02

## Why this matters

Many outcomes aren't continuous: pass/fail, click/no click, disease/healthy (binary), or counts of errors, purchases
and accidents. **Generalised linear models** (GLMs) extend regression to them by modelling a transformed mean as a
linear function of the predictors. **Logistic regression** is the GLM for binary outcomes: it's the standard baseline
classifier, the output layer of every binary neural network, and the main tool of medical and social research.
**Poisson regression** is the GLM for counts.

## Learning objectives

- Explain why linear regression fails for binary outcomes.
- Fit and interpret logistic regression: log-odds, odds ratios and predicted probabilities.
- Fit logistic regression from scratch by IRLS (Newton's method).
- Use deviance, likelihood-ratio tests and pseudo-R².
- Fit Poisson regression with an exposure offset, and detect and handle overdispersion.

## Before you start

The odds ratio for tutoring turns out to be 3.6. Does that mean tutored students are 3.6 times as *likely* to score
at least 70?

---

## 1. Why not a straight line?

A linear model for a 0/1 outcome (the "linear probability model") predicts P(score ≥ 70) = **−0.26** at 0 study
hours and **1.46** at 30 hours: impossible probabilities. Errors can't be normal either, since y is only ever 0 or 1.

## 2. The GLM recipe

A GLM has three parts:

| Part | Logistic regression | Poisson regression | Linear regression |
| --- | --- | --- | --- |
| distribution of y | Bernoulli(p) | Poisson(μ) | Normal(μ, σ²) |
| link g(mean) | **logit**: log(p/(1 − p)) | **log**: log μ | identity |
| linear predictor | β₀ + β₁x₁ + … | same | same |

So logistic regression says log-odds(yᵢ = 1) = **x**ᵢᵀ**β**, i.e. pᵢ = σ(**x**ᵢᵀ**β**) with the sigmoid of Lesson 01.02.
Fitting is by maximum likelihood (Lesson 05.02); the NLL is the binary cross-entropy (Lesson 04.06).

## 3. Interpreting logistic regression

**Worked example 2:** P(score ≥ 70) on study_hours and tutored.

| Term | Coefficient (log-odds) | SE | Odds ratio exp(β) | 95% CI for the OR |
| --- | --- | --- | --- | --- |
| intercept | −5.63 | 0.78 | | |
| study_hours | **0.410** | 0.062 | **1.51** | 1.34 to 1.70 |
| tutored | **1.285** | 0.436 | **3.61** | 1.54 to 8.50 |

- **Coefficient:** each extra study hour adds 0.41 to the *log-odds* of a high score, holding tutoring fixed.
- **Odds ratio:** each extra hour multiplies the *odds* by 1.51; tutoring multiplies them by 3.6.
- **Probabilities** depend on where you start, because the curve is S-shaped (`logistic_curves.png`):

| Study hours | P(high), no tutoring | P(high), tutoring | Difference |
| --- | --- | --- | --- |
| 5 | 0.027 | 0.091 | +0.06 |
| 10 | 0.178 | 0.439 | **+0.26** |
| 15 | 0.626 | 0.858 | +0.23 |
| 20 | 0.929 | 0.979 | +0.05 |

**Answer to "Before you start": no.** The odds ratio is constant at 3.6, but the *probability* gain from tutoring
ranges from 5 to 26 points depending on study hours. Odds ≈ probability only when the probability is small. Report
odds ratios for the model, and probability differences at meaningful values for the reader.

## 4. Fitting by IRLS (Newton's method)

There's no closed form, but the log-likelihood is concave, so Newton's method converges quickly:

$$\boldsymbol\beta \leftarrow \boldsymbol\beta + (X^\top W X)^{-1} X^\top(\mathbf{y} - \mathbf{p}), \qquad W = \operatorname{diag}\big(p_i(1 - p_i)\big).$$

Each step is a weighted least-squares fit, hence *iteratively reweighted least squares* (IRLS). The gradient
Xᵀ(**y** − **p**) has the same form as in linear regression. And (XᵀWX)⁻¹ at convergence is the inverse Fisher
information, so its diagonal gives the squared SEs (Lesson 05.02).

**Worked example 3.** From **β** = 0, IRLS converges in **7 iterations** to (−5.6291, 0.4097, 1.2849) with SEs
(0.775, 0.062, 0.436), identical to statsmodels. Gradient descent (Lesson 01.03) also works, but needs hundreds of steps.

## 5. How good is the fit?

- **Deviance** = −2 × log-likelihood (relative to a perfect model). It plays the role of SSE.
- **Likelihood-ratio test:** 2(ℓ_full − ℓ_reduced) ~ χ² with df = the number of extra parameters. Null model −126.12,
  full model −79.72: LR = **92.8** on 2 df, p ≈ 10⁻²¹.
- **McFadden pseudo-R²** = 1 − ℓ_model/ℓ_null = **0.37**. It's not comparable to linear R²; values of 0.2–0.4 indicate a
  good fit.
- **Classification accuracy** at a 0.5 threshold is 0.825, against 0.675 for always predicting "not high". The
  threshold is a *decision* separate from the model, and Module 08 covers better metrics (ROC, calibration).

**Separation warning:** if a predictor perfectly separates the classes, the MLE doesn't exist (coefficients run off to
infinity). statsmodels warns you. Use regularisation (Lesson 06.05).

## 6. Poisson regression for counts

log E[yᵢ] = **x**ᵢᵀ**β**, so exp(β) is a **rate ratio**.

**Offsets.** When counts are collected over different exposures (more traffic, longer time, bigger areas), model the
*rate*: log E[yᵢ] = log(exposureᵢ) + **x**ᵢᵀ**β**. The log(exposure) term has its coefficient fixed at 1 (an offset).

**Worked example 5.** Daily website errors over 300 days with traffic of 1,000–20,000 visits; some days use a new
release:

| Term | Coefficient | SE | Rate ratio |
| --- | --- | --- | --- |
| intercept | −1.010 | 0.037 | baseline e^(−1.01) ≈ 0.36 errors per thousand visits |
| new_release | **0.352** | 0.053 | **1.42**: the new release has 42% more errors per visit |

The true rate ratio is e^0.4 = 1.49, inside the 95% interval (exp(0.248) to exp(0.457) = 1.28 to 1.58). The Pearson
χ²/df of 0.93 is close to 1: no overdispersion.

## 7. Overdispersion

Poisson assumes variance = mean (Lesson 04.01). Real counts usually vary more, because units differ in ways the model
doesn't capture.

**Worked example 6.** Same mean structure, but extra variation:

| Model | Coefficient | SE | Dispersion (Pearson χ²/df) |
| --- | --- | --- | --- |
| Poisson | 0.438 | **0.052** | **3.24** |
| negative binomial | 0.421 | **0.101** | — |

Both give similar estimates, but the Poisson SE is about half what it should be, so its intervals are far too narrow
and its p-values far too small. **Always check the dispersion.** If it's well above 1, use the negative binomial (or a
quasi-Poisson or robust SEs).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Reading an odds ratio as a risk ratio | They're close only when the outcome is rare |
| Comparing logistic coefficients across models like linear ones | Log-odds scale; omitted variables change them even without confounding |
| Ignoring exposure in count data | Use an offset log(exposure) |
| Poisson with dispersion ≫ 1 | Negative binomial or robust SEs |
| Judging a classifier by accuracy alone | Module 08: ROC, PR, calibration |

## Summary

- GLM = distribution + link + linear predictor. Logistic: Bernoulli + logit; Poisson: Poisson + log.
- Logistic coefficients are log-odds changes; exp(β) are odds ratios; probabilities depend on the baseline.
- Fit by MLE using IRLS (Newton); SEs from (XᵀWX)⁻¹.
- Deviance and likelihood-ratio tests replace SSE and F; McFadden pseudo-R².
- Poisson exp(β) = rate ratios; offsets for exposure; check overdispersion and use the negative binomial if needed.

## Self-test

1. A coefficient of −0.7 in a logistic regression corresponds to which odds ratio? Interpret it.
2. Why do the same odds ratios give different probability differences at different baselines?
3. What does an offset do in Poisson regression?
4. How do you detect overdispersion, and why does it matter?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
