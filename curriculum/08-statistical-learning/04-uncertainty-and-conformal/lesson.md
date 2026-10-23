# 08.04 — Uncertainty and conformal prediction

**Time:** about 4 hours · **Prerequisites:** 05.03 (intervals), 08.02, 08.03

## Why this matters

A prediction without a measure of uncertainty is dangerous: "this patient's risk is 12%" and "this house is worth
$410,000" invite overconfidence. **Conformal prediction** wraps *any* model (a random forest, a gradient-boosted tree, a
deep network) and turns its predictions into intervals or sets that contain the true value with a **guaranteed**
probability, under one mild assumption, and with no distributional modelling. It's simple, fast, model-agnostic and
increasingly used in medicine, finance and LLM evaluation. This lesson also covers what conformal *can't* do: protect you
against distribution shift.

## Learning objectives

- Distinguish aleatoric and epistemic uncertainty.
- Implement split conformal prediction for regression and check its coverage.
- Explain the exchangeability assumption and the finite-sample guarantee.
- Use conformalised quantile regression (CQR) for intervals that adapt to difficulty.
- Build conformal prediction sets for classification.
- Recognise how distribution shift breaks the guarantee.

## Before you start

A random forest predicts disease progression. You want an interval that contains the true value for 90% of future
patients, without assuming the errors are normal or that the forest is any good. Is that possible?

---

## 1. Two kinds of uncertainty

| Kind | Source | Reducible with more data? | Example |
| --- | --- | --- | --- |
| **aleatoric** | inherent noise in the outcome | no | two identical-looking patients progress differently |
| **epistemic** | not knowing the right model or parameters | yes | few training examples near this input |

Bayesian methods (Module 07) and ensembles estimate epistemic uncertainty. Prediction intervals must cover both.
Conformal prediction delivers that coverage without having to model either source explicitly.

## 2. Split conformal prediction

1. Split the data into a **training** set and a **calibration** set.
2. Fit any model on the training set.
3. On the calibration set, compute **nonconformity scores**, which measure how wrong the model is. For regression:
   sᵢ = |yᵢ − ŷᵢ|.
4. Take the quantile q̂ = the ⌈(n + 1)(1 − α)⌉-th smallest score (a slightly inflated empirical (1 − α) quantile).
5. For a new x, the interval is **ŷ(x) ± q̂**.

**The guarantee:** if the calibration and test points are **exchangeable** (for example i.i.d.), then

$$P\big(y_{\text{new}} \in [\hat{y}(x_{\text{new}}) - \hat{q},\; \hat{y}(x_{\text{new}}) + \hat{q}]\big) \ge 1 - \alpha,$$

for **any** model and **any** distribution, at any sample size. *Why:* the new point's score is equally likely to rank
anywhere among the n + 1 scores, so the probability that it exceeds the ⌈(n + 1)(1 − α)⌉-th smallest is at most α.

**Worked example 1 (Before you start).** A random forest on the diabetes data, with 185 training, 124 calibration and 133
test points, α = 0.1: q̂ = 101.8, so every interval is prediction ± 101.8. Test coverage **0.932** (target 0.90). Answer:
**yes, it's possible.**

**Worked example 2.** A deliberately weak model (2 features, depth 2): coverage **0.940** with width 262 instead of 204.
**A bad model still gives valid intervals, just wider ones.** Model quality buys you narrower intervals, not validity.

**Worked example 3.** Over 200 random splits, coverage averages **0.912**, with 90% of splits between 0.857 and 0.962. The
guarantee is *marginal*: it holds on average over calibration sets and test points. Any single test set fluctuates,
roughly binomially.

## 3. Intervals that adapt: conformalised quantile regression

The constant width ± q̂ is the same everywhere, too wide for easy cases and too narrow for hard ones. **CQR** starts from
quantile-regression models for the lower and upper quantiles (α/2 and 1 − α/2), then conformally adjusts them:

- score sᵢ = max(lo(xᵢ) − yᵢ, yᵢ − hi(xᵢ)), i.e. how far yᵢ falls outside the predicted band;
- interval [lo(x) − q̂, hi(x) + q̂].

**Worked example 4.** Noise sd grows from 0.2 at x = 0 to 3.2 at x = 10:

| Method | Overall coverage | Coverage at x < 2 | Coverage at x > 8 | Width at x < 2 | Width at x > 8 |
| --- | --- | --- | --- | --- | --- |
| constant-width conformal | 0.904 | 1.000 | **0.740** | 6.82 | 6.82 |
| **CQR** | 0.904 | 0.939 | 0.873 | **1.83** | **9.18** |

Both meet the 90% target overall, but the constant-width method over-covers easy cases and badly under-covers hard ones.
CQR's intervals widen where the data are noisy (`conformal.png`), so its coverage is much more even. The guarantee is
still only *marginal*; no method can guarantee 90% within every subgroup without further assumptions or more data.

## 4. Prediction sets for classification

For a classifier with probabilities p̂(k | x):

- score sᵢ = 1 − p̂(true class of i | xᵢ);
- the prediction set is every class k with 1 − p̂(k | x) ≤ q̂, i.e. p̂(k | x) ≥ 1 − q̂.

**Worked example 5:** handwritten digits (10 classes), logistic regression (top-1 accuracy 0.951).

| Target | Coverage | Average set size | Sets of size 1 | Sets of size ≥ 3 |
| --- | --- | --- | --- | --- |
| 90% | 0.905 | 0.92 | 92.4% | 0% |
| 98% | 0.985 | 1.16 | 85.5% | 1.1% |

Set size is a direct, honest measure of uncertainty: easy digits get one label, ambiguous ones get two or three. At 90%,
*below* the model's 95% accuracy, some sets are **empty**: the method is allowed to say "none of these" for the least
confident cases, which is valid but unhelpful. Usually you'd choose a target above the accuracy. Adaptive scores (APS,
RAPS) give better-behaved sets.

## 5. What conformal can't do: distribution shift

The guarantee needs exchangeability: test data must look like calibration data. **Worked example 6:** the CQR model was
fitted and calibrated on x ∈ [0, 10]. On new inputs from [12, 18] (beyond the data, with noisier outcomes), coverage
falls from **0.904 to 0.724**.

Shift happens constantly in deployment: new hospitals, new user populations, changing prices, a new camera. Defences:
monitor coverage on fresh labelled data; recalibrate regularly; use weighted conformal methods when the shift is known
(covariate shift); use online conformal methods that adapt over time. Treat any uncertainty estimate as conditional on
"the future resembles the calibration data".

## 6. Other uncertainty tools

| Method | What it gives | Guarantee? |
| --- | --- | --- |
| conformal prediction | intervals or sets with coverage ≥ 1 − α | yes, finite-sample, marginal, under exchangeability |
| Bayesian posterior predictive (Module 07) | full predictive distributions | only if the model and prior are right |
| deep ensembles | spread across independently trained models | none, but often well calibrated in practice |
| MC dropout | approximate Bayesian spread | none; often underestimates |
| quantile regression alone | conditional quantiles | none in finite samples (CQR adds one) |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Calibrating on the training data | Use a separate calibration set (or cross-conformal) |
| Using the plain empirical quantile | Use the ⌈(n + 1)(1 − α)⌉ order statistic |
| Expecting 90% coverage in every subgroup | The guarantee is marginal |
| Trusting coverage after deployment drift | Monitor and recalibrate |
| Treating a narrow interval as proof of a good model | Width reflects model quality; validity is separate |

## Summary

- Uncertainty = aleatoric (noise) + epistemic (model); intervals must cover both.
- Split conformal: calibration scores → quantile q̂ → ŷ ± q̂; coverage ≥ 1 − α for any model under exchangeability.
- CQR adapts the width to local difficulty; classification gives prediction sets whose size signals uncertainty.
- Distribution shift breaks the guarantee: monitor and recalibrate.

## Self-test

1. What single assumption does split conformal prediction need?
2. Why does a bad model still produce valid conformal intervals?
3. What does an average prediction-set size of 1.4 tell you?
4. Your model goes live in a new country. Should you trust last year's conformal intervals?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
