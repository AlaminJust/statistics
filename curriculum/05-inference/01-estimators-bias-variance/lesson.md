# 05.01 — Estimators: bias, variance and mean squared error

**Time:** about 4 hours · **Prerequisites:** Modules 03–04

## Why this matters

An **estimator** is a recipe that turns data into a guess about an unknown quantity: x̄ for μ, s² for σ², a trained
model's weights for the "true" weights. How good is a recipe? This lesson gives the three criteria every statistician
and ML researcher uses: **bias** (is it right on average?), **variance** (how much does it jump around?) and **mean
squared error**, which combines the two. The decomposition **MSE = bias² + variance** is the bias–variance trade-off
that shapes regularisation, model selection and deep learning (Module 08).

## Learning objectives

- Define an estimator, its sampling distribution, bias, variance, standard error and MSE.
- Prove and use MSE = bias² + variance.
- Compare estimators by simulation.
- Explain why a biased estimator can be better (shrinkage).
- Define consistency and recognise consistent estimators.

## Before you start

To estimate the centre of a Uniform(0, 10) distribution from 10 observations, which do you expect to be most accurate:
the sample mean, the sample median, or the midpoint of the smallest and largest values?

---

## 1. Estimators and estimates

| Term | Meaning | Example |
| --- | --- | --- |
| parameter θ | the unknown truth | the population mean μ |
| estimator θ̂ | a *rule*, a function of the random sample | X̄ = (X₁ + … + Xₙ)/n |
| estimate | the number the rule gives on *your* data | x̄ = 63.93 |
| sampling distribution | the distribution of θ̂ over repeated samples | x̄ ~ approx N(μ, σ²/n) |
| standard error | sd of the sampling distribution | σ/√n |

Because the estimator is a random variable, we can ask about its mean and variance, as for any random variable
(Lesson 03.05).

## 2. Bias, variance and MSE

$$\text{Bias}(\hat\theta) = E[\hat\theta] - \theta, \qquad \operatorname{Var}(\hat\theta) = E\big[(\hat\theta - E[\hat\theta])^2\big], \qquad \text{MSE}(\hat\theta) = E\big[(\hat\theta - \theta)^2\big].$$

**The decomposition:**

$$\text{MSE}(\hat\theta) = \text{Bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta).$$

*Proof:* write θ̂ − θ = (θ̂ − E[θ̂]) + (E[θ̂] − θ). Square it and take expectations. The cross term is
2(E[θ̂] − θ) · E[θ̂ − E[θ̂]] = 0, leaving Var + Bias². ∎

Picture a dartboard: bias is aiming at the wrong spot, variance is a shaky hand. MSE is the average squared distance
from the bullseye, and both kinds of error count.

## 3. Worked example 1: three estimators of σ²

Normal data with σ² = 4 and n = 5, 200,000 simulated samples:

| Estimator | Mean | Bias | Variance | MSE |
| --- | --- | --- | --- | --- |
| s² = Σ(x − x̄)²/(n − 1) | 3.997 | **≈ 0** | 8.05 | 8.05 |
| σ̂² = Σ(x − x̄)²/n | 3.197 | −0.80 | 5.15 | **5.80** |
| Σ(x − x̄)²/(n + 1) | 2.664 | −1.34 | 3.58 | **5.36** |

s² is unbiased (Lesson 02.03), but it has the **highest** MSE. Dividing by a larger number shrinks every estimate: that
adds bias but removes more variance. For normal data, dividing by n + 1 minimises MSE. **Unbiased is not the same as
best.** We still usually report s², because unbiasedness is convenient (it combines well across studies and underpins
the t distribution), but it's a choice, not a law.

## 4. Worked example 2: comparing estimators of a centre

**Before you start:** Uniform(0, 10), n = 10, so the true centre is 5.

| Estimator | Bias | Variance (= MSE) |
| --- | --- | --- |
| sample mean | 0 | 0.830 |
| sample median | 0 | 1.888 |
| **midrange (min + max)/2** | 0 | **0.378** |

All three are unbiased, but the midrange is more than twice as precise as the mean. For a uniform distribution the
extremes carry most of the information about where the edges (and hence the centre) are. For normal data the mean wins,
and for heavy-tailed data the median wins (Lesson 02.02, exercise 11). **The best estimator depends on the
distribution**, which is part of why maximum likelihood (Lesson 05.02) is so useful: it tailors the estimator to the
model.

## 5. Worked example 3: shrinkage — trading bias for variance

Estimate μ = 1 from n = 10 observations with σ = 4, so x̄ is unbiased with variance σ²/n = 1.6. Try shrinking towards
zero, using c·x̄:

| c | Bias | Variance | MSE |
| --- | --- | --- | --- |
| 1.0 (plain x̄) | 0 | 1.59 | 1.59 |
| 0.8 | −0.20 | 1.02 | 1.06 |
| 0.6 | −0.40 | 0.57 | 0.73 |
| 0.4 | −0.60 | 0.26 | **0.61** |

The best c is μ²/(μ² + σ²/n) ≈ 0.385. It depends on the unknown μ, so we can't use it directly, but the principle is
powerful: **when data are noisy relative to the signal, shrinking estimates helps.** This is the idea behind ridge and
lasso regression (Module 06), Bayesian priors (Module 07), weight decay in neural networks, and the James–Stein
estimator. That last one showed in 1961 that for three or more means, shrinking towards a common value *always* beats
the individual sample means in total MSE.

## 6. Consistency

An estimator is **consistent** if it converges to θ as n → ∞: θ̂ₙ → θ in probability. A sufficient condition is that
both bias and variance go to 0, so MSE → 0.

**Worked example 4.** x̄ for exponential data with σ² = 100: MSE = 20.2, 4.94, 1.25, 0.31 for n = 5, 20, 80, 320,
matching σ²/n. Consistent, by the LLN.

**Worked example 5: biased but consistent.** For Uniform(0, θ), the sample maximum always underestimates θ:
E[max] = θ n/(n + 1), which is 8.33 for n = 5 and θ = 10. The bias shrinks as n grows (9.90 at n = 100), so the
estimator is consistent. Multiplying by (n + 1)/n removes the bias exactly (mean 9.99–10.00).

Bias that vanishes with n is often acceptable; inconsistency (an estimator that stays wrong however much data you
collect, e.g. because of selection bias) is not.

## 7. Standard error: reporting precision

Always report an estimate **with its standard error** (or an interval, Lesson 05.03). "Mean exam score 63.9
(SE 0.98)" tells the reader how much to trust the number. You'll compute SEs three ways in this module: formulas
(σ/√n), likelihood curvature (Lesson 05.02) and the bootstrap (Lesson 05.04).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "Unbiased means accurate" | An unbiased estimator can have huge variance |
| Comparing estimators by one sample's result | Compare sampling distributions (simulate) |
| Ignoring bias that doesn't shrink with n | That's inconsistency: more data won't help |
| Reporting an estimate without its SE | Always give the SE or an interval |

## Summary

- Estimator = rule; estimate = number; both have a sampling distribution.
- Bias = E[θ̂] − θ; variance = spread; MSE = bias² + variance.
- Unbiased ≠ best: shrinkage adds bias to cut variance and can lower MSE.
- The best estimator depends on the data's distribution (mean, median, midrange).
- Consistent: θ̂ → θ as n → ∞; bias and variance both vanish.

## Self-test

1. Estimator A has bias 0.5 and variance 1; B has bias 0 and variance 1.5. Which has the lower MSE?
2. Prove MSE = bias² + variance.
3. Why is the sample maximum a biased estimator of θ in Uniform(0, θ)?
4. Name two places in ML where shrinkage is used.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
