# 05.02 — Maximum likelihood estimation

**Time:** about 4 hours · **Prerequisites:** 01.02, 01.03, 04.01–04.03, 04.06, 05.01

## Why this matters

**Maximum likelihood** (MLE) is the most widely used way to fit a statistical model: choose the parameter values that
make the observed data most probable. Nearly every ML training objective is a likelihood in disguise. Mean squared
error is the normal likelihood, cross-entropy is the Bernoulli or categorical likelihood (Lesson 04.06), and Poisson
loss is the Poisson likelihood. MLE also comes with a recipe for **standard errors**: the sharper the peak of the
likelihood, the more precise the estimate.

## Learning objectives

- Write the likelihood and log-likelihood for i.i.d. data.
- Derive MLEs by calculus for the Bernoulli, Poisson and normal.
- Find MLEs numerically with `scipy.optimize` when there's no formula.
- Compute standard errors from the Fisher information.
- Show that least squares is the MLE under normal errors.

## Before you start

A coin lands heads 7 times in 10 flips. Which value of P(heads) makes this result most likely? Can you prove it?

---

## 1. The likelihood

For i.i.d. data x₁, …, xₙ from a model with density or PMF f(x | θ), the **likelihood** is the probability of the data,
viewed as a function of the parameter:

$$L(\theta) = \prod_{i=1}^n f(x_i \mid \theta), \qquad \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta).$$

- Independence turns the joint probability into a product (Lesson 03.03).
- We maximise the **log-likelihood** ℓ: it's a sum, it doesn't underflow (Lesson 01.02), and it peaks at the same θ.
- The likelihood is **not** a probability distribution over θ. It doesn't integrate to 1 over θ. (Bayesian inference,
  Module 07, turns it into one by multiplying by a prior.)

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \big[-\ell(\theta)\big].$$

Minimising the **negative log-likelihood** (NLL) is how ML phrases it: the NLL is the loss.

## 2. Worked example 1: the coin

7 heads in 10 flips: L(p) = p⁷(1 − p)³ (the C(10, 7) factor doesn't depend on p, so it can be dropped).

| p | 0.3 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| L(p) | 0.000075 | 0.000977 | 0.001792 | **0.002224** | 0.001678 | 0.000478 |
| ℓ(p) | −9.50 | −6.93 | −6.32 | **−6.11** | −6.39 | −7.65 |

**By calculus:** ℓ(p) = 7 ln p + 3 ln(1 − p). Setting ℓ′(p) = 7/p − 3/(1 − p) = 0 gives 7(1 − p) = 3p, so **p̂ = 7/10**.
In general, for k successes in n Bernoulli trials, **p̂ = k/n**, the sample proportion.

## 3. More closed-form MLEs

**Poisson(λ).** ℓ(λ) = Σ[xᵢ ln λ − λ − ln xᵢ!]. ℓ′ = Σxᵢ/λ − n = 0 ⇒ **λ̂ = x̄**. Worked example 2 checks this on
simulated goal counts: the grid maximum of ℓ is at λ = 2.437, which equals the sample mean.

**Normal(μ, σ²).**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum (x_i - \mu)^2.$$

- ∂ℓ/∂μ = 0 ⇒ **μ̂ = x̄**.
- ∂ℓ/∂σ² = 0 ⇒ **σ̂² = (1/n)Σ(xᵢ − x̄)²**, which divides by n, not n − 1. **The MLE of the variance is biased** (Lesson 05.01).

**Worked example 3.** For `exam_score`: μ̂ = 63.93 and σ̂² = 190.53 (vs s² = 191.48). `scipy.stats.norm.fit` returns
exactly the MLE.

**Exponential(λ):** λ̂ = 1/x̄. **Uniform(0, θ):** θ̂ = max(xᵢ). This one can't be found by calculus, because the
likelihood (1/θ)ⁿ is decreasing but θ must be at least every observation (Lesson 05.01, worked example 5).

## 4. Numerical MLE: when there's no formula

Most models have no closed form. Then you **minimise the NLL with an optimiser**, the same approach that trains neural
networks.

**Worked example 4: a gamma model for study_hours.** The gamma has a shape k and a scale θ, both positive, and no
closed-form MLE for k.

```python
def negative_log_likelihood(params):
    log_shape, log_scale = params          # optimise on the log scale so both stay positive
    return -stats.gamma.logpdf(h, a=np.exp(log_shape), scale=np.exp(log_scale)).sum()

result = optimize.minimize(negative_log_likelihood, x0=[0.0, 0.0], method="Nelder-Mead")
```

Result: shape **4.118** and scale **2.471**. The data were simulated with 4 and 2.5. `stats.gamma.fit(h, floc=0)` agrees.
The NLL at the MLE (589.07) is lower than at the true parameters (589.24): with finite data, the MLE fits *this* sample
slightly better than the truth does.

Tips: re-parameterise constrained parameters (log for positive values, logit for probabilities); try several starting
points; check `result.success`.

## 5. Properties of the MLE

Under mild conditions, as n → ∞ the MLE is:

| Property | Meaning |
| --- | --- |
| **Consistent** | θ̂ → θ |
| **Asymptotically normal** | θ̂ ≈ N(θ, 1/I(θ)) |
| **Efficient** | no consistent estimator has a smaller asymptotic variance |
| **Invariant** | the MLE of g(θ) is g(θ̂): the MLE of σ is √σ̂², of the odds p/(1 − p) is p̂/(1 − p̂) |

The MLE can be biased for small n (σ̂²), and it assumes the model is right. If the model is wrong, it converges to the
parameter that makes the model closest to the truth in KL divergence (Lesson 04.06).

## 6. Standard errors from the Fisher information

The **Fisher information** measures how sharply the log-likelihood curves at its peak:

$$I(\hat\theta) = -\ell''(\hat\theta), \qquad \text{SE}(\hat\theta) \approx \frac{1}{\sqrt{I(\hat\theta)}}.$$

A sharp peak means the data strongly favour a narrow range of θ, so the SE is small (`likelihood_curves.png`).

**Worked example 5: Bernoulli.** ℓ″(p) = −k/p² − (n − k)/(1 − p)². At p̂ = k/n this equals −n/(p̂(1 − p̂)). So
SE = √(p̂(1 − p̂)/n), the familiar formula. For 7 of 10: I = 47.62, so **SE = 0.145**. A numerical second derivative of
ℓ gives the same curvature, and 100,000 simulated MLEs with true p = 0.7 have sd 0.145.

With 70 of 100 the log-likelihood is ten times as curved, and the SE shrinks by √10. With several parameters, I is a
matrix (the negative Hessian), and SEs are the square roots of the diagonal of its inverse. Optimisers can estimate it
for you, and statsmodels reports SEs this way.

## 7. Least squares is maximum likelihood

Suppose yᵢ = β₀ + β₁xᵢ + εᵢ with εᵢ ~ N(0, σ²). Then

$$-\ell(\beta_0, \beta_1, \sigma) = \frac{n}{2}\ln(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i \big(y_i - \beta_0 - \beta_1 x_i\big)^2.$$

For any σ, minimising this over the β's means minimising **the sum of squared errors**. So OLS = normal MLE.

**Worked example 6.** Maximising the normal likelihood numerically for exam_score vs study_hours gives intercept 43.092
and slope 2.0476, identical to OLS from the normal equations (Lesson 01.05).

| Loss you minimise | Likelihood it comes from |
| --- | --- |
| squared error | normal |
| absolute error | Laplace (whose MLE of location is the median) |
| binary cross-entropy | Bernoulli |
| categorical cross-entropy | categorical |
| Poisson deviance | Poisson |

Choosing a loss function is choosing a noise model.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Treating L(θ) as the probability of θ | It's the probability of the data given θ |
| Multiplying densities directly | Sum log-densities |
| Optimising a positive parameter directly | Optimise its log |
| Reporting σ̂² as unbiased | The MLE divides by n |
| Trusting MLE SEs for tiny n | They're large-sample approximations; bootstrap to check |

## Summary

- L(θ) = Π f(xᵢ | θ); maximise ℓ(θ) = Σ log f(xᵢ | θ), or minimise the NLL.
- Closed forms: Bernoulli p̂ = k/n; Poisson λ̂ = x̄; normal μ̂ = x̄, σ̂² = (1/n)Σ(x − x̄)².
- No formula: `scipy.optimize.minimize` on the NLL, with parameters transformed to be unconstrained.
- SE ≈ 1/√(−ℓ″(θ̂)); the MLE is consistent, asymptotically normal, efficient and invariant.
- Least squares = MLE with normal errors; every common ML loss is an NLL.

## Self-test

1. Derive the MLE of λ for exponential data.
2. 12 successes in 40 trials: find the MLE and its SE.
3. Why do we maximise the log-likelihood instead of the likelihood?
4. What noise model is implied by minimising absolute error?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
