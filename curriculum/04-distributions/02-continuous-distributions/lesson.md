# 04.02 — Continuous distributions

**Time:** about 4 hours · **Prerequisites:** 01.04, 04.01

## Why this matters

Measurements such as times, lengths, weights and prices are continuous. Their distributions are described by
**densities**, and probabilities are areas (Lesson 01.04). This lesson covers the PDF, the CDF and the quantile
function, the three views of every continuous distribution, and the workhorses: uniform, exponential, gamma and beta.
You'll also learn **inverse-CDF sampling**, the trick that turns uniform random numbers into draws from any distribution
(and the basis of every random number generator in `scipy.stats`).

## Learning objectives

- Move between the PDF, the CDF and the quantile function (ppf).
- Use the uniform, exponential, gamma and beta distributions and their stories.
- Explain and use the memoryless property of the exponential.
- Generate samples by inverse-CDF sampling.

## Before you start

You have already waited 20 minutes for a bus whose waiting time is exponential with mean 25 minutes. Does having waited
make it more likely the bus comes in the next 20 minutes than it was when you first arrived?

---

## 1. Three views of a continuous distribution

| View | Definition | Use |
| --- | --- | --- |
| **PDF** f(x) | density: P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx | the shape; likelihoods (Module 05) |
| **CDF** F(x) | P(X ≤ x) = ∫₋∞ˣ f(t) dt | probabilities of ranges: P(a < X ≤ b) = F(b) − F(a) |
| **Quantile** F⁻¹(q) (`ppf`) | the x with F(x) = q | percentiles, critical values, sampling |

Since F′(x) = f(x), the density is the slope of the CDF. In `scipy.stats`, every distribution has `pdf`, `cdf`,
`sf` (= 1 − cdf), `ppf` (quantile), `mean`, `std` and `rvs`. For the standard normal, `pdf(0)` = 0.399 (a density, not
a probability), `cdf(1.96)` = 0.975 and `ppf(0.975)` = 1.96.

For continuous X, P(X = x) = 0 for every single x, so ≤ and < give the same probabilities.

## 2. Uniform(a, b)

Every value between a and b is equally likely: f(x) = 1/(b − a) on [a, b].
Mean (a + b)/2, variance (b − a)²/12.

**Worked example 1.** A bus comes every 10 minutes and you arrive at a random time. Your wait W ~ Uniform(0, 10).
P(W > 7) = 3/10 = **0.30**. Mean wait **5** minutes, sd √(100/12) ≈ **2.89**.

The Uniform(0, 1) is the raw material of all simulation: `rng.uniform()`.

## 3. Exponential(λ): waiting times

The time until the next event, when events happen at a constant rate λ (the continuous partner of the Poisson):

$$f(x) = \lambda e^{-\lambda x}, \quad F(x) = 1 - e^{-\lambda x}, \quad P(X > x) = e^{-\lambda x}, \qquad x \ge 0.$$

Mean 1/λ, variance 1/λ², median ln(2)/λ. In `scipy.stats.expon` you pass `scale` = 1/λ = the mean.

**Worked example 2.** Commute times with mean 25 minutes (λ = 1/25). That's exactly how `commute_minutes` in
`students.csv` was simulated.

| Question | Formula | Answer | In students.csv |
| --- | --- | --- | --- |
| P(X > 60) | e^(−60/25) | **0.0907** | 8.0% of students |
| P(X ≤ 10) | 1 − e^(−10/25) | 0.330 | |
| median | 25 ln 2 | **17.33** min | 18 |
| 90th percentile | −25 ln(0.1) | 57.6 min | |

The mean (25) is well above the median (17.3): the exponential is right-skewed, as Lesson 02.02 predicted.

### Memorylessness

$$P(X > s + t \mid X > s) = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t).$$

**Worked example 3 (Before you start).** P(X > 40 | X > 20) = P(X > 20) = e^(−0.8) = **0.449** (simulated: 0.450).
Having already waited 20 minutes changes nothing: the remaining wait has the same distribution as a fresh wait. That
suits truly random arrivals (radioactive decay, many independent customers). It does **not** suit things that wear out
(a car part is likelier to fail when old) or scheduled processes (a real bus *is* more likely soon after a long wait).

## 4. Gamma and beta: two flexible families

| Family | Values | Story / use | Mean |
| --- | --- | --- | --- |
| Gamma(shape k, scale θ) | x > 0 | sum of k exponential waits; any positive right-skewed quantity | kθ |
| Beta(α, β) | 0 < x < 1 | a proportion or probability; the standard prior for p in Bayesian inference | α/(α + β) |

**Worked example 5.** `study_hours` in `students.csv` was simulated from a Gamma(4, 2.5): mean 10, sd 5. The data have
mean 10.18 and sd 4.92. A Beta(8, 4) belief about a success probability has mean 0.667, with 95% of its mass between
0.390 and 0.891. Module 07 uses exactly this kind of belief.

You'll meet the normal (Lesson 04.03) and the t, χ² and F distributions (Lesson 04.05) next.

## 5. Inverse-CDF sampling

**Theorem.** If U ~ Uniform(0, 1) and F is a continuous, increasing CDF, then X = F⁻¹(U) has CDF F.

*Proof:* P(X ≤ x) = P(F⁻¹(U) ≤ x) = P(U ≤ F(x)) = F(x), because P(U ≤ u) = u. ∎

**Worked example 4.**

- **Exponential:** solve u = 1 − e^(−λx) for x to get x = −ln(1 − u)/λ. 200,000 draws this way have mean 24.91
  (expected 25) and median 17.27 (expected 17.33).
- **Density 2x on [0, 1]:** F(x) = x², so F⁻¹(u) = √u. That's why `np.sqrt(U)` worked in Lesson 01.04: mean 0.6665
  (expected 2/3), P(≤ 0.5) = 0.249 (expected 0.25).

The reverse also holds: if X has continuous CDF F, then F(X) ~ Uniform(0, 1). This *probability integral transform*
underlies Q–Q plots (Lesson 04.03), p-values (a p-value is uniform under the null, Module 05) and calibration checks
(Module 08).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Reading f(x) as a probability | Probabilities are areas; f can exceed 1 |
| `expon(λ)` in scipy | scipy uses `scale = 1/λ`: `stats.expon(scale=25)` |
| Assuming memorylessness for things that age | It holds only for exponential (and geometric) waits |
| `sf` vs `cdf` confusion | sf(x) = P(X > x) = 1 − cdf(x) |

## Summary

- PDF (shape), CDF (P(X ≤ x)), quantile (`ppf`): three views; f = F′.
- Uniform(a, b): flat, mean (a + b)/2, variance (b − a)²/12.
- Exponential(λ): waiting times, P(X > x) = e^(−λx), mean 1/λ, median ln 2/λ, memoryless.
- Gamma for positive skewed values; beta for proportions.
- Inverse-CDF sampling: X = F⁻¹(U).

## Self-test

1. X ~ Uniform(2, 8). Find P(3 < X < 5), the mean and the variance.
2. Light bulbs fail at rate 1 per 1,000 hours (exponential). P(a bulb lasts more than 2,000 hours)? Its median lifetime?
3. Write the inverse CDF of the exponential distribution.
4. What does `stats.norm.ppf(0.95)` return, and what does it mean?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
