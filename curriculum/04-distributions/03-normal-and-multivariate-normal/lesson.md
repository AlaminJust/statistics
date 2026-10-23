# 04.03 — The normal and the multivariate normal

**Time:** about 4 hours · **Prerequisites:** 04.02, 03.06, 02.04, 01.05

## Why this matters

The normal (Gaussian) distribution is the most important distribution in statistics. Measurement errors, heights and
test scores are roughly normal. More importantly, **averages are approximately normal whatever the data look like**
(the central limit theorem, Lesson 04.04), so confidence intervals and tests are built on it. In ML the normal is the
noise model behind least squares, the prior behind ridge regression and weight initialisation, and the latent space of
VAEs. Its multivariate version connects means, covariance matrices and regression in one object.

## Learning objectives

- Use the normal PDF, standardise to z-scores, and compute probabilities and percentiles.
- Know the key values: 68–95–99.7 and 1.645, 1.96, 2.576.
- Combine independent normals, and approximate the binomial by a normal.
- Check normality with Q–Q plots.
- Work with the bivariate normal: covariance matrix, contours, and the conditional mean (a preview of regression).

## Before you start

A student is 1.5 standard deviations above average on the midterm. The midterm and final correlate at 0.6. How many
standard deviations above average do you expect them to be on the final? 1.5? More? Less?

---

## 1. The normal distribution N(μ, σ²)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).$$

- Symmetric bell shape centred at μ; σ sets the width (the inflection points are at μ ± σ).
- The mean, median and mode all equal μ.
- The tails fall off very fast, as e^(−x²): extreme values are extremely rare.
- Fully described by two numbers, μ and σ.

You met the shape e^(−x²/2) in Lesson 01.02 and checked that the standard normal density integrates to 1 in
Lesson 01.04.

## 2. The standard normal and z-scores

If X ~ N(μ, σ²), then **Z = (X − μ)/σ ~ N(0, 1)**, the standard normal. Its CDF is written Φ(z). Every normal question
becomes a standard-normal question by standardising, exactly as with z-scores in Lesson 02.04.

**Worked example 1.** IQ ~ N(100, 15²).

| Question | Standardise | Answer |
| --- | --- | --- |
| P(IQ > 130) | z = (130 − 100)/15 = 2 → 1 − Φ(2) | **0.0228** |
| P(85 < IQ < 115) | z from −1 to 1 → Φ(1) − Φ(−1) | **0.6827** |
| 90th percentile | 100 + 15 × Φ⁻¹(0.9) = 100 + 15 × 1.2816 | **119.2** |

In code: `stats.norm(100, 15).sf(130)`, `.cdf(115) - .cdf(85)`, `.ppf(0.9)`.

**Values to memorise:**

| Central probability | 68.3% | 90% | 95% | 95.4% | 99% | 99.7% |
| --- | --- | --- | --- | --- | --- | --- |
| ± z | 1 | **1.645** | **1.96** | 2 | **2.576** | 3 |

The 1.96 gives the "95%" confidence intervals of Module 05.

## 3. Sums and averages of independent normals

If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²) are independent, then **X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)**: still normal, with means
adding and variances adding (Lesson 03.06). Also aX + b ~ N(aμ + b, a²σ²).

**Worked example 2.** Walking time ~ N(10, 2²) and train time ~ N(25, 4²), independent. Total ~ N(35, 20), so
sd = √20 = 4.47. P(total > 40) = P(Z > 5/4.47 = 1.118) = **0.132** (simulated: 0.132).

**The mean of n i.i.d. normals** is N(μ, σ²/n). The mean IQ of 25 random people ~ N(100, 3²), so P(mean > 106) =
P(Z > 2) = 0.0228. One person above 130 is as rare as a *group average* above 106.

## 4. The normal approximation to the binomial

For large n, Binomial(n, p) ≈ N(np, np(1 − p)). (It's the CLT of Lesson 04.04 applied to a sum of Bernoullis.) Rule of
thumb: it works when np and n(1 − p) are both at least 10.

**Worked example 3.** 100 fair flips: X ≈ N(50, 5²). P(X ≥ 60)?

| Method | Value |
| --- | --- |
| exact binomial | 0.0284 |
| normal, P(Z > (60 − 50)/5) | 0.0228 |
| normal with **continuity correction**, P(Z > (59.5 − 50)/5) | **0.0287** |

The continuity correction treats the integer 60 as the interval [59.5, 60.5], because a continuous curve is
approximating bars.

## 5. Is my data normal? Q–Q plots

A **Q–Q (quantile–quantile) plot** puts the sorted data against the quantiles a normal distribution would predict. If
the data are normal, the points lie on a straight line (`qq_plots.png`).

| Pattern in the Q–Q plot | Meaning |
| --- | --- |
| straight line | approximately normal |
| curve bending up at the right end | right skew (long upper tail) |
| S-shape, both ends steeper than the line | heavier tails than normal (outliers more common) |
| steps | discrete or rounded data |

**Worked example 4.**

| Variable | Skewness | Q–Q correlation |
| --- | --- | --- |
| exam_score | +0.05 | 0.998 |
| sleep_hours | −0.09 | 0.997 |
| commute_minutes | +1.37 | 0.930: clearly curved |

Two practical points: real data are never *exactly* normal, so the question is whether they are normal *enough* for
the method you plan to use. And many methods (t-tests, regression coefficients) need normality of the **averages or
errors**, not of the raw data, which the CLT often provides.

## 6. The bivariate and multivariate normal

A random vector **X** = (X₁, …, X_p) is **multivariate normal**, N(**μ**, Σ), with mean vector **μ** and covariance
matrix Σ (Lesson 03.06):

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol\mu)\right).$$

Key facts:

- The density's contours are **ellipses** (ellipsoids in higher dimensions), tilted by the correlation (`bivariate_normal.png`).
- Every **marginal** is normal: Xᵢ ~ N(μᵢ, Σᵢᵢ).
- Every linear combination **w**ᵀ**X** is normal, with variance **w**ᵀΣ**w**.
- For a multivariate normal, **zero correlation does imply independence** (unlike in general, Lesson 03.06).
- Every **conditional** is normal, with a mean that is linear in the conditioning variable.

### The conditional distribution, and why regression is linear

For a bivariate normal with correlation ρ:

$$E[Y \mid X = x] = \mu_Y + \rho\,\frac{\sigma_Y}{\sigma_X}(x - \mu_X), \qquad \operatorname{sd}(Y \mid X = x) = \sigma_Y\sqrt{1 - \rho^2}.$$

The conditional mean is a **straight line** in x, with slope ρσ_Y/σ_X. That's the population regression line
(Module 06). The conditional spread is smaller than σ_Y: knowing X reduces uncertainty about Y, by a factor √(1 − ρ²).

**Worked example 5.** Midterm X ~ N(60, 10²), final Y ~ N(65, 12²), ρ = 0.6. Covariance matrix
[[100, 72], [72, 144]] (Cov = 0.6 × 10 × 12 = 72). A student scores 75 on the midterm:

- E[final | 75] = 65 + 0.6 × (12/10) × (75 − 60) = 65 + 10.8 = **75.8**.
- sd(final | 75) = 12 × √0.64 = **9.6**, down from 12.
- Simulated (midterms within 0.5 of 75): mean 75.9, sd 9.67.

**Regression to the mean (Before you start).** The student is (75 − 60)/10 = 1.5 sd above average on the midterm, but
expected only (75.8 − 65)/12 = **0.9 sd** above on the final, which is ρ × 1.5. With |ρ| < 1, extreme performers are
expected to be *less extreme* next time, with no change in ability. Many "treatments" applied to the worst performers
look effective only because of this. Module 09 shows how a control group protects you.

### Sampling from N(**μ**, Σ)

Factor Σ = LLᵀ (the **Cholesky** decomposition). If **z** is a vector of independent standard normals, then
**μ** + L**z** ~ N(**μ**, Σ). `examples.py` does this by hand and recovers the covariance matrix
[[100.3, 72.1], [72.1, 143.9]]. VAEs use the same trick (the "reparameterisation trick") to sample in a way that
gradients can pass through.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| N(μ, σ) vs N(μ, σ²) confusion | Here the second number is the **variance**; scipy's `norm(loc, scale)` takes the **sd** |
| sd(X + Y) = sd(X) + sd(Y) | Variances add (for independent variables) |
| Requiring raw data to be normal for every method | Often only averages or errors need to be |
| Treating zero correlation as independence in general | Only for jointly normal variables |
| Expecting extremes to repeat | Regression to the mean |

## Summary

- N(μ, σ²): symmetric bell; standardise with z = (x − μ)/σ; Φ is the standard normal CDF.
- 68–95–99.7; 1.645 (90%), 1.96 (95%), 2.576 (99%).
- Independent normals add: means add, variances add. The mean of n i.i.d. normals ~ N(μ, σ²/n).
- Binomial ≈ N(np, np(1 − p)) for large n, with a continuity correction.
- Q–Q plot: a straight line means normal.
- Multivariate normal: elliptical contours, normal marginals and conditionals, and E[Y | X] linear with slope ρσ_Y/σ_X.

## Self-test

1. Heights ~ N(170, 8²) cm. What share are taller than 186 cm? What height is the 25th percentile?
2. X ~ N(5, 9) and Y ~ N(3, 16), independent. Distribution of X − Y?
3. Why does an S-shaped Q–Q plot suggest heavy tails?
4. If ρ = 0, what is E[Y | X = x]? What if ρ = 1?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
