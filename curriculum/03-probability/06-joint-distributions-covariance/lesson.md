# 03.06 — Joint distributions and covariance

**Time:** about 4 hours · **Prerequisites:** 03.05, 02.06, 01.05

## Why this matters

Real problems involve several random quantities at once: features and labels, the returns of several stocks, the
errors of different models. A **joint distribution** describes them together. From it come marginals (each variable on
its own), conditionals (one variable given another) and **covariance**, the population version of Lesson 02.06's
sample covariance. This lesson also proves a result used everywhere from here on: **the variance of an average of n
independent observations is σ²/n**. That's why larger samples give more precise estimates.

## Learning objectives

- Read a joint PMF and compute marginal and conditional distributions.
- Test whether two random variables are independent.
- Compute Cov(X, Y) and Corr(X, Y) for random variables.
- Use Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y), and derive Var(X̄) = σ²/n.
- Explain why zero covariance does not imply independence.
- Use a covariance matrix to get the variance of any weighted sum.

## Before you start

Two investments each have a 20% standard deviation of yearly return. You split your money equally between them. Guess
the standard deviation of your portfolio if their returns are (a) perfectly correlated, (b) unrelated.

---

## 1. Joint, marginal and conditional distributions

The **joint PMF** of X and Y is p(x, y) = P(X = x and Y = y). It's a table that sums to 1.

**Worked example 1.** X = 1 if a student studied, Y = 1 if they passed:

| | Y = 0 (fail) | Y = 1 (pass) | **p_X(x)** |
| --- | --- | --- | --- |
| X = 0 | 0.30 | 0.10 | **0.40** |
| X = 1 | 0.20 | 0.40 | **0.60** |
| **p_Y(y)** | **0.50** | **0.50** | 1 |

- **Marginals** are the row and column totals, written in the margins (hence the name): p_X(x) = Σᵧ p(x, y).
- **Conditionals** divide a row by its total: P(Y = 1 | X = 1) = 0.40/0.60 = **0.667**, and P(Y = 1 | X = 0) =
  0.10/0.40 = **0.25**. Studying is associated with passing.

This is the probability version of Lesson 02.06's contingency table: the same calculation with probabilities instead
of counts.

## 2. Independence of random variables

X and Y are **independent** if p(x, y) = p_X(x) p_Y(y) for **every** cell. One failing cell is enough to make them
dependent.

Here p(0, 0) = 0.30 but p_X(0)p_Y(0) = 0.40 × 0.50 = 0.20, so they're **dependent**.

For independent X and Y: E[XY] = E[X]E[Y], every conditional distribution equals the marginal, and functions of them,
g(X) and h(Y), are also independent.

## 3. Covariance and correlation of random variables

$$\operatorname{Cov}(X, Y) = E\big[(X - \mu_X)(Y - \mu_Y)\big] = E[XY] - E[X]\,E[Y].$$

$$\operatorname{Corr}(X, Y) = \rho = \frac{\operatorname{Cov}(X, Y)}{\operatorname{sd}(X)\,\operatorname{sd}(Y)} \in [-1, 1].$$

**Worked example 1, continued.**

| Quantity | Calculation | Value |
| --- | --- | --- |
| E[X] | 0 × 0.4 + 1 × 0.6 | 0.6 |
| E[Y] | 0 × 0.5 + 1 × 0.5 | 0.5 |
| E[XY] | only the (1, 1) cell has XY ≠ 0: 1 × 0.40 | 0.40 |
| Cov(X, Y) | 0.40 − 0.6 × 0.5 | **0.10** |
| Var(X) | 0.6 × 0.4 (Bernoulli) | 0.24 |
| Var(Y) | 0.5 × 0.5 | 0.25 |
| Corr(X, Y) | 0.10 / √(0.24 × 0.25) | **0.408** |

500,000 simulated pairs give Cov 0.1001 and Corr 0.409.

The sample covariance of Lesson 02.06 *estimates* this population covariance from data, just as x̄ estimates μ.

Properties: Cov(X, X) = Var(X); Cov(aX + b, cY + d) = ac Cov(X, Y); independence ⇒ Cov = 0.

### Zero covariance does not mean independence

**Worked example 2.** X is −1, 0 or 1, each with probability 1/3, and Y = X². Y is completely determined by X, so they
are as dependent as possible. But E[X] = 0 and E[XY] = E[X³] = 0, so **Cov = 0**. Covariance measures only *linear*
association: this is Lesson 02.06's parabola again.

## 4. The variance of a sum

$$\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y).$$

*Proof:* expand E[((X − μ_X) + (Y − μ_Y))²] and use linearity.

Worked example 1: Var(X + Y) = 0.24 + 0.25 + 2 × 0.10 = **0.69** (simulated: 0.690).

For **independent** (or merely uncorrelated) variables the covariance term vanishes and **variances add**.
Standard deviations never add: sd(X + Y) = √(Var(X) + Var(Y)) for independent X and Y.

**Worked example 3.** Var(sum of two dice) = 35/12 + 35/12 = **5.833** (simulated: 5.838).

## 5. The most important consequence: Var(X̄) = σ²/n

Let X₁, …, Xₙ be **independent and identically distributed** (i.i.d.), each with mean μ and variance σ². Their average
is X̄ = (X₁ + … + Xₙ)/n.

- E[X̄] = (1/n) × nμ = **μ** (by linearity).
- Var(X̄) = (1/n²) × Var(X₁ + … + Xₙ) = (1/n²) × nσ² = **σ²/n** (by the scale rule, and variances adding).
- sd(X̄) = **σ/√n**. This is called the **standard error** of the mean.

**Worked example 4.** Averages of n die rolls:

| n | Var(X̄) = (35/12)/n | Simulated | sd(X̄) |
| --- | --- | --- | --- |
| 1 | 2.917 | 2.926 | 1.708 |
| 4 | 0.729 | 0.727 | 0.854 |
| 16 | 0.182 | 0.183 | 0.427 |
| 64 | 0.046 | 0.045 | 0.214 |

**Four times the data halves the standard error.** This √n law explains Lesson 02.01, exercise 10 (sample means
cluster more tightly as n grows), Lesson 02.03, exercise 10 (it was the E[(x̄ − μ)²] = σ²/n step), the confidence
intervals of Module 05 and the error of Monte Carlo estimates (Lesson 01.04, exercise 11).

**The i.i.d. assumption matters.** If observations are positively correlated (repeated measurements of the same
person, days in a time series), Var(X̄) is larger than σ²/n, and naive standard errors are too small.

## 6. Diversification: covariance in action

**Worked example 5 (Before you start).** Half your money in each of two assets with sd 0.20 and correlation ρ:

$$\operatorname{Var}\big(\tfrac{1}{2}A + \tfrac{1}{2}B\big) = \tfrac{1}{4}(0.2)^2 + \tfrac{1}{4}(0.2)^2 + 2 \cdot \tfrac{1}{4}\,\rho\,(0.2)(0.2).$$

| Correlation ρ | Portfolio sd |
| --- | --- |
| +1.0 | 0.200 (no benefit) |
| +0.5 | 0.173 |
| 0.0 | **0.141** (= 0.2/√2) |
| −0.5 | 0.100 |

Combining imperfectly correlated things reduces risk. The same principle explains why **ensembles** of ML models (random
forests, deep ensembles) beat their members: the errors of different models are not perfectly correlated, so averaging
cancels part of them.

## 7. Many variables: the covariance matrix

For a random vector (X₁, …, X_p), the **covariance matrix** Σ has Var(Xᵢ) on the diagonal and Cov(Xᵢ, Xⱼ) off it. It's
symmetric. For any weights **w**, the weighted sum **w**ᵀ**X** has

$$\operatorname{Var}(\mathbf{w}^\top \mathbf{X}) = \mathbf{w}^\top \Sigma\, \mathbf{w}.$$

**Worked example 6.** Sample covariance matrix of `students.csv`:

| | study | sleep | score |
| --- | --- | --- | --- |
| study_hours | 24.20 | −0.39 | 49.55 |
| sleep_hours | −0.39 | 0.99 | 1.76 |
| exam_score | 49.55 | 1.76 | 191.48 |

With **w** = (1, 1, 0), **w**ᵀΣ**w** = 24.20 + 0.99 + 2(−0.39) = 24.41, which matches the variance of
study_hours + sleep_hours computed directly. The multivariate normal (Module 04), PCA (Module 08) and portfolio
optimisation are all built on this matrix.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Checking independence in one cell only | It must hold in every cell |
| Cov = 0 ⇒ independent | Only "no linear relationship" |
| Var(X − Y) = Var(X) − Var(Y) | Var(X − Y) = Var(X) + Var(Y) − 2Cov(X, Y) |
| sd(X̄) = σ/n | σ/√n |
| Using σ/√n for correlated data | Correlation inflates the true standard error |

## Summary

- Joint PMF → marginals (sum rows or columns) → conditionals (divide by the marginal).
- Independent ⇔ p(x, y) = p_X(x)p_Y(y) everywhere.
- Cov(X, Y) = E[XY] − E[X]E[Y]; Corr = Cov/(sd sd); Cov = 0 does not imply independence.
- Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y).
- i.i.d. average: E[X̄] = μ, Var(X̄) = σ²/n, standard error σ/√n.
- Var(**w**ᵀ**X**) = **w**ᵀΣ**w**.

## Self-test

1. From the table in section 1, find P(X = 1 | Y = 1).
2. Var(X) = 4, Var(Y) = 9, Cov(X, Y) = −2. Find Var(X + Y) and Var(X − Y).
3. How many observations do you need to cut the standard error of a mean to one third?
4. Give an example of dependent variables with zero covariance.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
