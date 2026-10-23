# 04.04 — The law of large numbers and the central limit theorem

**Time:** about 4 hours · **Prerequisites:** 03.06, 04.03

## Why this matters

Two theorems make statistics possible:

- The **law of large numbers (LLN)**: averages of many independent observations get close to the true mean. That's why
  sample statistics estimate population parameters, why Monte Carlo simulation works, and why a test-set accuracy
  estimates real-world accuracy.
- The **central limit theorem (CLT)**: the *error* of an average is approximately normal, whatever the shape of the
  data, with sd σ/√n. That's why confidence intervals and tests use the normal distribution even for skewed data.

This lesson also covers **concentration inequalities**, guaranteed bounds on how far an average can stray. They are the
starting point of ML learning theory (Module 08).

## Learning objectives

- State the LLN and the CLT, and show both by simulation.
- Use the CLT to approximate the distribution of a mean or a proportion.
- Judge when n is large enough, and recognise when the CLT fails.
- Apply Markov's, Chebyshev's and Hoeffding's inequalities, including to choose a test-set size.

## Before you start

Commute times are strongly right-skewed. If you average the commutes of 30 random students, and do that many times, what
shape will the histogram of those averages have?

---

## 1. The law of large numbers

If X₁, X₂, … are i.i.d. with mean μ, then the sample mean X̄ₙ gets arbitrarily close to μ as n grows:

$$P\big(|\bar{X}_n - \mu| > \varepsilon\big) \to 0 \quad\text{for every } \varepsilon > 0.$$

*Why:* Var(X̄ₙ) = σ²/n → 0 (Lesson 03.06), and Chebyshev's inequality (section 5) turns a vanishing variance into a
vanishing probability of a large error.

**Simulation** (`lln_running_mean.png`): the running mean of exponential draws with mean 25 is 23.8 after 10 draws, 23.1
after 1,000 and 24.99 after 100,000. Early averages wander, late ones settle.

**What the LLN does *not* say:** it doesn't say the *count* of heads gets close to n/2. The count's typical deviation
grows like √n; only the *proportion* settles. Nor does it say short runs balance out (the gambler's fallacy, Lesson
03.03).

**When it fails:** the **Cauchy distribution** has no mean (its tails are so heavy that ∫|x|f(x)dx is infinite). Its
running mean jumps around forever: −3.3, −0.35, 1.24, −1.19 and then 18.6 after 100,000 draws. A single huge value can
undo everything before it. Some real data (financial losses, file sizes, city populations) have very heavy tails and
behave a little like this.

## 2. The central limit theorem

If X₁, …, Xₙ are i.i.d. with mean μ and *finite* variance σ², then for large n

$$\bar{X}_n \approx N\!\left(\mu, \frac{\sigma^2}{n}\right), \qquad\text{equivalently}\qquad \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \approx N(0, 1).$$

The same holds for the sum: ΣXᵢ ≈ N(nμ, nσ²). The original distribution can be skewed, discrete or lumpy; only its
mean and variance matter in the limit.

**Simulation: means of exponential data** (skewness 2), 100,000 samples for each n (`clt_exponential.png`):

| n | sd of mean (theory 1/√n) | Skewness of mean (theory 2/√n) | P(Z < −1.96) | P(Z > 1.96) |
| --- | --- | --- | --- | --- |
| 1 | 1.001 | 1.99 | 0.0000 | 0.0521 |
| 2 | 0.708 | 1.42 | 0.0000 | 0.0485 |
| 5 | 0.448 | 0.89 | 0.0004 | 0.0432 |
| 30 | 0.183 | 0.38 | 0.0140 | 0.0347 |
| normal | | 0 | 0.025 | 0.025 |

Answer to "Before you start": the averages are **approximately normal**, centred on the true mean, with a slight right
skew left over. Three lessons from the table:

1. The **sd follows σ/√n exactly**, for any n (that part is not an approximation).
2. The **skewness of the mean shrinks like 1/√n**. Strongly skewed data need larger n.
3. The **tails converge slowest**. At n = 30 the lower and upper tails are 1.4% and 3.5% instead of 2.5% each. The
   rule "n ≥ 30 is enough" is a loose guide that's fine for mild skew, not a law.

### The CLT for proportions

A proportion is the mean of 0/1 values, so p̂ ≈ N(p, p(1 − p)/n). With a rare event (p = 0.03, like a click-through
rate):

| n | sd of p̂ (theory) | Skewness | P(p̂ = 0) |
| --- | --- | --- | --- |
| 30 | 0.031 (0.031) | 1.03 | 0.40 |
| 300 | 0.0099 (0.0098) | 0.31 | 0.0001 |
| 3000 | 0.0031 (0.0031) | 0.10 | 0 |

With n = 30 the normal approximation is useless: 40% of samples contain no clicks at all. For rare events you need
np ≥ 10 or so, here n ≥ 300 or more (Lesson 04.03's rule of thumb).

## 3. Why the CLT matters so much

- **Confidence intervals** (Module 05): x̄ ± 1.96 × s/√n works for non-normal data because x̄ is approximately normal.
- **Tests** compare (x̄ − μ₀)/(s/√n) with a normal (or t) distribution.
- **Measurement error and many natural quantities** are roughly normal because they are sums of many small independent
  effects.
- **Monte Carlo error bars**: a simulated probability has standard error √(p(1 − p)/n).
- **Stochastic gradient descent**: a mini-batch gradient is an average of per-example gradients, so its noise is
  roughly normal with variance shrinking like 1/(batch size).

**When the CLT breaks down:** infinite variance (Cauchy, some power laws); strong dependence between observations
(Lesson 03.06, exercise 11); or n too small for the amount of skew.

## 4. Markov's and Chebyshev's inequalities

These give **guaranteed** bounds that need almost no assumptions. They're loose, but always true.

**Markov:** for X ≥ 0 and a > 0, P(X ≥ a) ≤ E[X]/a.
*Example:* if the average commute is 25 minutes, at most 25/100 = 25% of commutes can be 100 minutes or more, whatever
the distribution.

**Chebyshev:** for any X with mean μ and sd σ, P(|X − μ| ≥ kσ) ≤ 1/k². (Lesson 02.04 used it in the form "at least
1 − 1/k² within k sd".)

## 5. Hoeffding's inequality: the ML workhorse

If X₁, …, Xₙ are independent with values in [0, 1] (like "this test example was classified correctly"), then

$$P\big(|\bar{X}_n - \mu| \ge \varepsilon\big) \le 2\exp(-2n\varepsilon^2).$$

The bound falls **exponentially** in n, much faster than Chebyshev's 1/n.

**Worked example: how big must a test set be?** To estimate accuracy within ±ε with probability at least 1 − δ, set
2e^(−2nε²) ≤ δ:

$$n \ge \frac{\ln(2/\delta)}{2\varepsilon^2}.$$

For ε = 0.02 and δ = 0.05: n ≥ ln(40)/(2 × 0.0004) = 3.689/0.0008 ≈ **4,612** test examples. The CLT version
(worst case p = 0.5) needs n ≥ (1.96/0.02)² × 0.25 = **2,401**: smaller, but only approximately valid. Hoeffding is a
guarantee for any n.

**Comparing the bounds** (n = 1,000, true accuracy 0.9, ε = 0.03):

| Method | Bound on P(\|accuracy − 0.9\| ≥ 0.03) |
| --- | --- |
| Chebyshev, using the true variance 0.09 | 0.100 |
| Hoeffding | 0.331 |
| actual (simulated) | **0.0019** |

Both bounds are true but loose. Chebyshev wins here because it uses the variance, which is small when accuracy is near
1; Hoeffding only uses the fact that values lie in [0, 1]. In Module 08 the same inequality, combined with a union bound
over many models, gives *generalisation bounds*: why a model that fits training data well should also do well on new
data.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "The CLT makes the data normal" | It makes the **mean** (or sum) approximately normal |
| "n ≥ 30 always suffices" | Depends on skew; rare-event proportions need much more |
| The LLN means counts balance out | Only proportions settle; counts drift like √n |
| Using σ/√n for dependent data | Correlation inflates the error |
| Bounds as estimates | Inequalities are worst-case guarantees, often far from the truth |

## Summary

- LLN: X̄ₙ → μ. It fails without a finite mean (Cauchy).
- CLT: X̄ₙ ≈ N(μ, σ²/n) for i.i.d. data with finite variance; the skewness of the mean shrinks like 1/√n; tails converge slowest.
- Proportions: p̂ ≈ N(p, p(1 − p)/n) when np and n(1 − p) are large.
- Markov: P(X ≥ a) ≤ E[X]/a. Chebyshev: P(|X − μ| ≥ kσ) ≤ 1/k². Hoeffding: P(|X̄ − μ| ≥ ε) ≤ 2e^(−2nε²).
- Test-set size: n ≥ ln(2/δ)/(2ε²).

## Self-test

1. Exam scores have mean 64 and sd 14. Approximate P(the mean of 49 random students exceeds 67).
2. Why does the CLT need finite variance?
3. How many test examples guarantee accuracy within ±0.01 with 99% confidence, by Hoeffding?
4. A coin is flipped 10,000 times. What is the approximate sd of the number of heads? Of the proportion of heads?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
