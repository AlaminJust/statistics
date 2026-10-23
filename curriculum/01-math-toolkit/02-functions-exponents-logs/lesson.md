# 01.02 — Functions, exponents and logarithms

**Time:** about 3 hours · **Prerequisites:** 01.01

## Why this matters

Three functions appear in almost every statistical model:

- the **exponential** eˣ: in the normal distribution, the Poisson distribution and softmax;
- the **logarithm** ln x: in likelihoods, log-odds, cross-entropy loss and the log-transform of skewed data;
- the **sigmoid** 1 / (1 + e⁻ᶻ): turns any number into a probability (logistic regression, neural networks).

If you are comfortable with these three, formulas like the normal density or the logistic-regression loss stop looking
scary.

## Learning objectives

- Read function notation and recognise linear, quadratic and exponential shapes.
- Apply the rules of exponents and logarithms.
- Explain three reasons statisticians take logs.
- Compute the sigmoid and softmax, and use the log-sum-exp trick to avoid overflow.

## Before you start

Guess: what does a computer give for 0.01 multiplied by itself 200 times? Write your guess, then check in section 5.

---

## 1. Functions

A **function** f takes an input x and returns one output f(x).

| Type | Formula | Shape | Example in statistics |
| --- | --- | --- | --- |
| Linear | f(x) = a + b·x | straight line; b = slope, a = intercept | regression line: score = 40 + 2.2 · hours |
| Quadratic | f(x) = (x − c)² | U-shape, lowest at x = c | squared error |
| Exponential | f(x) = eˣ | grows faster and faster | growth, densities |
| Logarithm | f(x) = ln x | grows slower and slower | log-likelihood, log-transform |

**Worked example 1.** For f(x) = 40 + 2.2x, f(10) = 40 + 22 = 62. The slope 2.2 means: one more unit of x adds 2.2 to
f(x). (In `data/students.csv`, exam score was simulated to rise about 2.2 points per study hour.)

## 2. Exponents

aⁿ means a multiplied by itself n times. The rules follow from that:

| Rule | Example |
| --- | --- |
| aᵐ · aⁿ = aᵐ⁺ⁿ | 2³ · 2² = 2⁵ = 32 |
| (aᵐ)ⁿ = aᵐⁿ | (2³)² = 2⁶ = 64 |
| a⁰ = 1 | 5⁰ = 1 |
| a⁻ⁿ = 1 / aⁿ | 2⁻³ = 1/8 = 0.125 |
| a^(1/2) = √a | 9^(1/2) = 3 |

**Scientific notation**: 1e-6 means 1 × 10⁻⁶ = 0.000001. Python prints tiny probabilities this way.

## 3. The number e and the exponential function

e ≈ 2.71828 is a constant that appears naturally in growth and probability. **exp(x) = eˣ**.

- eˣ is always positive.
- e⁰ = 1, e¹ ≈ 2.718, e⁻¹ ≈ 0.368.
- e⁻ˣ (decay) shrinks towards 0 as x grows.

The normal distribution (Module 04) has the shape e^(−x²/2): largest at x = 0, falling quickly on both sides. That is
the famous bell curve.

## 4. Logarithms: the inverse of exponents

$$\log_b(x) = y \iff b^y = x.$$

"The log asks: *to what power must I raise b to get x?*"

| Expression | Answer | Because |
| --- | --- | --- |
| log₁₀(1000) | 3 | 10³ = 1000 |
| log₂(8) | 3 | 2³ = 8 |
| ln(e) | 1 | e¹ = e (ln means log base e, the "natural log") |
| ln(1) | 0 | e⁰ = 1 |
| ln(0.5) | −0.693 | numbers below 1 have **negative** logs |
| ln(0) or ln(−3) | undefined | eʸ is never 0 or negative |

In statistics and ML, **"log" without a base almost always means ln.** In NumPy: `np.log` is ln, `np.log10`, `np.log2`.

### The rules of logs (memorise these four)

| Rule | Example |
| --- | --- |
| log(a·b) = log a + log b | ln 2 + ln 3 = 0.693 + 1.099 = 1.792 = ln 6 |
| log(a / b) = log a − log b | ln(6/3) = 1.792 − 1.099 = 0.693 = ln 2 |
| log(aᵏ) = k · log a | ln(2¹⁰) = 10 × 0.693 = 6.93 |
| log(eˣ) = x and e^(ln x) = x | they undo each other |

The first rule is the big one: **logs turn multiplication into addition.**

## 5. Three reasons statisticians take logs

### Reason 1: products of many probabilities underflow

The probability of a whole dataset under a model (the *likelihood*, Module 05) is a product of one probability per
observation. With 200 observations each of probability 0.01:

```python
0.01 ** 200          # 0.0   ← the true value is 1e-400, smaller than any float can hold ("underflow")
200 * np.log(0.01)   # -921.03   ← the LOG of the true value, perfectly representable
```

That answers "Before you start". Because log(Π pᵢ) = Σ log pᵢ, we always work with the **log-likelihood**, a sum.
And since log is increasing, the parameter that maximises the log-likelihood also maximises the likelihood.

### Reason 2: skewed data become more symmetric

Incomes, commute times and reaction times have long right tails: a few huge values. Taking logs compresses large
values more than small ones (ln 10 = 2.3, ln 100 = 4.6, ln 1000 = 6.9 — each ×10 adds only 2.3). Module 02 uses this on
`commute_minutes`.

### Reason 3: log-odds

If an event has probability p, its **odds** are p / (1 − p) and its **log-odds** (logit) are ln(p / (1 − p)).

**Worked example 2.** p = 0.8 → odds = 0.8 / 0.2 = 4 ("4 to 1") → log-odds = ln 4 ≈ 1.386.
p = 0.5 → odds 1 → log-odds 0. Log-odds run from −∞ to +∞, so a linear model can predict them freely. That is the
core idea of logistic regression (Module 06).

## 6. The sigmoid: from any number to a probability

The sigmoid is the inverse of the log-odds:

$$\sigma(z) = \frac{1}{1 + e^{-z}}.$$

| z | e⁻ᶻ | σ(z) |
| --- | --- | --- |
| −2 | 7.389 | 1 / 8.389 = 0.1192 |
| 0 | 1 | 1 / 2 = 0.5 |
| 2 | 0.1353 | 1 / 1.1353 = 0.8808 |
| large positive | ≈ 0 | → 1 |
| large negative | huge | → 0 |

Check the link: σ(1.386) ≈ 0.8, the p we started from in worked example 2. So: **logit turns a probability into a
score; sigmoid turns a score back into a probability.** Every binary classifier's final layer does this.

## 7. Softmax and the log-sum-exp trick

For more than two classes, **softmax** turns a vector of scores z₁, …, z_K into probabilities that sum to 1:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

**Worked example 3.** z = (1, 2, 3). e¹ = 2.718, e² = 7.389, e³ = 20.086; total 30.193.
softmax = (2.718, 7.389, 20.086) / 30.193 = **(0.090, 0.245, 0.665)**. The largest score gets the largest probability,
and they add to 1.

**The overflow problem.** For z = (1000, 1001, 1002), e¹⁰⁰⁰ is too large for a float (it becomes `inf`), so the naive
formula gives `nan`. The fix uses the log rules: subtract the maximum m first.

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}.$$

With m = 1002: 1002 + ln(e⁻² + e⁻¹ + e⁰) = 1002 + ln(1.5032) = **1002.4076**. Every value inside the exp is now ≤ 0,
so nothing overflows. This **log-sum-exp** trick is inside every deep-learning library; you'll implement it in
`statkit`.

Why subtracting m is allowed: e^(zⱼ) = e^m · e^(zⱼ − m), so Σe^(zⱼ) = e^m · Σe^(zⱼ − m), and the log of a product is a
sum of logs.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| log(a + b) = log a + log b | **No.** Only log(a·b) splits. log(a + b) has no simple rule |
| Taking log of 0 or negative values | Add a small constant, or don't log-transform that variable |
| `np.log` is base 10 | `np.log` is **ln**; use `np.log10` for base 10 |
| Multiplying many probabilities | Sum their logs instead |
| Computing softmax directly on large scores | Subtract the max first |

## Summary

- eˣ > 0 always; ln undoes eˣ.
- log(ab) = log a + log b; log(aᵏ) = k log a.
- Logs: avoid underflow, tame skew, give log-odds.
- σ(z) = 1/(1 + e⁻ᶻ) maps scores to probabilities; logit is its inverse.
- Log-sum-exp: subtract the max before exponentiating.

## Self-test

1. Simplify ln(e⁵) and e^(ln 7).
2. Rewrite ln(x³y / z) as a sum and difference of simple logs.
3. Compute the log-odds for p = 0.2.
4. What is σ(0)? What happens to σ(z) as z → ∞?
5. Why does subtracting the maximum not change the softmax probabilities?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
