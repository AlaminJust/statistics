# 08.01 — Generalisation and the bias–variance trade-off

**Time:** about 4 hours · **Prerequisites:** 05.01 (bias, variance, MSE), 06.05, 06.06, 04.04 (Hoeffding)

## Why this matters

The goal of machine learning is not to fit the training data but to **predict new data well**. A model that memorises
its training set (zero training error) can be useless in deployment. This lesson makes "generalisation" precise: the
difference between empirical and true risk, the bias–variance decomposition of prediction error, learning curves that
tell you whether more data or a better model would help, the surprising **double descent** of modern
over-parameterised models, and the simplest generalisation bound.

## Learning objectives

- Define true risk and empirical risk, and explain overfitting and underfitting.
- Decompose expected test error into bias², variance and noise, and estimate each by simulation.
- Read training/test error curves and learning curves to diagnose a model.
- Explain double descent and why huge models can still generalise.
- Derive a Hoeffding + union-bound generalisation bound for a finite set of models.

## Before you start

A degree-15 polynomial fits 30 training points with smaller error than a degree-3 polynomial. Which will predict new
points better? And why do neural networks with millions of parameters, far more than their training examples, still
predict well?

---

## 1. True risk vs empirical risk

For a model f and a loss L (squared error, 0/1 error, cross-entropy):

$$R(f) = E_{(x, y) \sim P}\big[L(f(x), y)\big] \quad\text{(true risk)}, \qquad \hat{R}_n(f) = \frac{1}{n}\sum_{i=1}^n L(f(x_i), y_i) \quad\text{(empirical risk)}.$$

Training minimises the empirical risk on the training set (**empirical risk minimisation**). The LLN says R̂ₙ(f) → R(f)
for a *fixed* f. But the trained f was chosen to make R̂ₙ small on *these* points, so the training error is optimistically
biased. A fresh **test set** gives an unbiased estimate of R(f), but only if it's used once (Lesson 08.02).

## 2. Underfitting and overfitting

**Worked example 1.** True curve sin(2x) + 0.5x with noise sd 0.4 (noise variance 0.16, the best achievable MSE); 30
training points; polynomials of degree 0 to 15:

| Degree | Training MSE | Test MSE | Diagnosis |
| --- | --- | --- | --- |
| 0 | 0.299 | 0.359 | underfit (too simple) |
| 1 | 0.270 | 0.335 | underfit |
| 3 | 0.116 | **0.179** | about right |
| 5 | 0.115 | **0.179** | about right |
| 9 | 0.103 | 0.327 | overfitting begins |
| 15 | **0.044** | 0.576 | overfit: memorises the noise |

**Training error always falls as complexity grows; test error is U-shaped** (`generalisation.png`, left). The gap
between them measures overfitting. Answer to the first part of "Before you start": degree 3, by a wide margin.

## 3. The bias–variance decomposition

For squared error at a point x₀, averaging over training sets and the noise in the new y:

$$E\big[(y_0 - \hat{f}(x_0))^2\big] = \underbrace{\big(E[\hat{f}(x_0)] - f(x_0)\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}\big(\hat{f}(x_0)\big)}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}.$$

It's Lesson 05.01's MSE = bias² + variance, applied to *predictions*, plus the noise no model can remove.

**Worked example 2.** Refit on 2,000 independent training sets of 30 points and look at predictions at x₀ = 2.5:

| Degree | Bias² | Variance | Noise | Expected test MSE |
| --- | --- | --- | --- | --- |
| 1 | **0.074** | 0.026 | 0.16 | 0.260 |
| 3 | 0.001 | 0.020 | 0.16 | **0.181** |
| 9 | 0.005 | **4.465** | 0.16 | 4.630 |

Simple models have high bias (they can't follow the curve). Flexible models have high variance (they change wildly
with each training set, especially near the edges of the data). The best model balances the two. Regularisation (Lesson
06.05), ensembles (Lesson 03.06) and more data all reduce variance; richer models and better features reduce bias.

**k-nearest neighbours** (worked example 3, breast-cancer data): k = 1 gives training accuracy 1.000 (pure memorisation).
Test accuracy stays about 0.94–0.95 for k from 1 to 51 and drops to 0.88 at k = 151 (too smooth). The complexity knob
is 1/k. Here the test accuracy is flat over a wide range: the classes are well separated, so many choices work, which
is common in practice.

## 4. Learning curves: more data or a better model?

Plot training and validation performance against training-set size (`learning_curve` in scikit-learn):

| Pattern | Diagnosis | What helps |
| --- | --- | --- |
| both low, close together | **high bias** (underfitting) | a more flexible model, better features; more data won't help much |
| training high, validation much lower | **high variance** (overfitting) | more data, regularisation, a simpler model |
| both high, close | good | — |

**Worked example 4** (kNN with k = 7): the gap shrinks from 0.022 at n = 45 to 0.002 at n = 455, and validation accuracy
rises from 0.907 to 0.970. The curve is still rising slightly, so more data would help a little.

## 5. Double descent: when bigger is better again

Classical theory says test error rises once a model is complex enough to interpolate the training data. Modern deep
networks have far more parameters than training examples and still generalise. **Worked example 5:** random ReLU
features, n = 100 training points, minimum-norm least squares:

| Features p | 10 | 25 | 50 | 95 | **100 (= n)** | 120 | 400 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test MSE | 1.09 | 0.51 | 0.51 | 3.44 | **168.9** | 1.44 | 0.46 | **0.42** |

The error first falls, then **explodes at the interpolation threshold p = n** (exactly one solution fits the data, and it
contorts itself to pass through every noisy point), then **falls again** as p grows. With far more features than points,
there are infinitely many interpolating solutions, and the minimum-norm one is smooth: an *implicit* regularisation.
Here 2,000 features beat the best small model. Gradient descent on large neural networks has a similar implicit
preference for "simple" solutions, which is part of why they generalise (`generalisation.png`, right).

The practical lessons: don't assume more parameters means more overfitting; avoid models sized near the interpolation
threshold; and **measure generalisation on held-out data** rather than counting parameters.

## 6. A generalisation bound

For a single fixed model with loss in [0, 1], Hoeffding's inequality (Lesson 04.04) gives
P(|R̂ₙ − R| > ε) ≤ 2e^(−2nε²). If you choose the best of M models using the same data, the **union bound** says *all* M
estimates are within ε of the truth, with probability at least 1 − δ, when

$$n \ge \frac{\ln(2M/\delta)}{2\varepsilon^2}.$$

| Models compared (M) | 1 | 10 | 1,000 | 10⁶ |
| --- | --- | --- | --- | --- |
| n for ±0.02 at 95% | 4,612 | 7,490 | 13,246 | 21,881 |

Selecting among more models needs more data to trust the winner's score, though only logarithmically more. That's why
reusing a test set for many comparisons slowly erodes its validity (Lesson 08.02). For infinite model classes, the VC
dimension or Rademacher complexity replaces ln M; these bounds are usually too loose to use numerically, but they
explain the shape of the trade-off.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Judging a model by training error | Held-out error estimates true risk |
| "More parameters always means overfitting" | Double descent: measure, don't assume |
| Collecting more data for a high-bias model | Improve the model or features first |
| Tuning many models on one test set | The test set's estimate becomes optimistic |

## Summary

- True risk R(f) vs empirical risk R̂ₙ(f); training error is optimistic.
- Expected test MSE = bias² + variance + noise; complexity trades bias for variance.
- Learning curves diagnose high bias (both curves low) vs high variance (a big gap).
- Double descent: error spikes at p = n, then falls again for very large models.
- n ≥ ln(2M/δ)/(2ε²) to trust the best of M models within ε.

## Self-test

1. Training accuracy 99%, validation accuracy 75%. Diagnosis and two remedies?
2. Which part of the decomposition can no model reduce?
3. Why does test error peak at the interpolation threshold?
4. Why does comparing many models on the same test set make the winner's score optimistic?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
