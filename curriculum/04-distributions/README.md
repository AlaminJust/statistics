# Module 04 — Distributions

**Weeks 9–11 · about 24 hours**

A handful of distributions describe most data and most models. This module covers each distribution's story and
formulas, the two theorems that make statistics work (the law of large numbers and the central limit theorem), the
sampling distributions behind every interval and test, and information theory, the source of ML loss functions.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 04.01 | [Discrete distributions](01-discrete-distributions/lesson.md) | Bernoulli, binomial, geometric, Poisson: recognise the story | `binomial_pmf`, `poisson_pmf`, `geometric_pmf` |
| 04.02 | [Continuous distributions](02-continuous-distributions/lesson.md) | PDF, CDF, quantile; exponential waits; inverse-CDF sampling | `exponential_cdf`, `sample_inverse_cdf` |
| 04.03 | [The normal and multivariate normal](03-normal-and-multivariate-normal/lesson.md) | z-scores, 1.96, Q–Q plots; E[Y \| X] is linear | `normal_pdf`, `normal_cdf`, `normal_ppf` |
| 04.04 | [LLN and CLT](04-lln-and-clt/lesson.md) | Averages converge, and their errors are ≈ normal with sd σ/√n | — |
| 04.05 | [Sampling distributions](05-sampling-distributions/lesson.md) | Standard errors; χ², t and F | — |
| 04.06 | [Entropy, cross-entropy and KL](06-information-theory/lesson.md) | Cross-entropy loss = negative log-likelihood | `entropy`, `cross_entropy`, `kl_divergence` |

Run `make test M=distributions` as you implement each function.

## Mini-project: distribution explorer and CLT simulator

Brief in [projects/m04-clt-explorer/README.md](../../projects/m04-clt-explorer/README.md).

## Checkpoint (without notes)

1. Which distribution models: emails per hour; whether a user clicks; the time until the next bus?
2. X ~ N(100, 15²). Find P(X > 130) and the 90th percentile.
3. State the CLT and explain why it lets us build confidence intervals for means of non-normal data. When does it fail?
4. Why does (x̄ − μ)/(s/√n) follow a t distribution rather than a normal?
5. Show that binary cross-entropy loss is the negative Bernoulli log-likelihood.
6. `make test M=distributions` passes.

Pass all six → `git tag -a m04-done -m "Module 04 complete"`.

## Resources

[Introduction to Probability](http://probabilitybook.net) chapters 4–7 and 10 ·
[Seeing Theory: probability distributions and the CLT](https://seeing-theory.brown.edu/) ·
[MacKay, Information Theory, Inference and Learning Algorithms](https://www.inference.org.uk/itila/book.html) chapter 2
