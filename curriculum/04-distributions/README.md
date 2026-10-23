# Module 04 — Distributions

**Weeks 9–11 · about 24 hours** · Status: **outline**

A handful of probability distributions describe most data and most models: Bernoulli and categorical for
classification, the normal for errors and averages, the Poisson for counts, the exponential for waiting times. This
module covers each distribution's story, formulas and uses. It ends with the two results that make statistics work
(the law of large numbers and the central limit theorem) and with information theory, the source of ML loss functions.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 04.01 | Discrete distributions | Choose and use the right discrete model | Bernoulli, binomial, geometric, Poisson, categorical; PMF, mean, variance; the Poisson as a limit of the binomial; `scipy.stats` |
| 04.02 | Continuous distributions | Work with densities and CDFs | uniform, exponential, normal, gamma, beta; PDF vs CDF; quantile functions; the memoryless property; inverse-CDF sampling |
| 04.03 | The normal and multivariate normal | Use the most important distribution in 1 and many dimensions | standardising, z-tables, sums of normals, the 2-D normal, covariance matrices, contour ellipses, marginals and conditionals |
| 04.04 | Law of large numbers and the CLT | Explain why averages are predictable | LLN, CLT, how fast the CLT kicks in for skewed data, when it fails (Cauchy), Markov, Chebyshev and Hoeffding bounds |
| 04.05 | Sampling distributions | Know the distribution of x̄, s² and their ratios | standard error σ/√n, the t, χ² and F distributions and where each comes from |
| 04.06 | Entropy, cross-entropy and KL divergence | Read ML loss functions as information quantities | entropy, cross-entropy loss, KL divergence and its asymmetry, mutual information, the link to maximum likelihood |

## statkit functions (`statkit/distributions.py`)

`binomial_pmf`, `poisson_pmf`, `normal_pdf`, `normal_cdf` (via `math.erf`), `exponential_cdf`,
`sample_inverse_cdf(inverse_cdf, n, rng)`, `entropy`, `cross_entropy`, `kl_divergence`.

## Mini-project: distribution explorer and CLT simulator

A script that, for a chosen distribution and sample size, simulates 10,000 sample means and overlays the normal
approximation the CLT predicts. Include a figure showing how the skewness of the sample mean falls as n grows for an
exponential population, and a case where the CLT fails (Cauchy).

## Checkpoint (without notes)

1. Which distribution models: the number of emails per hour; whether a user clicks; the time until the next bus?
2. X ~ N(100, 15²). Find P(X > 130) and the 90th percentile.
3. State the CLT and explain why it lets us build confidence intervals for means of non-normal data.
4. Show that cross-entropy loss for a binary classifier equals the negative Bernoulli log-likelihood.
5. `make test M=distributions` passes.

## Resources

[Introduction to Probability](http://probabilitybook.net) chapters 4–7 and 10 ·
[Seeing Theory: probability distributions](https://seeing-theory.brown.edu/) ·
[MacKay, Information Theory, Inference and Learning Algorithms](https://www.inference.org.uk/itila/book.html) chapter 2
