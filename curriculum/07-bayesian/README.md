# Module 07 — Bayesian statistics

**Weeks 20–23 · about 30 hours** · Uses `pymc` and `arviz` (installed by `make setup` from `requirements.txt`)

Bayesian statistics treats unknown parameters as uncertain quantities described by probability distributions. It gives
direct answers ("there's a 99.8% probability that version B is better"), handles small and grouped data gracefully, and
underlies uncertainty in deep learning, variational autoencoders and Bayesian optimisation.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 07.01 | [Bayesian thinking and conjugate priors](01-bayesian-thinking-conjugate-priors/lesson.md) | posterior ∝ likelihood × prior; conjugate updates by hand | `beta_binomial_update`, `normal_normal_update` |
| 07.02 | [MCMC and PyMC](02-mcmc-and-pymc/lesson.md) | Metropolis from scratch; R-hat, ESS, divergences; NUTS | `metropolis`, `effective_sample_size`, `rhat` |
| 07.03 | [Hierarchical models](03-hierarchical-models/lesson.md) | Partial pooling; the Eight Schools; non-centred parameterisation | — |
| 07.04 | [Model checking and comparison](04-model-checking-and-comparison/lesson.md) | Prior/posterior predictive checks; PSIS-LOO | `credible_interval`, `posterior_predictive_pvalue` |
| 07.05 | [Latent variables, EM and VI](05-latent-variables-em-vi/lesson.md) | EM for mixtures; the ELBO; ADVI's limits; Gaussian processes | `em_gaussian_mixture` |

Run `make test M=bayes` as you implement each function.

PyMC compiles models the first time they run, so each `examples.py` takes up to a minute. ArviZ 1.x reports equal-tailed
intervals with `az.summary(idata, ci_prob=0.95)`.

## Mini-project: a Bayesian analysis report

Brief in [projects/m07-bayesian-report/README.md](../../projects/m07-bayesian-report/README.md).

## Checkpoint (without notes)

1. Derive the Beta–Binomial posterior.
2. Explain the difference between a 95% credible interval and a 95% confidence interval.
3. What do R̂ > 1.01 and divergences tell you, and what do you do about each?
4. Explain partial pooling in the Eight Schools, and why school A's estimate shrinks so much.
5. Map each term of the ELBO to a part of a VAE's loss.
6. `make test M=bayes` passes.

Pass all six → `git tag -a m07-done -m "Module 07 complete"`.

## Resources

[Statistical Rethinking 2024 — McElreath](https://github.com/rmcelreath/stat_rethinking_2024) ·
[Think Bayes 2](https://allendowney.github.io/ThinkBayes2/) ·
[Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/) ·
[Bayesian Data Analysis, 3rd ed.](http://www.stat.columbia.edu/~gelman/book/) ·
[PyMC](https://www.pymc.io/) · [ArviZ](https://arviz-devs.github.io/arviz/)
