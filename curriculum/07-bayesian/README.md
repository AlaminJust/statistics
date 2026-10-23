# Module 07 — Bayesian statistics

**Weeks 20–23 · about 32 hours** · Status: **outline** · Adds `pymc` and `arviz` to `requirements.txt`

Bayesian statistics treats unknown parameters as uncertain quantities described by probability distributions. It gives
direct answers ("there is an 88% probability that version B is better"), handles small data and grouped data
gracefully, and underlies uncertainty in deep learning, variational autoencoders, diffusion models and Bayesian
optimisation.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 07.01 | Bayesian thinking and conjugate priors | Update beliefs with data by hand | prior, likelihood, posterior, Beta–Binomial and Normal–Normal updates, credible intervals, MAP vs posterior mean, posterior predictive, Bayesian A/B testing |
| 07.02 | MCMC and PyMC | Fit models that have no closed form | Monte Carlo integration, Metropolis–Hastings from scratch, Hamiltonian Monte Carlo and NUTS, PyMC models, diagnostics (trace plots, R-hat, effective sample size, divergences) |
| 07.03 | Hierarchical models | Share strength across groups | partial pooling, hyperpriors, the Eight Schools example, centred vs non-centred parameterisation |
| 07.04 | Model checking and comparison | Criticise and compare Bayesian models | prior and posterior predictive checks, LOO-CV, WAIC, sensitivity to the prior |
| 07.05 | Latent variables, EM and variational inference | Understand the statistics inside generative models | mixture models, the EM algorithm, the ELBO, mean-field VI, the link to VAEs, Gaussian processes |

## statkit functions (`statkit/bayes.py`)

`beta_binomial_update(alpha, beta, successes, failures)`, `credible_interval(samples, level)`,
`metropolis(log_posterior, x0, steps, proposal_sd, rng)`, `effective_sample_size(chain)`, `em_gaussian_mixture`.

## Mini-project: Bayesian analysis report

Re-analyse the m05 A/B test the Bayesian way and compare the conclusions. Then fit a hierarchical model to scores
across the three programs in `students.csv` and show partial pooling in action.

## Checkpoint (without notes)

1. Derive the Beta–Binomial posterior.
2. Explain the difference between a 95% credible interval and a 95% confidence interval.
3. What does R-hat > 1.01 tell you, and what do you do about it?
4. Map each term of the ELBO to a part of a VAE's loss.
5. `make test M=bayes` passes.

## Resources

[Statistical Rethinking 2024 — McElreath](https://github.com/rmcelreath/stat_rethinking_2024) ·
[Think Bayes 2](https://allendowney.github.io/ThinkBayes2/) ·
[Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/) ·
[PyMC](https://www.pymc.io/) · [ArviZ](https://arviz-devs.github.io/arviz/)
