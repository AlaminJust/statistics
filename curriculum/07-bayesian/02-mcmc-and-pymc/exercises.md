# 07.02 Exercises — MCMC and PyMC

`make lesson L=07-bayesian/02-mcmc-and-pymc`

## ★ Warm-up

1. **(C)** Why does the unknown normalising constant p(data) not matter for Metropolis?
2. **(M)** A chain of 10,000 draws for θ has posterior sd 2.0 and ESS 400. What's the Monte Carlo standard error of the
   posterior-mean estimate? What would it be if the draws were independent?
3. **(C)** You see R̂ = 1.3 for one parameter. What does that mean, and what do you do?

## ★★ Core

4. **(P)** Implement `metropolis(log_post, x0, steps, proposal_sd, rng)` in `statkit/bayes.py`, returning
   (draws as an array of shape (steps, dim), acceptance rate). Use it on the Beta(8, 4) coin posterior and check the mean.
5. **(P)** Implement `effective_sample_size(chain)` (the initial-positive-sequence estimator from `examples.py`) and
   `rhat(chains)` (classic Gelman–Rubin for an array of shape (chains, draws)).
6. **(P)** Run your Metropolis sampler on the coin posterior with proposal sds 0.01, 0.05, 0.15, 0.5, 1 and 2. Plot ESS
   per 1,000 draws against the acceptance rate. Where's the sweet spot?
7. **(P)** In PyMC, fit a Poisson model for the goals data (simulate 380 matches with rate 2.6) with a Gamma(2, 1) prior
   on the rate. Compare the posterior mean and 95% interval with the exact Gamma–Poisson answer from Lesson 07.01.
8. **(P)** In PyMC, fit `exam_score ~ study_hours + sleep_hours + tutored` as a Bayesian linear regression with Normal(0, 10)
   priors on the slopes, Normal(50, 30) on the intercept and HalfNormal(20) on σ. Compare the posterior means and 95%
   intervals with the OLS estimates and CIs from Lesson 06.02.

## ★★★ Challenge

9. **(M)** Show that the Metropolis acceptance rule with a symmetric proposal satisfies detailed balance:
   π(θ) q(θ* | θ) α(θ → θ*) = π(θ*) q(θ | θ*) α(θ* → θ).
10. **(P)** **A bimodal posterior.** Run Metropolis on the density 0.5 N(−4, 1) + 0.5 N(4, 1) with proposal sd 0.5 and four
    chains started at −4, −4, 4 and 4. What do the trace plots and R̂ show? Now start all four chains at −4. What
    happens to R̂, and why is that dangerous?
11. **(P)** Write a Gibbs sampler for the normal model with unknown μ and σ², using the conditionals
    μ | σ², y ~ N(ȳ, σ²/n) (flat prior on μ) and σ² | μ, y ~ Inv-Gamma(n/2, Σ(yᵢ − μ)²/2). Compare with the Metropolis
    results on exam_score.

Solutions: [solutions/solutions.md](solutions/solutions.md)
