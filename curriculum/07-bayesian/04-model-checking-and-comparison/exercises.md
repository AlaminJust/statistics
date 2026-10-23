# 07.04 Exercises — Model checking and comparison

`make lesson L=07-bayesian/04-model-checking-and-comparison`

## ★ Warm-up

1. **(C)** Why isn't a converged sampler (R̂ = 1.00, no divergences) evidence of a good model?
2. **(C)** Model A has elpd_loo 12 higher than model B, with an SE of the difference of 10. What do you conclude?
3. **(P)** Simulate from a Normal(0, 100²) prior on a logistic-regression intercept and look at the implied
   probabilities σ(intercept). What's wrong with this "vague" prior? What does Normal(0, 1.5²) imply instead?

## ★★ Core

4. **(P)** For the Bayesian linear regression of exam_score (Lesson 07.02, exercise 8), run posterior predictive checks
   for the mean, the sd, the minimum and the share of scores above 90. Does the normal model fit?
5. **(P)** Fit two models for exam_score in PyMC: (a) study_hours only; (b) study_hours + sleep_hours + tutored. Compare
   them with `az.compare`. Does LOO agree with the F-test of Lesson 06.03?
6. **(P)** Add a third model to the lesson's purchase-count comparison: a zero-inflated Poisson
   (`pm.ZeroInflatedPoisson`). How does it compare with the negative binomial on LOO and on the PPCs? Why?
7. **(P)** Implement `credible_interval(samples, level=0.95)` (equal-tailed, from percentiles) and
   `posterior_predictive_pvalue(replicated_stats, observed_stat)` (share of replicates ≥ observed) in `statkit/bayes.py`.
8. **(P)** **Prior sensitivity for Eight Schools.** Refit with τ ~ HalfCauchy(5), HalfNormal(5), HalfNormal(25) and
   Exponential(1). Tabulate the posterior median of τ and the posterior mean of θ_A. Which conclusions are robust?

## ★★★ Challenge

9. **(P)** **A PPC on a subgroup.** Fit a single normal model to `commute_minutes`. Use posterior predictive checks of the
   skewness and of the share of values below 5 minutes. Then fit an exponential model and compare by LOO.
10. **(M)** Show that for a model with log-likelihood ℓ, the leave-one-out predictive density of observation i is
    p(yᵢ | y₋ᵢ) = 1 / E_post[1/p(yᵢ | θ)], where E_post is over the full-data posterior. (This identity is the basis of
    importance-sampling LOO, which PSIS stabilises.)
11. **(C)** Explain the difference between "the model fits the data" (PPC) and "the model predicts new data best" (LOO).
    Construct an example where a model passes every PPC but still loses on LOO.

Solutions: [solutions/solutions.md](solutions/solutions.md)
