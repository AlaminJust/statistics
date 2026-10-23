# 07.04 — Model checking and comparison

**Time:** about 4 hours · **Prerequisites:** 07.01–07.03, 04.01 (overdispersion)

## Why this matters

A posterior is only as good as the model behind it. A model can converge perfectly (R̂ = 1.00, no divergences) and
still be badly wrong about the data. The **Bayesian workflow** builds checks into every step: simulate from the priors
to see whether they make sense; simulate from the fitted model to see whether it can reproduce the data; compare
candidate models by how well they predict *new* data; and check how much the conclusions depend on the priors. These
habits carry over directly to ML model evaluation.

## Learning objectives

- Run prior predictive checks and choose sensible weakly informative priors.
- Run posterior predictive checks with test statistics, and interpret posterior predictive p-values.
- Compare models with PSIS-LOO (expected log predictive density) and read `az.compare`.
- Carry out a prior sensitivity analysis.

## Before you start

A Poisson model of customer purchases converges perfectly, with tight intervals. Would that convince you it's a good
model? What would you check?

---

## 1. The data

400 customers' monthly purchase counts (simulated, as customers really behave: each one has their own rate). Mean
3.23, **variance 9.87**, 19% zeros. A Poisson model assumes variance = mean.

## 2. Prior predictive checks

Before seeing the data, **simulate data from the priors** and ask whether they look like plausible data.

| Prior on the rate λ | Simulated monthly purchases: median | 99th percentile |
| --- | --- | --- |
| Exponential(0.001), "uninformative" | **712** | **4,782** |
| Gamma(2, 0.5), weakly informative | 3 | 15 |

The "uninformative" prior says a typical customer buys 700 times a month. It isn't neutral, it's absurd. With plenty of
data the posterior may survive it, but with little data, or in complex models where priors interact, absurd priors cause
bad inferences and sampling problems. Prefer **weakly informative priors**: wide enough to allow every plausible
value, narrow enough to exclude impossible ones.

```python
with model:
    prior = pm.sample_prior_predictive(1000)
```

## 3. Posterior predictive checks (PPCs)

If the model is good, **data simulated from the fitted model should look like the real data**. Pick test statistics
that matter for your question, compute them on each replicated dataset, and see where the observed value falls.

| Statistic | Observed | Poisson: replicated 95% range | NegBin: replicated 95% range |
| --- | --- | --- | --- |
| variance | 9.87 | [2.75, 3.78] ✗ | [7.59, 14.92] ✓ |
| share of zeros | 0.190 | [0.020, 0.062] ✗ | [0.138, 0.237] ✓ |
| maximum | 16 | [8, 12] ✗ | [14, 30] ✓ |

**Posterior predictive p-values** (the share of replicates at least as extreme as the data): Poisson 0.000 for all
three; negative binomial 0.64, 0.44 and 0.92. The Poisson model can't produce data this variable, this zero-heavy, or
with values this large, even though it converged perfectly. That answers "Before you start": convergence says the
*sampler* worked, not that the *model* is right (`posterior_predictive_checks.png`).

Choose statistics that the model doesn't fit directly. The Poisson fits the mean exactly, so checking the mean would
reveal nothing.

```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
reps = idata.posterior_predictive["y"].to_numpy().reshape(-1, n)
```

## 4. Comparing models: expected predictive accuracy

The question for comparison: **which model predicts new data better?** The measure is the **expected log pointwise
predictive density** (elpd), the average log-probability the model assigns to a new observation (higher is better).
It is estimated by leave-one-out cross-validation; **PSIS-LOO** approximates it from a single fit, without refitting n
times.

```python
pm.compute_log_likelihood(idata)
az.loo(idata)
az.compare({"poisson": idata_p, "negative binomial": idata_nb})
```

| Model | elpd_loo | SE | elpd difference | SE of the difference | weight |
| --- | --- | --- | --- | --- | --- |
| negative binomial | −922 | 17 | 0 | — | 1.0 |
| Poisson | −1,099 | 35 | **−180** | 24 | 0.0 |

The negative binomial is better by about 180 log-units, with an SE of 24: overwhelming evidence (a difference of more
than about 4 SEs is decisive). `p_loo`, the effective number of parameters, is 1.8 for the NB (it has 2) but 3.1 for the
1-parameter Poisson: **p_loo larger than the actual number of parameters is itself a sign of misspecification.**

The **Pareto k diagnostic** checks the PSIS approximation: all 400 values are "good" (k < 0.7). Large k values flag
influential observations for which the approximation is unreliable. Then refit without them (exact LOO), or use
K-fold CV.

LOO and WAIC are the Bayesian counterparts of AIC (Lesson 06.02), and they estimate the same thing as the
cross-validation of Module 08. Prefer them over comparing marginal likelihoods (Bayes factors), which are very sensitive
to the priors.

## 5. Prior sensitivity

Refit with several reasonable priors and see whether the conclusions change.

| Prior on λ | Prior mean | Posterior mean, n = 5 | Posterior mean, n = 400 |
| --- | --- | --- | --- |
| Gamma(2, 0.5) | 4 | 3.45 | 3.231 |
| Gamma(1, 0.1) | 10 | 3.53 | 3.232 |
| Gamma(20, 10) | 2 | **2.47** | 3.200 |

With 5 customers, a confident prior (Gamma(20, 10), worth 10 pseudo-observations) moves the answer noticeably. With 400
the priors don't matter. **Report sensitivity whenever data are scarce**, as in Lesson 07.03's Eight Schools.

## 6. The Bayesian workflow

```text
choose model and priors  ──►  prior predictive check  ──►  fit (MCMC)
          ▲                                                    │
          │                                       diagnostics: R-hat, ESS, divergences
          │                                                    │
   revise the model  ◄── posterior predictive checks ◄─────────┘
          │
   compare candidates (LOO) ──► prior sensitivity ──► report
```

It's iterative: a failed check tells you how to improve the model (here: add overdispersion).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Treating R̂ = 1 as "the model is correct" | Convergence ≠ fit; run PPCs |
| "Uninformative" priors by default | Weakly informative priors, checked by simulation |
| Checking only the statistics the model fits by construction (the mean) | Check variance, zeros, extremes, tails, subgroups |
| Comparing models by in-sample fit | LOO or WAIC (or held-out data) |
| Ignoring high Pareto k values | Investigate those points; use exact LOO or K-fold |

## Summary

- Prior predictive: simulate data from the priors; reject absurd priors.
- Posterior predictive: simulate replicated data and compare test statistics; p near 0 or 1 signals misfit.
- PSIS-LOO estimates out-of-sample elpd; `az.compare` ranks models; judge differences against their SE.
- Prior sensitivity: matters with little data; report it.
- The workflow is iterative: checks guide model revision.

## Self-test

1. Why isn't a converged sampler evidence of a good model?
2. What does a posterior predictive p-value of 0.001 for the share of zeros mean?
3. Model A has elpd 12 higher than model B, with SE 10. What would you conclude?
4. Why is Exponential(0.001) a bad "default" prior for a purchase rate?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
