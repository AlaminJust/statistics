# 07.02 — MCMC and PyMC

**Time:** about 5 hours · **Prerequisites:** 07.01, 04.04 (LLN), 01.04 (Monte Carlo integration)

## Why this matters

Conjugate priors give exact posteriors only for simple models. For almost anything realistic (logistic regression,
hierarchical models, a neural network with priors) the posterior has no formula, and the normalising integral can't be
computed. **Markov chain Monte Carlo** (MCMC) solves this by *drawing samples* from the posterior using only
likelihood × prior. With enough samples, any summary (mean, interval, probability) is just an average. This lesson
builds a sampler from scratch, teaches the diagnostics that tell you whether to trust it, and then hands the work to
**PyMC**, a probabilistic programming library whose NUTS sampler is the modern standard.

## Learning objectives

- Estimate expectations by Monte Carlo averages of posterior draws.
- Implement the Metropolis algorithm and explain why it targets the posterior.
- Tune the proposal step size, and explain burn-in, autocorrelation and effective sample size.
- Diagnose convergence with trace plots and R-hat across multiple chains.
- Write models in PyMC, read `az.summary`, and check for divergences.

## Before you start

You can compute likelihood × prior for any value of θ, but not the integral that normalises it. How could you still
get samples from the normalised posterior?

---

## 1. Monte Carlo: averages replace integrals

If θ⁽¹⁾, …, θ⁽ˢ⁾ are draws from the posterior, then for any function g,

$$E[g(\theta) \mid \text{data}] \approx \frac{1}{S}\sum_{s=1}^S g(\theta^{(s)}).$$

Means, variances, quantiles (credible intervals) and probabilities (the share of draws with θ > 0.5) all come from the
same draws. **Worked example 1:** 100,000 draws from Beta(8, 4) give E[p²] ≈ 0.4614; the exact value is 0.4615. The
error shrinks like 1/√(effective number of draws), by the LLN and CLT (Lesson 04.04).

## 2. The Metropolis algorithm

To sample from a density π(θ) known only up to a constant:

1. Start somewhere, θ₀.
2. Propose a nearby value: θ* = θ + noise, with noise ~ N(0, step²).
3. Accept with probability min(1, π(θ*)/π(θ)). The unknown normalising constant **cancels** in the ratio.
4. If accepted, move to θ*; otherwise stay at θ (and record θ again). Repeat.

Uphill moves are always accepted; downhill moves sometimes. Over time the chain spends time in each region in
proportion to its posterior probability. The reason is *detailed balance*: the probability flow from θ to θ* equals the
flow back, so π is the chain's stationary distribution. Work on the log scale to avoid underflow: accept if
log u < log π(θ*) − log π(θ).

```python
def metropolis(log_post, x0, steps, proposal_sd, rng):
    x, lp = np.atleast_1d(x0), log_post(x0)
    for t in range(steps):
        proposal = x + rng.normal(0, proposal_sd, x.size)
        lp_new = log_post(proposal)
        if np.log(rng.random()) < lp_new - lp:
            x, lp = proposal, lp_new
        chain[t] = x
```

**Worked example 2: the coin again** (7 heads in 10, flat prior), 20,000 steps, the first 2,000 discarded as
**warm-up** (burn-in). Acceptance rate 0.68; posterior mean **0.668** (exact 0.667); 95% interval **[0.391, 0.888]**
(exact [0.390, 0.891]). The sampler recovers the exact posterior without knowing it.

## 3. Autocorrelation and effective sample size

Consecutive MCMC draws are **correlated**: each one starts from the previous. 18,000 correlated draws carry less
information than 18,000 independent ones. The **effective sample size** (ESS) is the number of independent draws they
are worth:

$$\text{ESS} = \frac{S}{1 + 2\sum_{k \ge 1}\rho_k},$$

where ρₖ is the autocorrelation at lag k. Here **ESS ≈ 2,860** out of 18,000 draws (`trace_and_acf.png`). Monte Carlo
errors are governed by the ESS, not by S.

**Worked example 3: tuning the step size.**

| Proposal sd | Acceptance rate | ESS (of 18,000) | Posterior mean |
| --- | --- | --- | --- |
| 0.01 (too small) | 0.97 | **25** | 0.624: still crawling, badly wrong |
| 0.15 (about right) | 0.68 | **2,524** | 0.664 |
| 2.0 (too large) | 0.08 | 981 | 0.665 |

Tiny steps are almost always accepted but barely move; huge steps are almost always rejected. Aim for an acceptance
rate around 0.2–0.5 in several dimensions (0.44 in one). With several parameters on different scales, give each its own
step size, or better, use a sampler that adapts automatically.

## 4. Convergence diagnostics

You can never *prove* a chain has converged, but you can catch most failures.

1. **Run several chains from scattered starting points** (at least 4).
2. **Trace plots:** after warm-up, every chain should look like a fuzzy caterpillar over the same range.
3. **R-hat** compares the variance *between* chains with the variance *within* them:
   R̂ = √((n − 1)/n · W + B/n) / √W. If all chains explore the same distribution, R̂ ≈ 1. **Require R̂ < 1.01.**
4. **ESS:** at least about 400 per parameter (bulk and tail) for reliable intervals.

**Worked example 4: mean and sd of exam_score, two parameters, four chains** started at μ = 30, 90, 60 and 50, with
step sizes 1.5 for μ and 0.08 for log σ:

| Parameter | Posterior mean | 95% interval | R-hat |
| --- | --- | --- | --- |
| μ | 63.94 | [62.00, 65.85] | **1.001** |
| σ | 13.88 | [12.60, 15.31] | **1.000** |

Using only the first 100 draws, before the chains have met, gives R̂ = **1.50**: the diagnostic correctly reports
non-convergence. (Sampling log σ instead of σ keeps the chain away from impossible negative values. That's the same
reparameterisation trick as in Lesson 05.02.)

## 5. PyMC: describe the model, let NUTS sample

Metropolis is slow in many dimensions. **Hamiltonian Monte Carlo** uses the gradient of the log posterior to make
long, well-directed moves, and **NUTS** (No-U-Turn Sampler) tunes it automatically. PyMC computes the gradients for
you.

**Worked example 5:**

```python
with pm.Model() as normal_model:
    mu = pm.Normal("mu", 60, 20)                    # prior
    sigma = pm.LogNormal("sigma", np.log(10), 1)    # prior on a positive parameter
    pm.Normal("score", mu, sigma, observed=y)       # likelihood
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=7)
az.summary(idata, ci_prob=0.95)
```

| | mean | sd | 95% interval | ESS bulk | R-hat |
| --- | --- | --- | --- | --- | --- |
| mu | 63.93 | 0.99 | [62, 66] | 3,743 | 1.00 |
| sigma | 13.85 | 0.70 | [13, 15] | 3,811 | 1.00 |

NUTS's 4,000 draws are worth about 3,700 independent ones, far more efficient than Metropolis's 974 per 8,000.

**Worked example 6: Bayesian logistic regression**, P(score ≥ 70) on study hours with weakly informative priors:
b₁ = **0.388** [0.28, 0.50], odds ratio per hour **1.48** [1.32, 1.66], R̂ = 1.01, **0 divergences**. The numbers are
close to the MLE of Lesson 06.04 (0.386): with vague priors and decent data, Bayesian and frequentist answers agree.

### Divergences

NUTS reports **divergent transitions** when the posterior has regions too sharply curved for its step size (typically
in hierarchical models, Lesson 07.03). Even a few divergences mean the results may be biased. Don't ignore them:
reparameterise the model, or raise `target_accept`.

## 6. A workflow checklist

1. Write the model: priors + likelihood.
2. Sample 4 chains.
3. Check R̂ < 1.01, ESS > 400, 0 divergences, and the trace plots.
4. Only then interpret the posterior summaries.
5. Check the model against the data (posterior predictive checks, Lesson 07.04).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| One chain, no diagnostics | 4+ chains; R-hat, ESS, traces |
| Treating S draws as S independent samples | Use the ESS |
| Keeping warm-up draws | Discard them (PyMC does this automatically) |
| Ignoring divergences | Reparameterise or adjust `target_accept` |
| Sampling a positive parameter on its natural scale with Metropolis | Sample its log |

## Summary

- Posterior draws turn every summary into an average.
- Metropolis: propose, accept with min(1, ratio); the normalising constant cancels.
- Correlated draws → ESS < S; tune the step size (acceptance about 0.2–0.5).
- Several chains plus R̂ < 1.01 and ESS > 400 before interpreting anything.
- PyMC + NUTS: write the model, sample, check `az.summary`, check divergences.

## Self-test

1. Why does the unknown normalising constant not matter for Metropolis?
2. A chain has 10,000 draws and ESS 200. What does that mean for the Monte Carlo error of the posterior mean?
3. What does R̂ = 1.3 tell you?
4. Why are tiny proposal steps bad even though nearly all are accepted?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
