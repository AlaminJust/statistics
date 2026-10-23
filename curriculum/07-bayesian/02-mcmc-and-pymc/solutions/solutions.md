# 07.02 Solutions

1. Metropolis uses only the *ratio* π(θ*)/π(θ). Any constant factor in π appears in both the numerator and the
   denominator and cancels.
2. MCSE ≈ sd/√ESS = 2/√400 = **0.10**. With 10,000 independent draws it would be 2/100 = 0.02.
3. The chains disagree: they haven't converged to the same distribution (or the posterior has separate modes the chains
   are stuck in). Don't interpret the results. Run longer, check the traces, reparameterise, or use a better sampler.
4. ```python
   def metropolis(log_post, x0, steps, proposal_sd, rng):
       x = np.atleast_1d(np.asarray(x0, float))
       lp = log_post(x)
       draws = np.empty((steps, x.size))
       accepted = 0
       for t in range(steps):
           proposal = x + rng.normal(0, proposal_sd, x.size)
           lp_new = log_post(proposal)
           if np.log(rng.random()) < lp_new - lp:
               x, lp = proposal, lp_new
               accepted += 1
           draws[t] = x
       return draws, accepted / steps
   ```
5. See `../solutions/statkit_bayes_reference.py`.
6. ESS per 1,000 draws (seed 3): sd 0.01 → 1.6 (acceptance 0.98); 0.05 → 30; 0.15 → 153 (0.68); **0.5 → 165 (0.31)**;
   1 → 94 (0.16); 2 → 47 (0.08). The best efficiency is at an acceptance rate of about 0.3–0.7, and very high or very
   low acceptance is wasteful.
7. PyMC: mean **2.433**, 95% [2.271, 2.594]. Exact Gamma(2 + Σg, rate 1 + 380): mean 2.436, [2.281, 2.595]. They agree
   within Monte Carlo error.
8. Posterior means (95% intervals): intercept 24 [16, 33], study 2.04 [1.8, 2.3], sleep 2.47 [1.3, 3.6], tutored 6.64
   [4.0, 9.3], σ 8.71. OLS: study 2.05 [1.80, 2.29], sleep 2.57 [1.35, 3.78], tutored 6.76 [4.12, 9.40]. Nearly
   identical: the priors are weak relative to 200 observations. The priors pull the intercept and sleep slightly, which
   are the least precisely estimated.
9. For θ* ≠ θ with symmetric q: if π(θ*) ≥ π(θ), then α(θ → θ*) = 1 and α(θ* → θ) = π(θ)/π(θ*), so both sides equal
   π(θ)q. If π(θ*) < π(θ), the roles swap. Either way the flows match, so π is stationary. ∎
10. Starting chains at both modes: each chain stays in its own mode (the means are −3.9, −4.1, 4.1 and 4.1), and R̂ ≈ **4.6**
    correctly flags the problem. Starting all four at −4: R̂ ≈ **1.006**, "converged", but every chain has missed the
    mode at +4 entirely, and the posterior mean is reported as about −4 instead of 0. R̂ can only detect disagreement
    between chains. Use dispersed starting points and think about whether the posterior could be multimodal.
11. Gibbs (5,000 draws after warm-up): μ mean 63.96, 95% [62.06, 65.83]; σ mean 13.89. This matches Metropolis and PyMC.
    Gibbs needs no tuning, but it requires you to know every full conditional distribution.
