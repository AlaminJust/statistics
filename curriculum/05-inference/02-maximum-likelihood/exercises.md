# 05.02 Exercises — Maximum likelihood

`make lesson L=05-inference/02-maximum-likelihood`

## ★ Warm-up

1. **(M)** 12 successes in 40 Bernoulli trials. Find the MLE of p and its standard error from the Fisher information.
2. **(M)** Derive the MLE of λ for i.i.d. Exponential(λ) data. Compute it for `commute_minutes`.
3. **(C)** Why is the likelihood not a probability distribution over θ?

## ★★ Core

4. **(M)** Geometric data (the number of trials until the first success) are 3, 1, 4, 2, 5. Write the log-likelihood of p
   and derive the MLE. Compute it.
5. **(M)** Show that the Fisher information for a Poisson sample is n/λ, so SE(λ̂) = √(λ̂/n). Compute it for the goals
   data of worked example 2 (λ̂ = 2.437, n = 380).
6. **(M)** Show that the MLE of σ for normal data is √σ̂² (invariance), and explain why that's not the same as the
   unbiased estimator of σ.
7. **(P)** Implement `mle_bernoulli(data)`, `mle_poisson(data)` and `mle_normal(data)` (returning (μ̂, σ̂²)) in
   `statkit/inference.py`.
8. **(P)** Plot the Poisson log-likelihood for the goals data (simulate them as in `examples.py`) over λ ∈ [2, 3], and
   mark λ̂ and the interval where ℓ is within 1.92 of its maximum. Compare that interval with λ̂ ± 1.96 SE. (Within
   1.92 of the max is the *likelihood-ratio* 95% interval.)
9. **(P)** Fit a normal and a gamma model to `study_hours` by MLE. Compare their maximised log-likelihoods. Which fits
   better? Why is it fair to compare these two models directly (both have 2 parameters)?

## ★★★ Challenge

10. **(P)** **Logistic regression by MLE.** Let y = 1 if exam_score ≥ 70. Model P(y = 1) = σ(β₀ + β₁ · study_hours)
    (Lesson 01.02). Write the NLL, minimise it with `scipy.optimize.minimize` (method BFGS), and get SEs from the
    inverse Hessian (`result.hess_inv`). Compare with `statsmodels.api.Logit`. Interpret β₁ as an odds ratio, exp(β₁).
11. **(M)** For i.i.d. N(μ, σ²) with σ known, compute the Fisher information for μ and show that the SE of μ̂ is σ/√n.
12. **(P)** Show by simulation that the MLE σ̂² is biased for n = 5 but that its bias disappears as n grows, while its
    sampling distribution becomes normal (check skewness for n = 5, 50, 500).

Solutions: [solutions/solutions.md](solutions/solutions.md)
