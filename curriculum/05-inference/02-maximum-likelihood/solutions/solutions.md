# 05.02 Solutions

1. p̂ = 12/40 = **0.30**. SE = √(0.3 × 0.7/40) ≈ **0.072**.
2. ℓ(λ) = n ln λ − λΣxᵢ. ℓ′ = n/λ − Σxᵢ = 0 ⇒ **λ̂ = 1/x̄**. For commute_minutes: 1/23.52 ≈ **0.0425 per minute** (mean 23.5 min).
3. It's a function of θ for *fixed* data, not a distribution over θ. Integrating it over θ doesn't give 1, and it says
   how probable the data are under each θ, not how probable each θ is.
4. ℓ(p) = n ln p + (Σxᵢ − n) ln(1 − p). Setting ℓ′ = n/p − (Σxᵢ − n)/(1 − p) = 0 gives **p̂ = n/Σxᵢ = 1/x̄** = 5/15 = **1/3**.
5. ℓ″(λ) = −Σxᵢ/λ². At λ̂ = x̄, I = nx̄/x̄² = n/λ̂. SE = √(λ̂/n) = √(2.437/380) ≈ **0.080**.
6. By invariance, the MLE of g(σ²) = √σ² is √σ̂². Unbiasedness does not survive a non-linear transformation (Lesson
   02.03, exercise 11), so neither √σ̂² nor √s² is unbiased for σ.
7. ```python
   def mle_bernoulli(data):
       return sum(data) / len(data)

   def mle_poisson(data):
       return sum(data) / len(data)

   def mle_normal(data):
       mu = sum(data) / len(data)
       return mu, sum((x - mu) ** 2 for x in data) / len(data)
   ```
8. λ̂ = 2.437. Likelihood-ratio interval: **[2.283, 2.597]**. Wald interval λ̂ ± 1.96 SE: **[2.280, 2.594]**. Nearly the
   same here; they differ more for small n or near a boundary.
9. Maximised log-likelihoods: normal −601.9 and gamma **−589.1**. The gamma fits much better, as it should (that's how the
   data were generated), and it respects the fact that hours can't be negative. With equal parameter counts, the higher
   log-likelihood wins. With different counts you need a penalty (AIC/BIC, Module 06) or cross-validation.
10. NLL(β) = −Σ[yᵢ(β₀ + β₁hᵢ) − log(1 + e^(β₀ + β₁hᵢ))]. BFGS gives **β₀ ≈ −4.96, β₁ ≈ 0.386**, matching statsmodels.
    statsmodels SEs: 0.680 and 0.058; the BFGS `hess_inv` approximation gives 0.663 and 0.056 (close, but only an
    approximation). exp(0.386) ≈ **1.47**: each extra weekly study hour multiplies the odds of scoring ≥ 70 by about 1.47.
11. ℓ(μ) = −Σ(xᵢ − μ)²/(2σ²) + const. ℓ″ = −n/σ², so I = n/σ² and SE = **σ/√n**.
12. Bias of σ̂² with σ² = 4: −0.81 (n = 5), −0.074 (50), −0.008 (500). Skewness: 1.42, 0.42, 0.12. Both the bias and the
    non-normality fade as n grows, as the asymptotic theory promises.
