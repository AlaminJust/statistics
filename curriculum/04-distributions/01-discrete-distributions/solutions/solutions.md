# 04.01 Solutions

1. X ~ Binomial(20, 0.25). Mean **5**, sd √(20 × 0.25 × 0.75) ≈ **1.94**. P(0) = 0.75²⁰ ≈ **0.0032**.
   P(X ≥ 10) = `binom.sf(9, 20, 0.25)` ≈ **0.0139**.
2. The rate for 30 minutes is 1. P(at least one) = 1 − e⁻¹ ≈ **0.632**.
3. (a) Binomial(50, 0.5); (b) Poisson; (c) geometric; (d) categorical.
4. C(5, 3)/32 = 10/32 = **0.3125**. P(X ≥ 2) for Binomial(8, 0.1) = 1 − 0.9⁸ − 8(0.1)(0.9⁷) ≈ **0.187**.
5. P(0) = e⁻³ ≈ **0.0498**. P(X > 4) = `poisson.sf(4, 3)` ≈ **0.185**. Over 4 months the rate is 12, so
   P(X ≤ 10) = `poisson.cdf(10, 12)` ≈ **0.347**.
6. P(X ≤ 4) = 1 − 0.8⁴ = **0.590**. Mean 1/0.2 = **5** calls.
7. Exact: `binom.sf(3, 500, 0.004)` ≈ **0.1425**. Poisson(λ = 2): `poisson.sf(3, 2)` ≈ **0.1429**. Very close.
8. ```python
   def binomial_pmf(k, n, p):
       return math.comb(n, k) * p ** k * (1 - p) ** (n - k)

   def poisson_pmf(k, lam):
       return lam ** k * math.exp(-lam) / math.factorial(k)

   def geometric_pmf(k, p):
       return (1 - p) ** (k - 1) * p if k >= 1 else 0.0
   ```
10. Σₖ λᵏe^(−λ)/k! = e^(−λ) Σₖ λᵏ/k! = e^(−λ)e^λ = 1. Mean: Σₖ k λᵏe^(−λ)/k! = λ e^(−λ) Σₖ≥₁ λᵏ⁻¹/(k − 1)! = λ e^(−λ) e^λ = **λ**.
11. C(n, k)(λ/n)ᵏ(1 − λ/n)ⁿ⁻ᵏ = [n(n − 1)…(n − k + 1)/nᵏ] × (λᵏ/k!) × (1 − λ/n)ⁿ × (1 − λ/n)⁻ᵏ. As n → ∞ the first
    bracket → 1, (1 − λ/n)ⁿ → e^(−λ) and (1 − λ/n)⁻ᵏ → 1, leaving λᵏe^(−λ)/k!. ∎
12. Mean ≈ **3.0** but variance ≈ **9.2**, three times larger. The total variance is the Poisson part (on average 3) plus
    the variance of the rates themselves, 1.5 × 2² = 6. Customers differ: some buy often, some rarely. A single Poisson
    assumes everyone has the same rate. Purchase counts, hospital visits, insurance claims and word counts all behave
    like this. Mixing a Poisson over gamma-distributed rates gives exactly the **negative binomial** (Module 06).
