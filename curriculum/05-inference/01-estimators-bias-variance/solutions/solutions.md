# 05.01 Solutions

1. MSE(A) = 0.25 + 1 = **1.25**; MSE(B) = **1.5**. A, the biased one, is better.
2. E[θ̂] = (μ + 2μ)/3 = μ ✓. Var = (σ² + 4σ²)/9 = **5σ²/9 ≈ 0.556σ²**. X̄ has variance σ²/2 = 0.5σ², so θ̂ is worse:
   with equal-quality observations, equal weights are best.
3. Low bias, low variance: darts clustered on the bullseye. Low bias, high variance: scattered, centred on the bullseye.
   High bias, low variance: tight cluster in the wrong place. High bias, high variance: scattered and off-centre.
4. E[X̄] = p; MSE = Var = **p(1 − p)/n**. For p = 0.5 and n = 100: 0.0025 (SE 0.05).
5. E[p̃] = (np + 2)/(n + 4), so bias = (2 − 4p)/(n + 4). It's zero only at p = 0.5 and pulls estimates towards 0.5. With
   n = 10, p̃ has lower MSE than p̂ for **p between about 0.2 and 0.8**: shrinkage helps unless the truth is extreme.
6. MSE (seed 0), mean / median / midrange: normal 0.100 / 0.138 / 0.185, so **the mean wins**; Laplace 0.199 / 0.145 / 0.821,
   so **the median wins**. Heavier tails favour more robust estimators.
7. Bias of s: −0.41 (n = 2), −0.12 (5), −0.057 (10), −0.011 (50). MSE: 1.60, 0.48, 0.22, 0.04. Both bias and variance
   vanish, so s is **consistent**, even though it's biased for every finite n.
8. ```python
   def standard_error(values):
       return std(values) / math.sqrt(len(values))
   ```
   Arts 64.08 (SE 1.76), Commerce 65.02 (SE 2.16), Science 63.11 (SE 1.35). The differences between programs are about
   the size of one SE: likely noise (tested in Lesson 05.06).
9. F(x) = (x/θ)ⁿ, so f(x) = n xⁿ⁻¹/θⁿ on [0, θ]. E[max] = ∫₀^θ x · n xⁿ⁻¹/θⁿ dx = n/θⁿ · θⁿ⁺¹/(n + 1) = **θ n/(n + 1)**. ∎
10. Bias = (c − 1)μ, Var = c²σ²/n, so MSE(c) = (1 − c)²μ² + c²σ²/n. dMSE/dc = −2(1 − c)μ² + 2cσ²/n = 0 ⇒
    **c = μ²/(μ² + σ²/n)**.
11. Average total squared error: raw ≈ **10.0**, James–Stein ≈ **6.0**, a 40% reduction, from a "worse" biased estimator
    that borrows strength across the 10 means. This surprising result led to empirical Bayes and hierarchical models
    (Module 07).
