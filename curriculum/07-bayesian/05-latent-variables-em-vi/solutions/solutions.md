# 07.05 Solutions

1. At x = 2: 0.7 × φ(2) = 0.0378 and 0.3 × φ(−2) = 0.0162, so r₂ = **0.30**. At x = 3: r₂ = **0.96**.
2. k-means assigns each point wholly to one cluster. EM gives probabilities (soft assignments). The difference matters for
   points in overlapping regions (a student at 13 hours is 68/32, not 100/0), and when clusters differ in size or
   spread, which k-means assumes away.
3. log p(x) = ELBO(q) + KL(q ‖ posterior). The left side doesn't depend on q, so raising the ELBO must lower the KL. The
   best q is as close as possible (within its family) to the true posterior.
4. ```python
   def em_gaussian_mixture(x, k=2, iterations=500, rng=None, n_init=5):
       rng = rng or np.random.default_rng()
       x = np.asarray(x, float)
       best = None
       for _ in range(n_init):
           w = np.full(k, 1 / k)
           m = rng.choice(x, k, replace=False)
           s = np.full(k, x.std())
           for _ in range(iterations):
               dens = np.vstack([w[j] * stats.norm.pdf(x, m[j], s[j]) for j in range(k)])
               r = dens / dens.sum(axis=0)
               nk = r.sum(axis=1)
               w, m = nk / len(x), (r @ x) / nk
               s = np.sqrt((r * (x - m[:, None]) ** 2).sum(axis=1) / nk)
           ll = np.log(dens.sum(axis=0)).sum()
           if best is None or ll > best[3]:
               best = (w, m, s, ll)
       return best
   ```
5. BIC: k = 1 → 3099.3, **k = 2 → 2875.3**, k = 3 → 2892.7, k = 4 → 2910.4. BIC correctly chooses 2.
6. EM finds components with weights 0.33 and 0.67, means 45.4 and 12.8, sds 20.5 and 8.0, and BIC even *prefers* 2
   components over 1 (1708.7 vs 1783.8). But commute was generated from a **single exponential**: there are no two
   groups. A mixture of normals is just a flexible way to fit a skewed shape. A good mixture fit is not evidence of real
   subgroups; you need theory or other variables.
7. Full-rank ADVI: b₁ sd 0.054 and corr(b₀, b₁) = **−0.96**, close to NUTS (sd 0.055). Mean-field on standardised hours:
   b₁ mean 2.326, sd 0.279 against NUTS's 2.337 and 0.278. Centring the predictor removes most of the b₀–b₁ correlation,
   so the independence assumption of mean-field VI is nearly true. Centring and scaling helps every sampler and optimiser.
8. The GP learns a long length-scale (13.8 hours), so the function is nearly linear. Predictions: 5 h → 53.4 (linear
   53.3), 15 h → 74.6 (73.8), 30 h → **89.7** (linear 104.5). Beyond the data the GP reverts towards the mean and widens
   its band (±21.7), while the line keeps rising past the maximum possible score of 100.
9. ∂/∂μ_k Σᵢ rᵢₖ[−(xᵢ − μ_k)²/(2σ_k²)] = Σᵢ rᵢₖ(xᵢ − μ_k)/σ_k² = 0 ⇒ **μ_k = Σᵢ rᵢₖxᵢ / Σᵢ rᵢₖ**.
10. E_q[log p(θ)] = −½ log 2π − (m² + s²)/2; E_q[log p(y | θ)] = −(n/2) log 2π − ½ Σ[(yᵢ − m)² + s²]; the entropy term
    is −E_q[log q] = ½ log(2πe s²). So ELBO(m, s) = const − (m² + s²)/2 − ½ Σ(yᵢ − m)² − ns²/2 + log s.
    ∂/∂m: −m + Σ(yᵢ − m) = 0 ⇒ m = Σyᵢ/(n + 1). ∂/∂s: −(n + 1)s + 1/s = 0 ⇒ s² = 1/(n + 1). That's exactly the posterior
    N(Σy/(n + 1), 1/(n + 1)). ∎
11. EM recovers the means, variances and ρ ≈ 0.7 with smaller error than complete-case analysis, because the 30% of
    rows with only the first coordinate still carry information about its mean and variance, and, through ρ, about the
    second coordinate. Here the values are missing completely at random, so dropping rows is unbiased, just wasteful. If
    missingness depended on the observed coordinate, complete-case analysis would also be **biased**, while EM (which
    assumes "missing at random") would not.
