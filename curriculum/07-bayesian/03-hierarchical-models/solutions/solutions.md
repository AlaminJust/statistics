# 07.03 Solutions

1. τ → 0: every θⱼ is forced to equal μ (**complete pooling**). τ → ∞: the population distribution is flat, so each θⱼ is
   estimated from its own yⱼ alone (**no pooling**).
2. w_A = 25/(25 + 225) = **0.10**; w_E = 25/(25 + 81) = **0.24**. School A (the noisier estimate) keeps only 10% of its own
   value and is shrunk more.
3. Where τ is small, all θⱼ must sit very close to μ, which is a narrow neck; where τ is large, they can spread out,
   which is a wide mouth. One step size can't suit both, so the sampler either takes tiny steps everywhere or overshoots
   in the neck, which is what a divergence is. Writing θⱼ = μ + τzⱼ makes the sampled zⱼ have the same scale whatever τ
   is, so the funnel disappears from the space being sampled.
4. P(τ < 5) ≈ **0.75**. The data are most consistent with small between-school variation, but they can't exclude τ ≈ 10:
   with 8 noisy estimates, how much schools truly differ remains quite uncertain.
5. P(θ_A > θ_C) ≈ **0.63**. The raw estimates (28 vs −3) suggest a big difference, but after pooling it's only slightly
   more likely than not that A's programme worked better.
6. With μ ~ N(0, 20²): μ ≈ 7.5 (close to the complete-pooling 7.7), τ median 2.8, θ_A ≈ 9.0. With τ ~ HalfNormal(10):
   μ ≈ 4.4, τ median 4.2, θ_A ≈ 7.1. The prior on μ mattered more here: N(0, 5²) is informative relative to the data.
   Report such sensitivity.
7. τ posterior median ≈ **1.4**, 95% interval [0.08, 9.1]: the programs could be identical (they are, in the generating
   script), but with only 3 groups τ is barely identified. The mixed model of Lesson 06.06 estimated τ ≈ 0, a single
   point with no uncertainty. The Bayesian posterior shows honestly how little 3 groups can tell you.
8. Stores with few visitors have raw rates scattered widely (some near 0%, some over 20%). Their posterior means are
   pulled close to the overall rate of about 10%. Stores with 1,000+ visitors barely move. The plot looks like Lesson
   06.06's partial-pooling figure.
9. Given μ and τ, the prior is θⱼ ~ N(μ, τ²) and the likelihood is yⱼ ~ N(θⱼ, σⱼ²). By Lesson 07.01:
   precision 1/τ² + 1/σⱼ², mean = (μ/τ² + yⱼ/σⱼ²)/(1/τ² + 1/σⱼ²) = wⱼyⱼ + (1 − wⱼ)μ with wⱼ = τ²/(τ² + σⱼ²).
10. With `target_accept = 0.99` the centred model still has **14 divergences** and ESS(τ) ≈ 140, against 0 divergences
    and about 2,200 for the non-centred model. Smaller steps reduce, but don't remove, the problem. Reparameterisation
    fixes it.
11. Raw lifts: desktop **+1.45** points (n = 7,589), mobile **+1.07** points (n = 12,411). A hierarchical model shrinks both
    towards a common effect. With only two segments, τ is essentially determined by the prior: two groups can't tell
    you much about how much segments vary in general. You can still estimate each segment's effect more stably, but
    claims like "effects differ by device" need much more evidence (or many segments).
