# 07.03 Exercises — Hierarchical models

`make lesson L=07-bayesian/03-hierarchical-models`

## ★ Warm-up

1. **(C)** What happens to the θⱼ estimates as τ → 0? As τ → ∞? Which pooling does each limit correspond to?
2. **(M)** With τ = 5 fixed, compute the shrinkage weight wⱼ = τ²/(τ² + σⱼ²) for schools A (σ = 15) and E (σ = 9). Which
   is shrunk more towards μ?
3. **(C)** Explain the funnel in your own words, and why the non-centred version avoids it.

## ★★ Core

4. **(P)** Fit the non-centred Eight Schools model, and plot the posterior of τ. Report P(τ < 5). What does it say about
   how much the coaching effect really varies between schools?
5. **(P)** Compute the posterior probability that school A's effect is larger than school C's. Compare with what the raw
   estimates suggest.
6. **(P)** **Prior sensitivity.** Refit with μ ~ N(0, 20²) and with τ ~ HalfNormal(10). How do μ, τ and θ_A change? Which
   prior matters more?
7. **(P)** Build a hierarchical model for `exam_score` across the three programs of `students.csv`: θ_program ~ N(μ, τ²),
   scores ~ N(θ_program + 2·(study_hours − 10), σ²). (Fix the study slope at 2 for simplicity, or give it a prior.) What's
   the posterior of τ? Compare with the mixed model of Lesson 06.06, exercise 6.
8. **(P)** **Shrinkage with uneven sizes.** Simulate 20 "stores" whose true conversion rates are Beta(20, 180) (about 10%),
   with 20 to 2,000 visitors each. Fit a hierarchical logistic model (logit θⱼ ~ N(μ, τ²)) in PyMC. Plot each store's raw
   rate and its posterior mean against its number of visitors, as in Lesson 06.06.

## ★★★ Challenge

9. **(M)** For the two-level normal model with μ and τ known, derive the posterior of θⱼ given yⱼ. (It's a Normal–Normal
   update, Lesson 07.01.)
10. **(P)** Fit the **centred** model with `target_accept = 0.99`. Does that remove the divergences? Compare the ESS of τ
    with the non-centred model.
11. **(P)** **A/B tests across segments.** Using `data/ab_test.csv`, estimate the treatment effect on conversion
    separately for mobile and desktop (no pooling), and with a hierarchical model for the segment-level effects. With
    only two segments, what can τ tell you, and what can't it?

Solutions: [solutions/solutions.md](solutions/solutions.md)
