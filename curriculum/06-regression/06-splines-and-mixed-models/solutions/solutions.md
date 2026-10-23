# 06.06 Solutions

1. A polynomial's coefficients are global and the high powers grow very fast, so to fit the middle of the data, the curve
   must swing wildly where data are sparse (the edges). Spline pieces are local cubics joined smoothly at knots: each
   piece only has to fit nearby data.
2. ICC = 25/(25 + 100) = **0.20**. wⱼ = 25/(25 + 100/nⱼ): n = 1 → **0.20**, 4 → **0.50**, 16 → **0.80**, 100 → **0.96**.
3. Examples: patients with several visits (groups = patients); pupils in classrooms; sales by store; benchmark scores of
   models on several datasets (groups = datasets); repeated runs of one model with different seeds.
4. CV RMSE (shuffled folds): 3 knots 4.043, 4 knots 4.047, 5 knots 4.052, 6–15 knots ≈ 4.07–4.08, 20 knots 4.172. The
   curve is simple, so a few knots suffice; beyond about 15 knots variance starts to cost.
5. F = 0.79, **p = 0.50**: no evidence of curvature in study hours. That's right: the score was generated as linear in
   study hours.
6. τ̂ ≈ **0.03**, essentially 0, with the fixed effects identical to OLS. Program truly has no effect, and with only 3
   groups a between-group variance can't be estimated reliably anyway. With so few groups, use fixed effects (dummies),
   or a Bayesian model with an informative prior on τ.
7. wⱼ = τ̂²/(τ̂² + σ̂²/nⱼ), and the estimate is wⱼȳⱼ + (1 − wⱼ)μ̂. It matches the mixed model's school estimates up to
   small differences: statsmodels uses REML estimates of τ and σ, and μ̂ is itself a precision-weighted mean.
8. Random intercept only: average slope 2.14, SE **0.049**. Random slopes: 2.10, SE **0.089**, slope sd 0.52 (true 0.5).
   When slopes really vary, the random-intercept model's SE for the average slope is far too small, because it ignores
   the school-to-school variation in slopes.
9. Averaged over 500 simulations: of the true 10 worst hospitals, raw rates identify **4.6** and shrunk rates **4.9**.
   The bigger difference is *which* hospitals get flagged: the raw "worst 10" have a median of only **176** patients,
   against **310** for the shrunk list. Raw rankings are dominated by small hospitals that were unlucky. Shrinkage
   moves them towards the average and gives large hospitals with consistently high mortality the attention they
   deserve. Publishing raw league tables unfairly penalises small hospitals (and rewards small lucky ones).
10. Write E[(wȳⱼ + (1 − w)μ − μⱼ)²]. Given μⱼ ~ (μ, τ²) and ȳⱼ | μⱼ ~ (μⱼ, σ²/nⱼ), the error is
    w(ȳⱼ − μⱼ) − (1 − w)(μⱼ − μ), whose two parts are independent. So MSE(w) = w²σ²/nⱼ + (1 − w)²τ². Setting the
    derivative 2wσ²/nⱼ − 2(1 − w)τ² to 0 gives **w = τ²/(τ² + σ²/nⱼ)**. ∎
11. 20 knots unpenalised: 4.17. 20 knots with a CV-chosen ridge penalty: **4.12**, slightly better, approaching the best
    small-knot spline (4.04). Penalisation lets you use many knots without having to choose their number exactly.
