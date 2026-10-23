# 04.02 Solutions

1. Density 1/6 on [2, 8]. P(3 < X < 5) = 2/6 = **1/3**. Mean **5**, variance 36/12 = **3**.
2. λ = 0.001. P(X > 2000) = e⁻² ≈ **0.135**. Median = ln 2/0.001 ≈ **693 hours**.
3. `norm.ppf(0.95)` ≈ **1.645**: 95% of a standard normal lies below 1.645. `norm.cdf(-1)` ≈ **0.159**: 15.9% lies below −1.
4. λ = 1/8. P(> 10) = e^(−10/8) ≈ **0.287**. P(2 < W < 5) = e^(−2/8) − e^(−5/8) ≈ **0.244**. The 99th percentile is
   −8 ln(0.01) ≈ **36.8 minutes**.
5. Rate 0.3 per minute. P(no customer in 2 min) = e^(−0.6) ≈ **0.451**. Poisson with mean 0.6: P(0) = e^(−0.6) ✓. "No events
   in time t" is the same event as "the first event comes after t".
6. Mean 2/7 ≈ **0.286**. P(X < 0.5) ≈ **0.891**. Central 90%: **[0.063, 0.582]**.
7. Mean 3 × 2 = **6**, variance 3 × 4 = **12** (independent waits: variances add). P(X > 10) ≈ **0.125**.
8. ```python
   def exponential_cdf(x, rate):
       return 1 - math.exp(-rate * x) if x >= 0 else 0.0

   def sample_inverse_cdf(inverse_cdf, n, rng):
       return [inverse_cdf(u) for u in rng.uniform(size=n)]

   samples = sample_inverse_cdf(lambda u: -8 * math.log(1 - u), 100_000, np.random.default_rng(0))
   ```
9. The rate estimate is 1/23.52. The ECDF follows the fitted CDF closely: the data *were* generated as exponential, with
   rounding to whole minutes.
10. Median: 1 − e^(−λm) = 1/2 ⇒ m = ln 2/λ. Var = E[X²] − (E[X])² = 2/λ² − 1/λ² = **1/λ²**.
11. F(x) = x³, so F⁻¹(u) = u^(1/3). `np.cbrt(rng.uniform(size=100_000)).mean()` ≈ 0.75.
12. A **flat histogram (uniform)**: by the probability integral transform, F(X) ~ Uniform(0, 1) when F is X's true CDF.
    With the wrong CDF the histogram is not flat: here almost all values land between about 0.4 and 0.8, and very
    few near 0 or 1. The exponential puts much more probability than the gamma on very small values and on very large
    ones, so the gamma's draws rarely reach those extremes of the wrong CDF. A flat histogram of "predicted CDF values" is how forecasters check that their
    predictive distributions are calibrated.
