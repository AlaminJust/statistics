# 05.07 Solutions

1. d = (72 − 66)/12 = **0.5**, a "medium" effect by Cohen's benchmarks.
2. 2(1.96 + 1.2816)²/0.16 ≈ **131.4**; statsmodels (t-test) gives 132.3, so **133 per group**.
3. With 15 per group, power for d = 0.5 is only about **26%**. A non-significant result is expected even if a medium
   effect exists. Report the confidence interval (it will be wide), call the result inconclusive, and plan a larger study
   (64 per group for 80% power at d = 0.5).
4. (a) Two groups: **176 per group** (352 people). (b) Paired: **90 people** in total (a one-sample test on the
   differences). The paired design removes between-person variation and each person gives two measurements, so it's
   much cheaper, provided the differences really have effect size 0.3.
5. For d = 0.5: (a) α = 0.01 → **96** per group (up from 64); (b) 90% power → **86** per group.
6. Cohen's h for 4% vs 5% gives about **6,726 users per arm**.
7. ```python
   def cohens_d(x, y):
       nx, ny = len(x), len(y)
       pooled = math.sqrt(((nx - 1) * variance(x) + (ny - 1) * variance(y)) / (nx + ny - 2))
       return (mean(x) - mean(y)) / pooled

   def sample_size_two_means(delta, sd, alpha=0.05, power=0.8):
       z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
       return math.ceil(2 * (z_a + z_b) ** 2 * sd ** 2 / delta ** 2)
   ```
8. atanh(0.128) × √197 ≈ 1.80, so power ≈ P(Z > 1.96 − 1.80) ≈ **0.44**. The test was more likely to miss the effect than
   to find it. For 80% power you'd need n ≈ ((1.96 + 0.84)/atanh(0.13))² + 3 ≈ **460** students.
9. On the log scale, the shift is ln 1.3 ≈ 0.26 sd. Simulated Mann–Whitney power is about 0.77 at n = 220, 0.80 at 240
   and 0.83 at 260, so roughly **240 per group**.
10. Average significant estimate for true d = 0.2: n = 20 → **0.75** (3.8×); 50 → 0.50 (2.5×); 100 → 0.37 (1.8×);
    400 → 0.22 (1.1×). The exaggeration fades only once power is high (about 80% at n = 400).
11. Power = P(Z + d√(n/2) > z_{1−α/2}) = P(Z > z_{1−α/2} − d√(n/2)) = 1 − β. So z_{1−α/2} − d√(n/2) = −z_{1−β}, i.e.
    d√(n/2) = z_{1−α/2} + z_{1−β}, giving **n = 2(z_{1−α/2} + z_{1−β})²/d²**. ∎
12. With 1,000 users per arm at a 10% base rate, the test had 80% power only for lifts of about 4 percentage points or
    more (to roughly 14%). A realistic design change moves conversion by well under a point, so the test was unlikely to
    find a real effect; "no significant difference" is exactly what we'd expect even if the new design is better. The
    result is inconclusive, not evidence of no effect. Look at the confidence interval for the difference: it probably
    includes lifts that would matter commercially. To learn anything, the test needs either far more traffic (about
    15,000 per arm for a 1-point lift) or less noise (for example by adjusting for pre-experiment behaviour, CUPED).
