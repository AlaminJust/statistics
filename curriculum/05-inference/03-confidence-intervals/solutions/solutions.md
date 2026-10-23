# 05.03 Solutions

1. SE = 10/5 = 2; t₀.₉₇₅,₂₄ = 2.064. Interval 50 ± 4.13 = **[45.87, 54.13]**.
2. Wald: **[0.817, 0.983]**. Wilson: **[0.786, 0.957]**. Report Wilson: with p̂ = 0.9 and n = 50, Wald is too narrow at the
   top and too optimistic overall.
3. (a) "We are 95% confident that μ is between 62.0 and 65.9: the method captures μ in 95% of samples." (b) The interval
   is about the *mean*; individual scores range far more widely (sd ≈ 14). (c) A repeat study's mean falls in it less
   often, about 83% on average, because the repeat has its own noise.
4. n = (1.96 × 14/0.5)² ≈ **3,012**. For a proportion near 50% with ±1 point: (1.96 × 0.5/0.01)² ≈ **9,604**.
5. | program | mean | 95% interval |
   | --- | --- | --- |
   | Arts | 10.52 | [9.29, 11.75] |
   | Commerce | 11.13 | [9.51, 12.75] |
   | Science | 9.27 | [8.38, 10.17] |

   The intervals overlap heavily, so there's no clear difference. (Overlap isn't a formal test, but here even the most
   separated pair, Commerce vs Science, isn't clear-cut; Lesson 05.06 tests it.)
6. n = 100: 0.05 ± 1.96 × 0.0466 = **[−0.041, 0.141]**, which includes 0, so it's not clear. n = 1,000: **[0.021, 0.079]**,
   clear. The same 5-point gap means very different things on different test-set sizes.
7. ```python
   def t_interval(values, level=0.95):
       n = len(values)
       m, se = mean(values), std(values) / math.sqrt(n)
       t = stats.t.ppf(0.5 + level / 2, n - 1)
       return m - t * se, m + t * se

   def wilson_interval(successes, n, level=0.95):
       z = stats.norm.ppf(0.5 + level / 2)
       p = successes / n
       centre = (p + z * z / (2 * n)) / (1 + z * z / n)
       half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
       return centre - half, centre + half
   ```
8. Coverage: about **90.1%** for n = 10 and **93.5%** for n = 50, below the nominal 95%. With right-skewed data, samples
   with an unusually low mean also tend to have a small s, so their intervals are too short and miss μ. The CLT
   correction takes a while; the bootstrap (or log-scale methods) does better.
9. [6.82, 7.09] hours. "Students slept 6.96 hours per night on average (95% CI 6.82 to 7.09, n = 200)."
10. Square both sides: (p̂ − p)² ≤ z² p(1 − p)/n. Rearrange to a quadratic in p:
    (1 + z²/n)p² − (2p̂ + z²/n)p + p̂² ≤ 0. Its two roots are the Wilson endpoints given in the lesson.
11. Exact coverage at n = 30:

    | p | Wald | Wilson |
    | --- | --- | --- |
    | 0.01 | **0.26** | 0.96 |
    | 0.05 | 0.78 | 0.94 |
    | 0.10 | 0.81 | 0.97 |
    | 0.20 | 0.95 | 0.96 |
    | 0.50 | 0.96 | 0.96 |

    Wald fails catastrophically for small p (rare events, such as error rates of good models, click rates, adverse events).
12. For two equal-SE estimates, the individual 95% intervals overlap while the means are less than 2 × 1.96 = 3.92 SEs apart.
    The difference has SE √2, so it is significant once the means are more than 1.96√2 ≈ 2.77 SEs apart. **Between 2.77
    and 3.92 SEs apart**, the intervals overlap even though the difference is significant.
