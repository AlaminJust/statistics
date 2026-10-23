# 04.05 Solutions

1. SE = 12/√16 = **3**. t₀.₉₇₅ with 15 df = **2.131**.
2. Mean **20**, variance **40**.
3. The sd (13.84) describes how much individual students' scores vary. The SE of the mean, 13.84/√200 ≈ **0.98**,
   describes how much the *average* would vary if we drew another sample of 200. The sd stays about the same as n grows;
   the SE shrinks like 1/√n.
4. √(100/50 + 144/40) = √(2 + 3.6) ≈ **2.37**.
5. √(0.42 × 0.58/600) ≈ **0.020**. Halving the SE needs 4 times the sample: **2,400** people.
6. P(T₅ > 2.5) ≈ **0.0272** and P(Z > 2.5) ≈ **0.0062**. The normal understates the tail by a factor of about **4.4**.
7. χ²₉ 2.5% and 97.5% points: 2.700 and 19.023. Solve 2.700 ≤ 9 × 196/σ² ≤ 19.023: **σ² between 92.7 and 653.2**
   (σ between 9.6 and 25.6). The interval is very wide, as expected: 10 observations say little about a variance.
8. Mean range ≈ 3.08 and SE ≈ **0.80**. Skewness ≈ 0.40: right-skewed, not symmetric.
9. About **3.7%**, below 5%. Sampling 25 of only 200 students *without replacement* makes sample means slightly less
   variable than the i.i.d. formula assumes (the finite population correction, √((N − n)/(N − 1)) ≈ 0.94), so the t
   statistic is a little too small on average.
10. T = Z/√(V/k) with V ~ χ²_k independent of Z. Then T² = Z²/(V/k) = (χ²₁/1)/(χ²_k/k), which is the definition of F(1, k). ∎
11. False-positive rates at nominal 5% (seed 0):

    | Data | P(\|t\| > 2.262) | lower tail | upper tail |
    | --- | --- | --- | --- |
    | uniform | 0.054 | 0.027 | 0.026 |
    | exponential | **0.100** | 0.096 | 0.004 |
    | t₃ | 0.041 | 0.021 | 0.020 |

    The t-test is robust to symmetric non-normality (uniform, t₃) but not to strong skew at n = 10: with exponential
    data the rate doubles, and the errors fall almost all in one tail. Use larger n, transform the data, or use the
    bootstrap or a non-parametric test.
12. Simulated SE of r: ρ = 0 → 0.186 (approximation 0.183); 0.5 → 0.143 (0.137); 0.9 → 0.038 (0.035). When the points lie
    close to a line, every sample shows that line clearly, so r varies little. And r can't exceed 1, which squeezes its
    distribution near ±1 (it also becomes skewed there; Fisher's z = atanh(r) transform fixes that).
