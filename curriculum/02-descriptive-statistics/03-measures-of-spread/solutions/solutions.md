# 02.03 Solutions

1. Mean = 40/8 = 5. Range = 9 − 2 = **7**.

   | xᵢ | 2 | 4 | 4 | 4 | 5 | 5 | 7 | 9 | sum |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | xᵢ − x̄ | −3 | −1 | −1 | −1 | 0 | 0 | 2 | 4 | 0 |
   | (xᵢ − x̄)² | 9 | 1 | 1 | 1 | 0 | 0 | 4 | 16 | **32** |

   s² = 32/7 ≈ **4.571**, s ≈ **2.138**. Population sd = √(32/8) = **2.0**.
2. Σxᵢ² = 4 + 16 + 16 + 16 + 25 + 25 + 49 + 81 = 232; n x̄² = 8 × 25 = 200; 232 − 200 = **32** ✓.
3. Deviations above and below the average cancel out exactly, so their plain average is always zero. Squaring makes
   every deviation count as positive, and it counts big misses much more than small ones.
4. Mean = 1.8 × 20 + 32 = **68 °F**. sd = 1.8 × 5 = **9 °F** (adding 32 doesn't change spread). Variance = 81 °F².
5. s_P = **0**. Q: deviations ±9, squares sum to 324, s² = 324/3 = 108, s_Q ≈ **10.39**. The range (0 and 18) is easy
   to see in your head.
6. | program | mean | sd | CV |
   | --- | --- | --- | --- |
   | Arts | 64.08 | 14.94 | 0.233 |
   | Commerce | 65.02 | 15.11 | 0.232 |
   | Science | 63.11 | 11.98 | 0.190 |

   Arts and Commerce are about equally variable; Science scores are the most consistent.
7. See `../solutions/statkit_descriptive_reference.py`.
8. The ÷ n curve starts at half the true value for n = 2 and rises towards it as (n − 1)/n · σ². The ÷ (n − 1) curve
   stays flat at the true value for every n.
9. ȳ = a x̄ + b, so yᵢ − ȳ = a(xᵢ − x̄). Then Σ(yᵢ − ȳ)² = a² Σ(xᵢ − x̄)², so s_y² = a² s_x² and s_y = |a| s_x.
10. (a) Σ(xᵢ − μ)² = Σ[(xᵢ − x̄) + (x̄ − μ)]² = Σ(xᵢ − x̄)² + 2(x̄ − μ)Σ(xᵢ − x̄) + n(x̄ − μ)². The middle term is 0, because
    deviations from x̄ sum to zero. Rearranging gives the identity.
    (b) E[Σ(xᵢ − x̄)²] = nσ² − n(σ²/n) = **(n − 1)σ²**. So E[s²] = σ². ∎
11. The average s is about **1.88**, not 2: s underestimates σ by about 6% when n = 5. The square root is a curved
    function, so the average of square roots is below the square root of the average (this is Jensen's inequality,
    Module 04). Unbiasedness doesn't survive non-linear transformations.
12. ```python
    def welford(values):
        n, mean, m2 = 0, 0.0, 0.0
        for x in values:
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
        return mean, m2 / (n - 1)
    ```
    On `exam_score` it returns variance ≈ 191.48 (sd 13.84), matching `np.var(x, ddof=1)`.
