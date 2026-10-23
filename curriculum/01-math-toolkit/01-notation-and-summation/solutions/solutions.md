# 01.01 Solutions

1. Σi = 1 + 2 + 3 + 4 + 5 = **15**. Σi² = 1 + 4 + 9 + 16 + 25 = **55**.
2. n = 3, x₂ = 4, Σxᵢ = 15, Σxᵢ² = 4 + 16 + 81 = 101, (Σxᵢ)² = 225, x̄ = 5.
3. Σᵢ₌₁⁵ 3i.
4. "The average, over all n observations, of the squared difference between the true value yᵢ and the prediction ŷᵢ"
   — the mean squared error.
5. Σ(3xᵢ + 2) = 3Σxᵢ + Σ2 = **3Σxᵢ + 2n**. Check: 3(15) + 2(3) = 51; directly 8 + 14 + 29 = 51. ✓
6. Σ(xᵢ − c) = Σxᵢ − Σc = Σxᵢ − nc. It equals 0 when c = Σxᵢ / n = **x̄**: the mean is the one number that deviations
   balance around.
7. Deviations from x̄ = 5: (−3, −1, 4); squares 9 + 1 + 16 = **26**. Shortcut: 101 − 3 × 25 = **26**. ✓
8. Σxᵢyᵢ = 1·3 + 2·4 = **11**. (Σxᵢ)(Σyᵢ) = 3 × 7 = **21**. The product of sums includes the cross terms x₁y₂ and
   x₂y₁ (1·4 + 2·3 = 10), which Σxᵢyᵢ does not: 21 − 10 = 11.
9. | | j = 1 | j = 2 |
   | --- | --- | --- |
   | i = 1 | 2 | 3 |
   | i = 2 | 3 | 4 |
   | i = 3 | 4 | 5 |

   Total = **21**. (Rule check: Σᵢ Σⱼ i + Σᵢ Σⱼ j = 2(1+2+3) + 3(1+2) = 12 + 9 = 21.)
10. ```python
    def summation(f, start, stop):
        total = 0
        for i in range(start, stop + 1):     # stop is INCLUDED in Σ notation
            total += f(i)
        return total
    summation(lambda i: 2 * i + 1, 1, 4)    # 24
    ```
11. `product(range(1, 7))` = **720**. 0.9¹⁰ ≈ **0.3487** — even very likely events rarely *all* happen when there are
    many of them.
12. S = 1 + 2 + … + n and S = n + (n−1) + … + 1. Adding column by column, each of the n columns sums to n + 1, so
    2S = n(n + 1) and **S = n(n + 1)/2**. For n = 100: 5050.
13. Σwxᵢ / Σw = w·Σxᵢ / (n·w) = Σxᵢ / n = x̄.
14. Errors: 0.5, −0.5, −1 → squares 0.25, 0.25, 1 → sum 1.5 → **L = 0.5**. NumPy: `np.mean((y - y_hat) ** 2)`.
