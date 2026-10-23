# 04.03 Solutions

1. z(186) = 2, so P(> 186) = **0.0228**. P(160 < H < 180) = Φ(1.25) − Φ(−1.25) ≈ **0.789**. 25th percentile = 170 − 0.674 × 8 ≈
   **164.6 cm**; 95th = 170 + 1.645 × 8 ≈ **183.2 cm**.
2. X − Y ~ N(2, 9 + 16 = 25), so sd 5. P(X > Y) = P(X − Y > 0) = P(Z > −0.4) ≈ **0.655**.
3. The team. Averages have a smaller sd (σ/√5), so an *average* far from the mean is much rarer than a single person
   that far away. (With mean 175 and sd 7: one adult over 200 cm has P ≈ 1.8 × 10⁻⁴; a team average over 200 cm has
   P ≈ 7 × 10⁻¹⁶.)
4. P(W < 490) = Φ(−2.5) ≈ **0.0062**. For 1% under 500: μ = 500 + 2.326 × 4 ≈ **509.3 g**.
5. Total ~ N(60, 27), so sd 5.20. P(> 66) = P(Z > 1.155) ≈ **0.124**.
6. X ≈ N(80, 8²). P(X ≤ 70) ≈ Φ((70.5 − 80)/8) = Φ(−1.19) ≈ **0.118**; exact binomial **0.116**.
7. E[son | father 189] = 177 + 0.5 × (7/7) × (189 − 175) = **184 cm**. Conditional sd = 7√0.75 ≈ **6.06**.
   P(son > 189) = P(Z > 5/6.06) ≈ **0.205**. The father is 2 sd above average; his son is expected to be only 1 sd above.
   Tall fathers tend to have tall sons, but less extremely tall. Francis Galton found exactly this in the 1880s, and
   coined the word "regression" for it.
8. ```python
   def normal_cdf(x, mu=0.0, sigma=1.0):
       return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

   def normal_pdf(x, mu=0.0, sigma=1.0):
       z = (x - mu) / sigma
       return math.exp(-z * z / 2) / (sigma * math.sqrt(2 * math.pi))
   ```
9. Q–Q correlations: raw 0.981 (skew 0.76), log 0.990 (skew −0.54, now left-skewed), **sqrt 0.998** (skew 0.13). The
   square root is the best here. The log over-corrects, because gamma data with shape 4 aren't skewed enough to need it.
10. ```python
    def normal_ppf(q, mu=0.0, sigma=1.0):
        lo, hi = -10.0, 10.0
        while hi - lo > 1e-10:
            mid = (lo + hi) / 2
            if normal_cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return mu + sigma * (lo + hi) / 2
    ```
11. The slope is ρσ_Y/σ_X = [Cov/(σ_Xσ_Y)] × σ_Y/σ_X = **Cov(X, Y)/σ_X²**. ∎
12. **w**ᵀΣ**w** = 1 + 1 + 4 + 2(1)(−1)(0.8) = **4.4**. The sample variance of X₁ − X₂ + 2X₃ should be close to 4.4.
