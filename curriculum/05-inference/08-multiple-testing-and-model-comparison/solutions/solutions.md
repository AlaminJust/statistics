# 05.08 Solutions

1. Threshold 0.05/50 = **0.001**. Without correction, P(at least one false positive) = 1 − 0.95⁵⁰ ≈ **0.92**.
2. Sorted p: 0.0002, 0.004, 0.019, 0.03, 0.05, 0.12, 0.31, 0.48 (m = 8).
   - Bonferroni (threshold 0.00625): rejects **2** (0.0002, 0.004).
   - Holm: 0.0002 ≤ 0.05/8 ✓, 0.004 ≤ 0.05/7 = 0.0071 ✓, 0.019 > 0.05/6 = 0.0083 ✗, so it rejects **2**.
   - BH: thresholds (k/8) × 0.05 = 0.00625, 0.0125, 0.01875, 0.025, … The largest k with p₍ₖ₎ ≤ threshold is k = 2
     (0.019 > 0.01875, just), so it rejects **2**.
3. FWER: the probability of making *any* false discovery. FDR: the expected *share* of discoveries that are false.
   (a) FDR: you can tolerate a few false leads among candidates you'll study further. (b) FWER, or a single
   pre-registered test: one false approval is very costly.
4. 15 vs 5: exact binomial test on 20 disagreements, p ≈ **0.041**. 9 vs 3: p ≈ **0.146**, not significant.
5. ```python
   def bonferroni(pvalues, alpha=0.05):
       m = len(pvalues)
       return [p <= alpha / m for p in pvalues]

   def benjamini_hochberg(pvalues, q=0.05):
       m = len(pvalues)
       order = sorted(range(m), key=lambda i: pvalues[i])
       k_max = 0
       for rank, i in enumerate(order, start=1):
           if pvalues[i] <= rank / m * q:
               k_max = rank
       rejected = set(order[:k_max])
       return [i in rejected for i in range(m)]
   ```
6. Ten subgroups qualify. Only **one** (Science, year 4: n = 9 vs 15, p = 0.003) has p < 0.05, and it's also the only one
   BH rejects. The other nine all show non-significant results, but with only 3–18 students per arm they had very little
   power. The overall effect (d ≈ 0.6, p = 0.0001) is clear, and the data were generated with the *same* tutoring effect
   for everyone. "Works only for group X" from subgroup tests is a classic false conclusion: absence of significance in
   small subgroups is not absence of effect. To claim the effect differs between groups, test the *interaction* directly
   (Module 06).
7. False-positive rates (3,000 simulations): peeking every **1** subject → **41%**; every **10** → **25%**; every **50**
   → **13%**. More frequent peeking means more chances to cross 0.05 by chance.
8. Accuracies: C = 1 → 0.959, C = 0.01 → 0.942. The disagreements split 6 vs 3, McNemar p ≈ **0.51**; paired bootstrap
   interval for the difference about **[−0.018, 0.053]**. There's no evidence that either is better.
9. With the correction (k = 10, n_test/n_train = 100/900) the false-positive rate drops from about **32%** to about
   **17%**. That's better, but still above 5%, because in this simulation the folds are *more* correlated (ρ ≈ 0.26) than
   the correction assumes (ρ ≈ n_test/n_train ≈ 0.11). The correction is a heuristic. When it matters, use repeated
   cross-validation with independent repetitions, or truly independent test data.
10. Let Aᵢ be "test i gives a false positive". Each P(Aᵢ) ≤ α/m when tested at α/m. Then
    FWER = P(A₁ ∪ … ∪ A_m) ≤ Σ P(Aᵢ) ≤ m × α/m = α. No independence is needed. ∎
11. Typical findings: dozens of pairwise comparisons with no correction; best-of-several seeds reported; no intervals;
    the authors' method tuned more carefully than the baselines. Improvements: report mean ± a 95% interval over at
    least 5 seeds; use a paired test across datasets (Wilcoxon or Friedman–Nemenyi); and give baselines the same tuning
    budget.
