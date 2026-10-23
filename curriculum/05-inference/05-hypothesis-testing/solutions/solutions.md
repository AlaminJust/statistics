# 05.05 Solutions

1. z = (0.65 − 0.5)/0.05 = **3.0**; two-sided p ≈ **0.0027**. Exact binomial: **0.0035**.
2. Type I (reject H₀ when it's true): a real email goes to the spam folder. Type II (fail to reject a false H₀): spam
   reaches the inbox. Most users find Type I worse (a missed job offer is costlier than one spam email), so spam
   filters use a strict threshold for labelling something spam.
3. Can conclude: the data are reasonably consistent with H₀; the study found no strong evidence of an effect. Cannot
   conclude: that there is no effect (the study may have low power); that P(H₀ true) = 0.2.
4. Two-sided p = **0.146**; one-sided (H₁: p > 0.5) p = **0.073**. A one-sided test is justified only if we committed in
   advance to "the tester does *better* than chance". Doing worse than chance would be no evidence of skill. Either way,
   it's not significant at 0.05.
5. p̂ = 0.20 vs 1/6. z = (0.2 − 0.1667)/√(0.1667 × 0.8333/600) ≈ **2.19**; exact binomial p ≈ **0.032**. Some evidence
   that sixes come up too often.
6. ```python
   def permutation_test(x, y, n_perm=10_000, rng=None):
       rng = rng or np.random.default_rng()
       x, y = np.asarray(x, float), np.asarray(y, float)
       observed = abs(x.mean() - y.mean())
       pooled = np.concatenate([x, y])
       count = 0
       for _ in range(n_perm):
           shuffled = rng.permutation(pooled)
           if abs(shuffled[:len(x)].mean() - shuffled[len(x):].mean()) >= observed:
               count += 1
       return (count + 1) / (n_perm + 1)
   ```
7. Sleep: r = 0.128, permutation p ≈ **0.071** (pearsonr 0.071): weak, not significant at 0.05. (The data were generated
   with a real but small sleep effect, so this is a Type II error. Lesson 05.07 explains why.) Commute: r = −0.021,
   p ≈ **0.77**: nothing.
8. The median difference is 8 points; permutation p ≈ **0.001**.
9. Exponential data with n = 10: **10.1%** of p-values fall below 0.05, so the test is invalid (twice the promised false
   positives). With n = 100: 5.8%, much better. The t-test's validity for skewed data depends on the CLT having kicked in.
10. P(p ≤ u) = P(1 − F(T) ≤ u) = P(F(T) ≥ 1 − u) = 1 − (1 − u) = u, because F(T) ~ Uniform(0, 1) when T has continuous
    CDF F. A random variable with P(p ≤ u) = u is uniform. ∎
11. True difference 0.02 sd: n = 100 gives p = 0.96 with interval [−0.29, 0.28]; n = 10,000 gives p = 0.11; n = 1,000,000
    gives **p ≈ 10⁻³⁷** with interval **[0.015, 0.021]**. With enough data any non-zero difference becomes
    "significant", however unimportant. In large datasets, look at the effect size and its interval, and ask whether it
    matters in practice (Lesson 05.07).
12. In brief: (1) p-values measure incompatibility between the data and a model; (2) they don't measure the probability
    that a hypothesis is true; (3) decisions shouldn't rest only on whether p crosses a threshold; (4) proper inference
    needs full reporting and transparency; (5) p doesn't measure effect size or importance; (6) by itself a p-value is
    not a good measure of evidence.
