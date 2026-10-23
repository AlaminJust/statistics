# 03.05 Solutions

1. E[X] = 0 + 0.3 + 0.4 = **0.7**. E[X²] = 0 + 0.3 + 0.8 = **1.1**. Var(X) = 1.1 − 0.49 = **0.61**.
   CDF: F(x) = 0 for x < 0; 0.5 for 0 ≤ x < 1; 0.8 for 1 ≤ x < 2; 1 for x ≥ 2.
2. E[2X − 1] = **7**; Var(2X − 1) = 4 × 9 = **36**; sd = **6**.
3. (18/38)(1) + (20/38)(−1) = −2/38 ≈ **−$0.053** per $1 bet.
4. Profit = 300 − 20,000 with probability 0.01, and 300 otherwise. E[profit] = 300 − 0.01 × 20,000 = **$100**.
   Var = 20,000² × 0.01 × 0.99 = 3,960,000, so sd ≈ **$1,990**. One policy is very risky (sd ≫ mean). Over many
   independent policies the *average* profit per policy is very stable: its sd shrinks like 1/√n (Lesson 03.06), so
   the insurer is nearly sure to earn about $100 per policy.
5. P(face j appears at least once) = 1 − (5/6)⁶ ≈ 0.665. By linearity, E[distinct faces] = 6 × 0.665 ≈ **3.99**.
6. max = k in 2k − 1 of the 36 outcomes. E = Σ k(2k − 1)/36 = (1 + 6 + 15 + 28 + 45 + 66)/36 = 161/36 ≈ **4.47**.
7. Mean 50, variance 100 × 0.25 = 25, sd = **5**. Fewer than about 40 or more than about 60 (two sds away) would be
   surprising (Module 04 makes this precise).
8. ```python
   def expectation(values, probs):
       if abs(sum(probs) - 1) > 1e-9:
           raise ValueError("probabilities must sum to 1")
       return sum(v * p for v, p in zip(values, probs))

   def variance_rv(values, probs):
       mu = expectation(values, probs)
       return sum((v - mu) ** 2 * p for v, p in zip(values, probs))
   ```
10. E = 6/6 + 6/5 + 6/4 + 6/3 + 6/2 + 6/1 = 6 × (1 + 1/2 + … + 1/6) = **14.7** rolls.
11. E[aX + bY] = Σₓ Σᵧ (ax + by) p(x, y) = a Σₓ x Σᵧ p(x, y) + b Σᵧ y Σₓ p(x, y) = a Σₓ x p_X(x) + b Σᵧ y p_Y(y) =
    aE[X] + bE[Y]. The step Σᵧ p(x, y) = p_X(x) is just marginalisation. **Independence is never used.**
12. P(first heads on flip k) = 1/2ᵏ, so E = Σₖ 2ᵏ × (1/2ᵏ) = 1 + 1 + 1 + … = ∞. In simulation the running average never
    settles: it creeps up roughly like log₂(n)/2, with sudden jumps whenever a rare long run of tails pays a fortune.
    People pay little because huge prizes are extremely unlikely, and because the value of money isn't linear: an
    extra dollar matters less when you're already rich. That observation founded utility theory.
