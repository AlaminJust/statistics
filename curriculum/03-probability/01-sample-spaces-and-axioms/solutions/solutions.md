# 03.01 Solutions

1. S = {HHH, HHT, HTH, THH, HTT, THT, TTH, TTT}, 8 outcomes. P(exactly two heads) = 3/8. P(at least one head) = 1 − 1/8 = **7/8**.
2. P(A ∪ B) = 0.6 + 0.5 − 0.3 = **0.8**; P(Aᶜ) = **0.4**; P(neither) = 1 − 0.8 = **0.2**.
3. No. If they were disjoint, axiom 3 would give P(A ∪ B) = 1.3 > 1 = P(S), but A ∪ B ⊆ S, so P(A ∪ B) ≤ 1.
4. P(sum ≥ 10) = (3 + 2 + 1)/36 = **1/6**. P(at least one 1) = 1 − (5/6)² = **11/36**. P(differ) = 1 − 6/36 = **5/6**.
5. B = A ∪ (B ∩ Aᶜ), and the two pieces are disjoint. So P(B) = P(A) + P(B ∩ Aᶜ) ≥ P(A), since the second term is ≥ 0 (axiom 1). ∎
6. Apply the two-event rule to (A ∪ B) and C, then expand P((A ∪ B) ∩ C) = P((A ∩ C) ∪ (B ∩ C)) with the two-event rule again.
   Check: A ∪ B ∪ C = {1, 2, 3, 4, 6}, so the left side is 5/6. Right side: 3/6 + 3/6 + 2/6 − 1/6 − 1/6 − 1/6 + 0 = 5/6 ✓.
7. 1 − 0.5¹⁰ = **0.99902**. De Méré: 1 − (35/36)²⁴ = **0.4914**, just *below* 1/2. He lost money on it, even though
   (by a faulty argument) he expected to win.
8. ```python
   def simulate_probability(event, trials, rng):
       hits = 0
       for _ in range(trials):
           if event(rng):
               hits += 1
       return hits / trials

   simulate_probability(lambda r: r.integers(1, 7, 2).sum() >= 10, 100_000, np.random.default_rng(0))  # ≈ 0.167
   ```
9. Early on the five curves look completely different. For large n they all squeeze towards 0.5. Their spread around 0.5
   shrinks roughly like 1/√n.
10. 12 face cards + 26 red cards − 6 red face cards = 32, so P = 32/52 = **8/13 ≈ 0.615**.
11. **2/3**, not 1/2. There are three gold coins you could have drawn, all equally likely. Two of them are in the
    gold–gold box. A simulation gives about 0.667.
12. Collect all the days with a 70% forecast and compute the share of days it actually rained. If the forecaster is
    calibrated, it's close to 70%. Do the same for every forecast level and plot actual frequency against stated
    probability: a calibrated forecaster lies on the diagonal.
