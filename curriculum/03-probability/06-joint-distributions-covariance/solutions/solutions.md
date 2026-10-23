# 03.06 Solutions

1. P(X = 1 | Y = 1) = 0.40/0.50 = **0.8**. P(X = 0 | Y = 0) = 0.30/0.50 = **0.6**.
2. Var(X + Y) = 4 + 9 − 4 = **9**. Var(X − Y) = 4 + 9 + 4 = **17**. Var(2X + 3Y) = 16 + 81 + 2·2·3·(−2) = **73**.
   Corr = −2/(2 × 3) = **−1/3**.
3. 12/√n = 2 ⇒ √n = 6 ⇒ **n = 36**.
4. p_X = (0.4, 0.6) and p_Y = (0.25, 0.5, 0.25). Every cell equals the product of its marginals (e.g. 0.4 × 0.25 = 0.10),
   so X and Y are **independent**, and Cov = **0**.
5. Cov(S, D) = Cov(A + B, A − B) = Var(A) − Var(B) + Cov(B, A) − Cov(A, B) = 0, since both dice have the same variance.
   They are **not** independent: S = 2 forces A = B = 1, so D = 0 with certainty, whereas P(D = 0) is only 1/6 in general.
6. Var(X + Y) = E[(X + Y − μ_X − μ_Y)²] = E[(X − μ_X)² + (Y − μ_Y)² + 2(X − μ_X)(Y − μ_Y)] = Var(X) + Var(Y) + 2Cov(X, Y),
   by linearity. ∎
7. Var = (1 + 9)/4 = 2.5, so sd ≈ **1.58**. That's **worse** than X₁ alone (sd 1). Averaging helps only when you weight
   the measurements sensibly; inverse-variance weighting gives the optimal combination (Module 05).
8. ```python
   def joint_to_marginals(table):
       rows = [sum(row) for row in table]
       cols = [sum(column) for column in zip(*table)]
       return rows, cols

   def covariance_rv(joint, x_values, y_values):
       px, py = joint_to_marginals(joint)
       ex = sum(x * p for x, p in zip(x_values, px))
       ey = sum(y * p for y, p in zip(y_values, py))
       exy = sum(x * y * joint[i][j] for i, x in enumerate(x_values) for j, y in enumerate(y_values))
       return exy - ex * ey
   ```
9. Simulated sds: 10.02, 3.16, 1.00, matching 10/√n = 10, 3.16, 1.0. The √n law doesn't need normal data, only
   independent draws with a finite variance.
10. Var(X/σ_X ± Y/σ_Y) = 1 + 1 ± 2ρ ≥ 0. With "+" this gives ρ ≥ −1, and with "−" it gives ρ ≤ 1. ∎
11. The actual sd of the mean is about **0.295**, nearly **3 times** the naive 0.1. Positive autocorrelation means each new
    observation brings less new information; the *effective* sample size here is only about 100 × (0.1/0.295)² ≈ 11.
    Ignoring this gives overconfident intervals for time series and for repeated measures.
12. Simulated sds of the average: ρ = 0 → 0.448, 0.3 → 0.662, 0.7 → 0.872, 1 → 1.0, matching √(ρ + (1 − ρ)/5).
    Derivation: Var(average) = (1/25)(5 × 1 + 20ρ) = ρ + (1 − ρ)/5. With highly correlated members, averaging barely
    helps: even infinitely many members leave a variance of ρ. Random forests use bootstrap samples and random feature
    subsets to make their trees *decorrelated*, which shrinks ρ and so the ensemble's error.
