# 03.06 Exercises — Joint distributions and covariance

`make lesson L=03-probability/06-joint-distributions-covariance`

## ★ Warm-up

1. **(M)** From the joint table in the lesson, find P(X = 1 | Y = 1) and P(X = 0 | Y = 0).
2. **(M)** Var(X) = 4, Var(Y) = 9, Cov(X, Y) = −2. Find Var(X + Y), Var(X − Y), Var(2X + 3Y) and Corr(X, Y).
3. **(M)** A measurement has sd 12. How many independent measurements must you average to get a standard error of 2?

## ★★ Core

4. **(M)** The joint PMF of X ∈ {1, 2} and Y ∈ {1, 2, 3} is

   | | Y = 1 | Y = 2 | Y = 3 |
   | --- | --- | --- | --- |
   | X = 1 | 0.10 | 0.20 | 0.10 |
   | X = 2 | 0.15 | 0.30 | 0.15 |

   Find both marginals, and decide whether X and Y are independent. Compute Cov(X, Y).
5. **(M)** Roll two dice. Let S = sum and D = first − second. Show that Cov(S, D) = 0. Are S and D independent?
   (Hint: what does S = 2 tell you about D?)
6. **(M)** Prove Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y) from the definitions.
7. **(M)** You average two independent measurements that have different standard deviations, σ₁ = 1 and σ₂ = 3. What is
   the sd of the simple average (X₁ + X₂)/2? Is it better than using X₁ alone? (Module 05 shows the optimal weighting.)
8. **(P)** Implement in `statkit/probability.py`: `joint_to_marginals(table)` (table = list of rows; returns
   (row marginals, column marginals)) and `covariance_rv(joint, x_values, y_values)`. Check exercise 4 and worked
   example 1.
9. **(P)** Simulate 100,000 averages of n draws from an exponential distribution with mean 10 (`rng.exponential(10, ...)`,
   σ = 10) for n = 1, 10, 100. Compare their standard deviations with σ/√n.

## ★★★ Challenge

10. **(M)** Prove that −1 ≤ Corr(X, Y) ≤ 1. (Hint: Var(X/σ_X + Y/σ_Y) ≥ 0 and Var(X/σ_X − Y/σ_Y) ≥ 0.)
11. **(P)** **Correlated observations.** Generate n = 100 observations with an AR(1) dependence: x₀ ~ N(0, 1) and
    xₜ = 0.8 xₜ₋₁ + √(1 − 0.64) εₜ with εₜ ~ N(0, 1), so every xₜ has variance 1. Simulate 10,000 such series and find the
    actual sd of their means. Compare with the naive standard error 1/√100 = 0.1. By what factor is the naive one wrong?
12. **(P)** **Ensembles.** Simulate 5 "models" whose prediction errors each have sd 1 and pairwise correlation ρ. Measure
    the sd of the average error for ρ = 0, 0.3, 0.7 and 1, and derive the formula √(ρ + (1 − ρ)/5). Why do random forests
    try to make their trees less correlated?

Solutions: [solutions/solutions.md](solutions/solutions.md)
