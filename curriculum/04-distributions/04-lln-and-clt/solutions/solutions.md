# 04.04 Solutions

1. X̄ ≈ N(64, (14/7)²) = N(64, 2²). P(X̄ > 67) = P(Z > 1.5) ≈ **0.067**.
2. Count: sd = √(10,000 × 0.25) = **50**. Proportion: sd = 50/10,000 = **0.005**.
3. The LLN says the average of many observations gets close to the true mean. The CLT says *how* it scatters around the
   true mean on the way: approximately normally, with spread σ/√n.
4. Sum ≈ N(350, 100 × 35/12 = 291.7), sd 17.08. P(sum ≥ 370) ≈ P(Z > (369.5 − 350)/17.08 = 1.14) ≈ **0.127**.
5. p̂ ≈ N(0.52, 0.52 × 0.48/1000), sd 0.0158. P(p̂ < 0.5) = P(Z < −1.27) ≈ **0.103**. One poll in ten would show the
   leading candidate losing.
6. Count ≈ N(100, 98), sd 9.90. P(X ≥ 120) ≈ P(Z > (119.5 − 100)/9.90) ≈ **0.024**; exact binomial **0.027**.
7. Hoeffding: n ≥ ln(200)/(2 × 0.0001) ≈ **26,492**. CLT: (2.576/0.01)² × 0.25 ≈ **16,590**.
8. Markov: P(X ≥ 100) ≤ 23.5/100 = **0.235**. Chebyshev: 100 is 76.5 above the mean, i.e. k = 3.75 sd, so the bound is
   ≤ 1/k² ≈ **0.071**. Actual share in `students.csv`: **0.5%** (one student).
9. The uniform converges fastest: it's symmetric, so its mean has no skewness to lose, and by n = 5 the histogram is
   nearly normal. Poisson(1) is skewed (skewness 1), so it needs larger n, and it stays visibly discrete for small n.
   Beta(0.5, 0.5) is U-shaped but symmetric, and converges quickly.
10. Typical results (seed 0): n = 100 → 2.92, 95% interval [2.57, 3.27]; n = 10⁴ → 3.152 [3.120, 3.184];
    n = 10⁶ → 3.1410 [3.1378, 3.1443]. The error shrinks like **1/√n**: 100 times more points give one more correct
    decimal.
11. Log-normal(0, 1.5) means (seed 0):

    | n | skew of mean | P(Z < −1.96) | P(Z > 1.96) |
    | --- | --- | --- | --- |
    | 30 | 3.5 | 0.000 | 0.035 |
    | 100 | 2.0 | 0.000 | 0.039 |
    | 1000 | 0.9 | 0.008 | 0.034 |

    Even n = 1,000 is not enough: the mean is still clearly skewed, and the tails are far from 2.5% each. For data like
    incomes, purchase values or file sizes, use the bootstrap (Module 05), analyse on the log scale, or use much larger
    samples.
12. Hoeffding's n = ln(2/δ)/(2ε²) does not involve p at all. The CLT's n = (z/ε)² p(1 − p) is largest at p = 0.5, and
    Hoeffding matches the CLT's worst case only up to a constant factor. When accuracy is near 0 or 1 (say 0.99),
    p(1 − p) = 0.0099 is 25 times smaller than 0.25, so Hoeffding demands far more data than needed. Variance-aware bounds
    (Bernstein's inequality) fix this.
