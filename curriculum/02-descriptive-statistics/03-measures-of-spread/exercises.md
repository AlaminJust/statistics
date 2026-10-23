# 02.03 Exercises — Measures of spread

`make lesson L=02-descriptive-statistics/03-measures-of-spread`

## ★ Warm-up

1. **(M)** For 2, 4, 4, 4, 5, 5, 7, 9, compute the range, the sample variance and the sample standard deviation with a
   deviation table. Then compute the population standard deviation (÷ n).
2. **(M)** Recompute the sum of squared deviations in exercise 1 with the shortcut Σxᵢ² − n x̄².
3. **(C)** Explain to a friend, without formulas, why we square deviations instead of just averaging them.

## ★★ Core

4. **(M)** Daily temperatures have mean 20 °C and standard deviation 5 °C. Give the mean, standard deviation and variance
   in °F (°F = 1.8 × °C + 32).
5. **(M)** Data set P = (10, 10, 10, 10) and Q = (1, 19, 1, 19). Both have mean 10. Compute both standard deviations.
   Which measure of spread could you compute in your head?
6. **(P)** For each `program` in `students.csv`, compute the mean, standard deviation and CV of `exam_score`. Which
   program has the most variable scores?
7. **(P)** Implement in `statkit/descriptive.py`: `data_range`, `variance(values, ddof=1)`, `std(values, ddof=1)`,
   `median_absolute_deviation` and `coefficient_of_variation` (sample sd / mean). Use your own `mean` and `median`.
   Run `make test M=descriptive`.
8. **(P)** Repeat the n − 1 simulation for n = 2, 3, 5, 10, 50. Plot the average variance estimate against n for both
   divisors, with a horizontal line at the true variance.
9. **(M)** Show that if yᵢ = a·xᵢ + b, then s_y² = a² s_x² and s_y = |a| s_x. (Hint: first find ȳ, then yᵢ − ȳ.)

## ★★★ Challenge

10. **(M)** Prove that E[Σ(xᵢ − x̄)²] = (n − 1)σ², the reason for n − 1. Steps:
    (a) show Σ(xᵢ − x̄)² = Σ(xᵢ − μ)² − n(x̄ − μ)² by writing xᵢ − μ = (xᵢ − x̄) + (x̄ − μ);
    (b) take expectations, using E[(xᵢ − μ)²] = σ² and E[(x̄ − μ)²] = σ²/n (proved in Module 03).
11. **(P)** s² is unbiased for σ². Is s unbiased for σ? Simulate 200,000 samples of size 5 from a normal distribution
    with σ = 2 and average the sample standard deviations. Explain the result.
12. **(P)** **Streaming variance.** When data arrive one value at a time (sensor streams, training logs), you can't
    store them all. Implement Welford's algorithm: keep n, the running mean and M2 = running Σ(xᵢ − x̄)², updating

    ```text
    n += 1;  delta = x - mean;  mean += delta / n;  M2 += delta * (x - mean)
    ```

    Then variance = M2 / (n − 1). Check it against `np.var(..., ddof=1)` on `exam_score`.

Solutions: [solutions/solutions.md](solutions/solutions.md)
