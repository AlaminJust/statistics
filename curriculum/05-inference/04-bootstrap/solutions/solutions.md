# 05.04 Solutions

1. The sd of 10,000 resample means is about **2.57**. The formula s/√5 gives 2.88. With tiny n the bootstrap behaves like
   the population formula σ̂/√n with σ̂ computed by dividing by n (2.58); that's one reason it's unreliable for n below
   about 15.
2. Sampling with replacement mimics drawing fresh independent observations from the population, which the sample
   stands in for. Without replacement, every resample would just be the original data reordered. Keeping the size n
   matters because the variability of a statistic depends on n.
3. A resample's maximum can never exceed the sample maximum, and it equals it about 63% of the time. So the bootstrap
   distribution is a lump at one value, not the true sampling distribution, whose upper part lies *beyond* the sample
   maximum.
4. ```python
   def bootstrap_ci(data, statistic, n_boot=10_000, level=0.95, rng=None):
       rng = rng or np.random.default_rng()
       data = np.asarray(data)
       stats_ = [statistic(data[rng.integers(0, len(data), len(data))]) for _ in range(n_boot)]
       alpha = (1 - level) / 2
       return float(np.quantile(stats_, alpha)), float(np.quantile(stats_, 1 - alpha))
   ```
5. The median difference is 8 points; the 95% interval is about **[2, 14.5]**. It excludes 0.
6. CV = 0.483; 95% interval about **[0.434, 0.530]**.
7. 90th percentile = 54.2 minutes. Percentile interval about [43.0, 64.2]; BCa about **[43.4, 66.0]**, stretched upwards
   because the upper tail of a skewed variable is less certain.
8. Precision 0.972 [0.936, 1.000] and recall 0.963 [0.922, 0.991] (seed 0, test set of 171). Resample test examples and
   recompute both metrics on each resample.
9. The bootstrap correlations centre on **0** (mean ≈ 0.0005, sd ≈ 0.07). Resampling columns independently breaks the
   pairing, so you're bootstrapping the correlation of *unrelated* variables. Always resample whole rows.
10. Coverage (1,000 samples of n = 30, seed 1): percentile **92.7%**, BCa **93.5%**, t-interval **93.2%**. All fall a
    little short of 95% for skewed data at n = 30. BCa is the best of the three, but no method is perfect for small
    skewed samples.
11. Accuracies: logistic **0.959** [0.930, 0.988] and tree **0.901** [0.854, 0.942]. The separate intervals overlap, but
    the paired interval for the difference, **[0.023, 0.099]**, excludes 0. The pairing removes noise shared by both
    models (the same hard examples hurt both), so the comparison is much sharper.
12. Each of the n draws misses a given observation with probability 1 − 1/n, and the draws are independent, so
    P(never drawn) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368 (for n = 200 it's 0.367). Each bootstrap tree in a random forest therefore
    leaves out about 37% of the data, which serves as a built-in test set.
