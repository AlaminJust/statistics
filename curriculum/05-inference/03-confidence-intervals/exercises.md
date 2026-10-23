# 05.03 Exercises — Confidence intervals

`make lesson L=05-inference/03-confidence-intervals`

## ★ Warm-up

1. **(M)** A sample of 25 has x̄ = 50 and s = 10. Give the 95% t-interval for μ.
2. **(M)** 45 of 50 test images are classified correctly. Give 95% Wald and Wilson intervals for the accuracy. Which
   would you report?
3. **(C)** Correct each statement: (a) "There's a 95% chance μ is in [62.0, 65.9]." (b) "95% of students scored between
   62.0 and 65.9." (c) "If we repeated the study, its mean would fall in [62.0, 65.9] with 95% probability."

## ★★ Core

4. **(M)** What sample size gives a ±0.5-point margin of error for the mean exam score (σ ≈ 14)? For a ±1-point margin on
   a proportion near 50%?
5. **(P)** Compute 95% t-intervals for the mean `study_hours` in each program. Plot them as points with error bars.
   Do any two programs clearly differ?
6. **(M)** Model A gets 90% accuracy and model B 85% on the same-size test set. Using the interval for a difference of
   two proportions, p₁ − p₂ ± 1.96√(p₁(1 − p₁)/n + p₂(1 − p₂)/n), is the gap clear with n = 100? With n = 1,000? (Treat
   the two test results as independent for now; Lesson 05.08 does it properly with paired data.)
7. **(P)** Implement `t_interval(values, level=0.95)` and `wilson_interval(successes, n, level=0.95)` in
   `statkit/inference.py`. You may use `scipy.stats.t.ppf` and `scipy.stats.norm.ppf` for the critical values. Check
   against the lesson.
8. **(P)** **Coverage under skew.** Simulate 50,000 samples of size 10 from an exponential with mean 1, build a 95%
   t-interval from each, and record the coverage. Repeat with n = 50. Explain the shortfall.
9. **(P)** Compute a 95% interval for the mean `sleep_hours`. Then write one sentence reporting it as you would in a paper.

## ★★★ Challenge

10. **(M)** Derive the Wilson interval: solve |p̂ − p| ≤ z√(p(1 − p)/n) for p (a quadratic inequality in p).
11. **(P)** For p from 0.01 to 0.5 and n = 30, plot the true coverage of the Wald and Wilson 95% intervals, computed
    exactly by summing binomial probabilities over all k. Where does Wald fail worst?
12. **(P)** Show by simulation that two independent 95% intervals can overlap while the 95% interval for the difference
    excludes zero. Find the largest separation of the two means (in SE units) at which the individual intervals still
    overlap.

Solutions: [solutions/solutions.md](solutions/solutions.md)
