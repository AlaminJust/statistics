# 05.05 Exercises — Hypothesis testing

`make lesson L=05-inference/05-hypothesis-testing`

## ★ Warm-up

1. **(M)** 65 heads in 100 flips. Compute z and the two-sided p-value (normal approximation), then the exact binomial
   p-value.
2. **(C)** For a spam filter with H₀ = "this email is not spam", describe a Type I error and a Type II error. Which is
   worse for the user?
3. **(C)** A study reports "p = 0.20". List two things you can conclude and two you cannot.

## ★★ Core

4. **(M)** A taste tester correctly identifies the expensive wine in 9 of 12 trials. Test H₀: guessing (p = 0.5), both
   two-sided and one-sided. Which alternative is justified here, and why?
5. **(M)** A die is rolled 600 times and shows 120 sixes. Test whether it's fair with respect to sixes.
6. **(P)** Implement `permutation_test(x, y, n_perm=10_000, rng=None)` in `statkit/inference.py`: a two-sided test of
   the difference in means, returning the p-value (count + 1)/(n_perm + 1).
7. **(P)** Use a permutation test of the correlation (shuffle one variable) to test whether `sleep_hours` is related to
   `exam_score`. Then do the same for `commute_minutes`. Compare with `scipy.stats.pearsonr`.
8. **(P)** Test whether tutored students have a higher **median** score, with a permutation test.
9. **(P)** Reproduce the "p-values are uniform under H₀" simulation, but with exponential data (true mean 1) and a
   one-sample t-test of H₀: μ = 1 with n = 10. Is the test still valid (5% below 0.05)? What about n = 100?

## ★★★ Challenge

10. **(M)** Prove that if a test statistic T has a continuous null distribution with CDF F, then the one-sided p-value
    1 − F(T) is Uniform(0, 1) under H₀. (Hint: Lesson 04.02's probability integral transform.)
11. **(P)** **p-values and effect sizes.** Simulate two groups with a tiny true difference of 0.02 sd, for n = 100, 10,000
    and 1,000,000 per group. Report the p-value and the 95% interval for the difference each time. What does this
    teach about "significant" results in very large datasets?
12. **(C)** Read the [ASA statement on p-values](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
    (Wasserstein and Lazar, 2016). Summarise its six principles in one sentence each, in your own words.

Solutions: [solutions/solutions.md](solutions/solutions.md)
