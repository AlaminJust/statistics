# 05.06 Solutions

1. (a) Paired t-test (or Wilcoxon signed-rank) on each user's difference. (b) χ² test of independence on the 3 × 2
   table of design × clicked. (c) Mann–Whitney U (or a t-test on log salaries, or a permutation test of the medians).
   (d) χ² goodness of fit.
2. E = 79 × 60/200 = **23.7**. Contribution (28 − 23.7)²/23.7 ≈ **0.78**.
3. The unpaired test's noise is the between-student variation (sd ≈ 12). The paired test works on each student's own
   change, so the large baseline differences cancel, leaving the much smaller noise of the change (sd ≈ 5.3). A smaller
   SE for the same effect gives a larger t.
4. Welch t-test, Commerce vs Science: **p = 0.048**, "significant". ANOVA across programs is not significant, and
   Tukey's HSD (which corrects for making 3 comparisons) gives **p = 0.095** for Commerce vs Science. Picking out the
   biggest of three pairwise differences after looking at the data overstates the evidence.
5. Table: no tutoring 103 below 70 and 37 at or above; tutoring 32 and 28. χ² = **7.84**, p = **0.005**. The t-test on raw
   scores gives p = 0.0001. Dichotomising a numeric outcome throws information away and costs power.
6. χ² = 5.53, df = 3, **p = 0.137**. No evidence that tutoring depends on year.
7. ```python
   def welch_t_test(x, y):
       nx, ny = len(x), len(y)
       vx, vy = variance(x) / nx, variance(y) / ny
       t = (mean(x) - mean(y)) / math.sqrt(vx + vy)
       df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
       p = 2 * stats.t.sf(abs(t), df)
       return t, df, p
   ```
8. Over 1,000 simulations: the paired test rejects about **93%** of the time, the unpaired test **0%**. The 1 kg effect
   is invisible next to 15 kg of between-person variation unless you pair.
9. With the pooled proportion p = 65/200, z = (28/60 − 37/140)/√(p(1 − p)(1/60 + 1/140)) = **2.80**, and z² = **7.84** = χ² ✓.
10. Power with t₂ data: Welch **28%**, Mann–Whitney **43%**. With normal data: Welch **75%**, Mann–Whitney 72%. The rank test
    loses only a little under normality and gains a lot under heavy tails.
11. Fisher's exact p = **0.49**. The expected counts are all 2, far below 5, so the χ² approximation is unreliable. With
    8 patients, even a 75% vs 25% recovery split is entirely consistent with chance.
12. (1) 10 pairwise tests at α = 0.05 inflate the family-wise false-positive rate (to as much as 40%). (2) Accuracies on
    the same datasets are *paired*, so an unpaired test ignores the dataset effect; a paired test, or a Friedman test
    with post-hoc comparisons (Demšar, 2006), is appropriate. (3) Each accuracy estimate has its own test-set noise, and
    differences in tuning effort or random seeds are ignored. Report the variance across seeds as well.
