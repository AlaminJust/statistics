# 05.04 Exercises — The bootstrap

`make lesson L=05-inference/04-bootstrap`

## ★ Warm-up

1. **(P)** Draw 5 bootstrap resamples of (3, 7, 8, 12, 20) by hand (roll a die, or use `rng.choice`) and compute each
   one's mean. Then compute 10,000 resample means and their sd. Compare with s/√5.
2. **(C)** Why must resamples be drawn with replacement and have the same size as the original sample?
3. **(C)** Why does bootstrapping the maximum fail?

## ★★ Core

4. **(P)** Implement `bootstrap_ci(data, statistic, n_boot=10_000, level=0.95, rng=None)` in `statkit/inference.py`,
   returning the percentile interval. Test it on the mean exam score.
5. **(P)** Bootstrap a 95% interval for the difference in **median** exam score between tutoring and non-tutoring
   students. Resample within each group separately.
6. **(P)** Bootstrap a 95% interval for the coefficient of variation of `study_hours`.
7. **(P)** Bootstrap the 90th percentile of `commute_minutes`. Compare the percentile and BCa intervals.
8. **(P)** Train any classifier on the breast-cancer data (as in `examples.py`) and report its precision and recall on
   the test set, with 95% bootstrap intervals.
9. **(P)** **Wrong unit.** Bootstrap r(study_hours, exam_score) the wrong way, resampling each column *independently*.
   What happens to the bootstrap distribution, and why?

## ★★★ Challenge

10. **(P)** **Coverage check.** Simulate 1,000 samples of size 30 from an exponential (mean 1). For each, compute the
    percentile and BCa 95% intervals for the mean (B = 2,000). What share of each kind contains the true mean 1? Compare
    with the t-interval's coverage (Lesson 05.03, exercise 8).
11. **(P)** **Paired bootstrap for model comparison.** Train logistic regression and a decision tree
    (`DecisionTreeClassifier(max_depth=3)`) on the same split. Bootstrap the test set, computing **both** accuracies on
    each resample, and form an interval for their *difference*. Compare it with what you'd conclude from the two separate
    intervals.
12. **(M)** Show that the probability a particular observation is *not* in a bootstrap resample is (1 − 1/n)ⁿ, and that
    this tends to e⁻¹ ≈ 0.368. (This "out-of-bag" fraction is used by random forests to estimate test error for free.)

Solutions: [solutions/solutions.md](solutions/solutions.md)
