# 08.04 Exercises — Uncertainty and conformal prediction

`make lesson L=08-statistical-learning/04-uncertainty-and-conformal`

## ★ Warm-up

1. **(M)** You have 99 calibration scores and want 90% coverage. Which order statistic (the k-th smallest) is q̂? What if
   you want 99% coverage with only 50 calibration points?
2. **(C)** What single assumption does split conformal prediction need? Give a real situation where it fails.
3. **(C)** Classify each as aleatoric or epistemic: (a) a coin flip; (b) a model that has never seen a cat breed;
   (c) measurement noise of a thermometer; (d) not knowing whether a relationship is linear.

## ★★ Core

4. **(P)** Implement `split_conformal_interval(cal_residuals, alpha)` in `statkit/learning.py`, returning q̂ (use the
   ⌈(n + 1)(1 − α)⌉-th smallest absolute residual). Reproduce worked example 1 with linear regression instead of the forest.
   Compare coverage and width.
5. **(P)** Plot the test coverage of split conformal intervals against α (0.05 to 0.5) for the diabetes data, averaged over
   50 random splits. Does it track 1 − α?
6. **(P)** For the digits classifier, compute prediction sets at 95% and report coverage by true digit. Which digits get
   the largest sets? Show a few images with set size ≥ 2.
7. **(P)** **Ensemble spread.** Train 10 gradient-boosted regressors with different random seeds and subsamples
   (`subsample=0.7`) on the heteroscedastic data of worked example 4. Use the mean ± 1.645 sd across the ensemble as a
   90% interval. What's its coverage? Why is it so different from 90%?
8. **(P)** Use conformal prediction to give an interval for exam_score from study_hours, sleep and tutoring in
   `students.csv` (linear regression; 60/40 split of the non-test data). Compare its width with the 90% prediction
   interval from statsmodels. When would you expect them to differ?

## ★★★ Challenge

9. **(M)** Prove the lower bound: if the n + 1 scores (n calibration plus 1 test) are exchangeable with no ties, then
   P(s_test ≤ s₍ₖ₎) = k/(n + 1). Hence choosing k = ⌈(n + 1)(1 − α)⌉ gives coverage ≥ 1 − α.
10. **(P)** **Weighted conformal under covariate shift.** In worked example 6, the shift is in x only. Weight each
    calibration point by the density ratio w(x) = p_test(x)/p_train(x) (use a simple estimate: a logistic classifier
    distinguishing calibration and test inputs). Does weighted conformal restore coverage? What happens when the test
    inputs lie completely outside the calibration range?
11. **(P)** **Conformal for an LLM-style task.** Simulate a 20-class "answer" classifier whose confidence is miscalibrated
    (draw logits, then multiply them by 3). Show that conformal sets still achieve 90% coverage, and that the set sizes
    reveal which questions are hard.

Solutions: [solutions/solutions.md](solutions/solutions.md)
