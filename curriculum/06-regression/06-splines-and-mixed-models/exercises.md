# 06.06 Exercises — Splines and mixed models

`make lesson L=06-regression/06-splines-and-mixed-models`

## ★ Warm-up

1. **(C)** Why does a degree-15 polynomial behave badly near the edges of the data, while a cubic spline doesn't?
2. **(M)** A random-intercept model gives τ = 5 and σ = 10. Compute the ICC. Then compute the partial-pooling weight wⱼ
   for groups of size 1, 4, 16 and 100.
3. **(C)** Give two real datasets where observations come in groups, and say what the groups are.

## ★★ Core

4. **(P)** Using the simulated practice data from `examples.py`, compute the 10-fold CV RMSE (shuffled folds) of cubic
   splines with 3, 4, 5, 6, 8, 10, 15 and 20 knots. Plot CV RMSE against the number of knots. Where's the minimum?
5. **(P)** In `students.csv`, fit `exam_score ~ bs(study_hours, df=4) + sleep_hours + tutored` and compare it with the
   linear model using an F-test. Is there evidence of curvature? (Recall how the data were generated.)
6. **(P)** Fit a random-intercept model for `exam_score` with `program` as the grouping variable and study hours, sleep
   and tutoring as fixed effects. What τ does it estimate? Why is a mixed model with only 3 groups a poor idea?
7. **(P)** **Shrinkage formula by hand.** In the lesson's 40-school simulation, compute wⱼ for every school from the
   fitted τ̂ and σ̂, form wⱼȳⱼ + (1 − wⱼ)μ̂ yourself, and check it matches the mixed model's estimates.
8. **(P)** Simulate 50 schools in which each has its own hours slope, drawn from N(2, 0.5²). Fit random-intercept and
   random-slope models (`re_formula="~hours"`) and compare their estimates of the average slope and of the slope sd.

## ★★★ Challenge

9. **(P)** **Ranking.** Simulate 100 hospitals with true mortality rates ~ Beta(20, 180) (mean 10%) and patient counts
   from 10 to 500. Rank them by (a) the raw rate and (b) a shrunk rate (fit a mixed logistic model, or use the simple
   empirical-Bayes formula (deaths + 20)/(patients + 200)). Among the 10 "worst" hospitals by each method, how many are
   truly in the worst 10? Why does raw ranking mislead?
10. **(M)** Show that wⱼ = τ²/(τ² + σ²/nⱼ) is the weight that minimises the expected squared error of
    wȳⱼ + (1 − w)μ as an estimate of μⱼ, assuming μ, τ² and σ² are known.
11. **(P)** Compare a penalised spline with a GAM (`pygam` is not installed, so use scikit-learn: `SplineTransformer`
    with 20 knots followed by `RidgeCV`) against the unpenalised 20-knot spline on the practice data. Does the penalty help?

Solutions: [solutions/solutions.md](solutions/solutions.md)
