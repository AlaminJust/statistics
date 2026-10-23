# 06.02 Solutions

1. X: 200 × 5; **β**: 5; **y**: 200; XᵀX: 5 × 5; Xᵀ**y**: 5.
2. "Among students with the same weekly study hours and tutoring status, each extra hour of nightly sleep is associated
   with about 2.6 more exam points on average (95% CI 1.3 to 3.8)."
3. The three dummies add up to the intercept column, so the columns of X are linearly dependent and XᵀX has no inverse.
   Infinitely many coefficient combinations fit equally well.
4. Significant: study_hours (2.05), sleep_hours (2.46) and tutored (6.92), all p < 0.001. Not significant: both program
   dummies (p ≈ 0.8), year (p = 0.17) and commute (p = 0.76). That's exactly right: in `make_students.py` only study,
   sleep and tutoring affect the score, so there are no false positives and no false negatives this time.
5. Predicted mean **75.0**; 95% prediction interval **[57.6, 92.4]**.
6. The interaction sleep × tutored is −0.17 with p = **0.91**: no evidence that sleep's effect differs by tutoring.
7. SE of the study coefficient: 0.137 without sleep, **0.133** with it. Sleep explains part of the residual variation, so
   σ̂² falls, and with it the SE of every coefficient. Adding a predictor *correlated* with study would instead tend to
   inflate study's SE (multicollinearity, Lesson 06.03).
8. ```python
   def ols_standard_errors(X, y):
       X, y = np.asarray(X, float), np.asarray(y, float)
       beta = fit_ols(X, y)
       e = y - X @ beta
       sigma2 = e @ e / (X.shape[0] - X.shape[1])
       return np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
   ```
9. Regressing y on x alone estimates Cov(x, y)/Var(x) = [β₁Var(x) + β₂Cov(x, z)]/Var(x) = β₁ + β₂ × Cov(x, z)/Var(x) =
   **β₁ + β₂δ**. In the simulation the omitted variable is hours (effect 2) and δ = slope of hours on tutor ≈ 3.53, so
   the expected naive coefficient is 3 + 2 × 3.53 ≈ **10.06**. The naive fit gave 10.14 ✓.
10. The residual-on-residual slope is **2.0477**, exactly the full-model study coefficient. "Holding sleep and tutoring
    constant" means using only the part of study-hours variation that sleep and tutoring *can't* predict, and asking
    how much of the score variation not explained by them it accounts for.
11. The normal equations say Xᵀ(**y** − X**β̂**) = **0**, i.e. Xᵀ**e** = **0**. Geometrically, X**β̂** is the orthogonal
    projection of **y** onto the space spanned by X's columns, and the residual is perpendicular to that space.
12. About **41%** of the "best" single predictors have p < 0.05 when refitted, even though none is related to y. That's
    close to 1 − 0.95¹⁰ = 0.40. Searching for the best predictor and then testing it on the same data is multiple
    testing in disguise; the reported p-value is meaningless.
