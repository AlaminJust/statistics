# 06.03 Exercises — Diagnostics and assumptions

`make lesson L=06-regression/03-diagnostics-and-assumptions`

## ★ Warm-up

1. **(C)** A residuals-vs-fitted plot shows a funnel opening to the right. What's wrong? What's the quickest fix for the
   standard errors, and what might fix the model itself?
2. **(C)** Why can a point with a small residual still be very influential? (Hint: it may pull the line towards itself.)
3. **(M)** A predictor has VIF = 20. By what factor is its SE inflated compared with uncorrelated predictors? What R_j²
   does that correspond to?

## ★★ Core

4. **(P)** Fit `target ~` all ten predictors on the scikit-learn diabetes data (`load_diabetes(scaled=False, as_frame=True)`).
   Make the four diagnostic plots. Report R², the Breusch–Pagan p-value, the maximum Cook's distance and the largest
   leverage. What do you conclude?
5. **(P)** Compute the VIFs for the diabetes predictors. Which are collinear, and why might that be, given what s1–s4
   measure (cholesterol components)? Refit with robust (HC3) SEs and compare.
6. **(P)** Use a nested F-test to check whether s1, s2, s4 and s6 together add anything to a model with age, sex, bmi, bp,
   s3 and s5.
7. **(P)** Implement `vif(X)` in `statkit/regression.py` (X without the constant column; for each column j, regress it on
   the others plus an intercept with your `fit_ols` and return 1/(1 − R_j²)). Check against statsmodels on the diabetes data.
8. **(P)** Take the students data, add one point (study 30 h, score 100, consistent with the trend) and refit. Then add
   (study 30 h, score 30) instead. Compare leverage, residuals and Cook's distance for the two points. Which one changes
   the slope?

## ★★★ Challenge

9. **(M)** Show that the hat matrix H = X(XᵀX)⁻¹Xᵀ is symmetric and idempotent (H² = H), and that its trace equals p + 1.
   Why does that make the average leverage (p + 1)/n?
10. **(P)** **Clustered data.** Simulate 30 classrooms of 20 students: a classroom effect ~ N(0, 5²), a treatment assigned
    to whole classrooms (15 treated) with no true effect, and individual noise ~ N(0, 10²). Run OLS of score on treatment
    1,000 times. How often does it reject at α = 0.05? Repeat with cluster-robust SEs
    (`fit(cov_type="cluster", cov_kwds={"groups": classroom})`). Explain.
11. **(P)** Fit `revenue ~ pre_spend` on `data/ab_test.csv` for converters only. Diagnose the residuals, then try
    log(revenue). Which model satisfies the assumptions better?

Solutions: [solutions/solutions.md](solutions/solutions.md)
