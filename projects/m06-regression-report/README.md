# Mini-project m06 — A regression modelling report

**Module 06 · week 19 · about 7 hours** · Branch: `git switch -c project/m06-regression-report`

## Goal

Build, check and explain regression models on two datasets: one where you know the truth (so you can test your
methods), and one real dataset (where you can't).

## Part A — recover the truth (2 hours)

Using `data/students.csv`, *without* looking at `data/make_students.py`:

1. Explore and propose a model for `exam_score` (Module 02 tools and residual plots).
2. Fit it, run the diagnostics of Lesson 06.03, and report each coefficient with a 95% CI.
3. Check for curvature (splines) and interactions you think are plausible, using F-tests or likelihood-ratio tests.
4. **Then** open `make_students.py`. Which true coefficients fall inside your intervals? Did you include any variable that
   doesn't matter, or miss one that does? Write a paragraph on what you learned.

## Part B — disease progression (4 hours)

Continue from your m02 EDA on the scikit-learn diabetes data (`load_diabetes(scaled=False, as_frame=True)`).

1. **Explanatory model.** Fit OLS with all ten predictors. Report the diagnostics, VIFs and robust SEs. Which conclusions
   about individual predictors are trustworthy, and which are undermined by collinearity?
2. **Predictive model.** Hold out 25% as a test set (seed 0). On the training set, compare by 10-fold CV: OLS, ridge, lasso,
   elastic net, and a model with spline terms for bmi and bp. Use pipelines so that scaling and spline bases are learned
   inside each fold.
3. Choose one model *using CV only*, then evaluate it **once** on the test set. Report the test RMSE with a bootstrap 95%
   interval (Lesson 05.04).
4. **Interpretation.** Present the chosen model's effects in a reader-friendly way: for example, the predicted change in
   progression when bmi rises from its 25th to its 75th percentile, holding the others at their medians.

## Part C — the write-up (1 hour)

`report.md` (from `templates/project-report.md`), with separate sections for *explanation* and *prediction*. Make clear
which claims are associations and which, if any, could be causal.

## What "done" looks like

- [ ] Every coefficient is reported with an interval, and every model choice is justified (by CV, diagnostics or theory)
- [ ] The test set is used exactly once (`git log` should show the test evaluation in a single late commit)
- [ ] Diagnostic plots are included for the final explanatory model
- [ ] Merged into `main`; tag `m06-done` after the checkpoint
