# 06.04 Exercises — Logistic regression and GLMs

`make lesson L=06-regression/04-logistic-regression-and-glms`

## ★ Warm-up

1. **(M)** A logistic coefficient is −0.7. What is the odds ratio? Interpret it for a one-unit increase in x.
2. **(M)** Using the lesson's model, compute by hand P(high) for a tutored student who studies 12 hours:
   σ(−5.63 + 0.410 × 12 + 1.285).
3. **(C)** Explain to a doctor why an odds ratio of 2 doesn't mean "twice the risk" when the baseline risk is 40%.

## ★★ Core

4. **(P)** Fit `high ~ study_hours + sleep_hours + tutored + C(program)` (logistic). Which terms are significant? Then
   test program as a whole with a likelihood-ratio test. Compare with how the data were generated, and explain any
   surprise.
5. **(P)** On `data/ab_test.csv`, fit a logistic regression of `converted` on the variant, device and pre_spend. Report
   the odds ratio for B with its CI. Compare it with the unadjusted odds ratio and with the simple ratio of conversion
   rates. Why are they so close here?
6. **(P)** Implement `fit_logistic_gd(X, y, learning_rate=0.1, steps=5000)` in `statkit/regression.py` (gradient descent on
   the average NLL; X includes the column of 1s; standardise features before calling it). Compare its coefficients with
   your own IRLS version and with statsmodels.
7. **(P)** Plot the predicted probability curves for tutored and non-tutored students (as in `examples.py`), and add the
   observed proportion of high scores in 2-hour bins of study time. Does the logistic curve fit the bins well?
8. **(M)** Show that for logistic regression the gradient of the log-likelihood is Xᵀ(**y** − **p**). (Hint: use
   σ′ = σ(1 − σ) from Lesson 01.03.)

## ★★★ Challenge

9. **(P)** **Separation.** Create a dataset where x > 5 always gives y = 1 and x ≤ 5 always gives y = 0. Try to fit a
   logistic regression with statsmodels. What happens? Then fit scikit-learn's `LogisticRegression` (which is regularised
   by default) and explain the difference.
10. **(P)** **Poisson vs negative binomial.** Simulate counts for 500 customers whose purchase rates vary (gamma
    distributed, mean 3, shape 1.5) and depend on a loyalty flag (rate × 1.5 for members). Fit Poisson and negative
    binomial GLMs. Compare the coefficient, SE and dispersion. Which CI covers the true log rate ratio, ln 1.5?
11. **(M)** For a single binary predictor x (0/1), show that the logistic MLE of β₁ equals the log of the sample odds
    ratio from the 2 × 2 table. Verify on tutoring × high.

Solutions: [solutions/solutions.md](solutions/solutions.md)
