# Mini-project m07 — A Bayesian analysis report

**Module 07 · week 23 · about 6 hours** · Branch: `git switch -c project/m07-bayesian-report`

## Part A — the A/B test, the Bayesian way (2 hours)

Re-analyse `data/ab_test.csv` (your m05 project):

1. Beta–Binomial model for conversion in each arm. Report P(B > A), the posterior of the lift, P(lift > 0.5 points) and
   the expected loss of each decision.
2. A Bayesian logistic regression in PyMC: `converted ~ variant + device + pre_spend` (standardise pre_spend). Report the
   posterior of the variant coefficient and convert it to a lift in conversion probability for a typical mobile user.
3. A hierarchical model for the treatment effect by device (Lesson 07.03, exercise 11). What can and can't two segments
   tell you?
4. Compare your conclusions with the m05 frequentist memo. Where do they agree? What does each framework let you say
   that the other doesn't?

## Part B — a hierarchical model of hospital performance (3 hours)

Simulate a realistic dataset with `projects/m07-bayesian-report/make_hospitals.py` (write it yourself):

- 60 hospitals with between 15 and 1,500 surgeries each (log-uniform);
- true complication rates from Beta(12, 188) (about 6% on average);
- one hospital-level predictor, "high volume" (more than 500 surgeries), which lowers the log-odds of complications by 0.3.

Then:

1. Fit no-pooling, complete-pooling and partial-pooling (hierarchical logistic) models in PyMC, the last including the
   volume predictor.
2. Run the full workflow: prior predictive check, diagnostics (R̂, ESS, divergences; use the non-centred form),
   posterior predictive check of the spread of hospital rates, LOO comparison of models with and without the volume
   predictor.
3. Produce a "league table" with posterior means and 90% intervals, and compare it with a table of raw rates. Which
   hospitals would be wrongly flagged as "worst" by the raw table?
4. Because you simulated the data, check how often each method's interval covers the true rate.

## Part C — the report (1 hour)

`report.md`: methods (with priors justified), diagnostics, results, the league-table figure, a prior-sensitivity
paragraph, and limitations.

## What "done" looks like

- [ ] Every PyMC model reports R̂ < 1.01, ESS > 400 and 0 divergences (or explains why not)
- [ ] At least one prior predictive and one posterior predictive check, shown as figures
- [ ] Interval coverage in Part B is computed against the simulated truth
- [ ] Merged into `main`; tag `m07-done` after the checkpoint
