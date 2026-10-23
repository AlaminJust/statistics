# Mini-project m09 — Does job training raise earnings?

**Module 09 · week 31 · about 7 hours** · Branch: `git switch -c project/m09-causal-analysis`

## Situation

A government agency offered a voluntary job-training programme. You have observational data on 8,000 adults:
`data/job_training.csv` (synthetic, made by `data/make_job_training.py`; **don't read that script until you've
finished**, since it contains the true effects).

| Column | Meaning |
| --- | --- |
| age, educ, female, region | characteristics before the programme |
| unemployed | unemployed at the start (1/0) |
| earnings_before | annual earnings in the year before |
| trained | joined the programme (1/0), self-selected |
| earnings_after | annual earnings two years later |

## Steps

1. **Pre-register** (commit `PLAN.md` first): the estimand (ATE or ATT?), the DAG you assume, the adjustment set, the
   methods you'll use, and the primary analysis.
2. **Describe the selection.** Who joins? Compare covariates by treatment with standardised mean differences.
3. **Naive estimate**, and why it's misleading.
4. **Regression adjustment**, with a sensible functional form (age², interactions?), and robust SEs.
5. **Propensity scores:** fit a model, plot the overlap, and report balance (SMDs) after weighting. Estimate the ATE by
   IPW and the ATT by matching.
6. **Doubly robust (AIPW)**, with flexible models (gradient boosting for both the outcome and the propensity, with
   cross-fitting: fit the models on one half and predict on the other, then swap). Bootstrap the SE.
7. **Difference-in-differences** using earnings_before and earnings_after. What does it assume here, and is that
   plausible? (Hint: does everyone's earnings grow at the same rate?)
8. **Heterogeneity.** Estimate the effect separately for unemployed vs employed participants, and across education levels.
9. **Sensitivity analysis.** How strong would an unmeasured confounder need to be to explain away your main estimate?
   (Use an E-value or a simulation like Lesson 09.03, exercise 6.)
10. **Only then** read `make_job_training.py`. Which methods got close to the truth, and why did the others fail?

## Deliverable: `report.md`

The report should cover: the question and estimand; the DAG; a results table with every method's estimate and 95% CI;
balance and overlap figures; heterogeneity; sensitivity; a recommendation to the agency; and a "post-mortem" against the
known truth.

## What "done" looks like

- [ ] `PLAN.md` committed before any treatment–outcome comparison
- [ ] At least four estimators, each with an interval and its key assumption stated
- [ ] Overlap and balance shown as figures
- [ ] Post-mortem explains each method's error relative to the truth
- [ ] Merged into `main`; tag `m09-done` after the checkpoint
