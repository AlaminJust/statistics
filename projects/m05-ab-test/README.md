# Mini-project m05 — Analyse an A/B test and write a decision memo

**Module 05 · week 15 · about 6 hours** · Branch: `git switch -c project/m05-ab-test`

## Situation

An online shop tested a new checkout page. 20,000 visitors were randomly assigned to the current page (A) or the new
one (B). The product manager wants a recommendation: **ship B or not?** Data: `data/ab_test.csv` (synthetic, made by
`data/make_ab_test.py`; don't read that script until you've finished, since it contains the true answer).

| Column | Meaning |
| --- | --- |
| user_id | visitor |
| variant | A (control) or B (new page) |
| device | mobile or desktop |
| pre_spend | the visitor's spend in the 30 days before the test |
| converted | 1 if they bought during the test |
| revenue | amount spent during the test (0 if no purchase) |

## Before you look at the results (commit this first)

Write `projects/m05-ab-test/PLAN.md` and commit it **before** running any comparison between A and B:

- the primary metric (conversion) and one secondary metric (revenue per visitor);
- the test you'll use for each, and α;
- the smallest lift that would matter to the business (say 0.5 percentage points);
- what power the test had to detect that lift with these sample sizes (Lesson 05.07).

This mimics pre-registration (Module 09), and your git history proves you did it first.

## Analysis

1. **Sanity checks.** Is the split close to 50/50? (χ² goodness of fit, the "sample ratio mismatch" check.) Are the groups
   balanced on device and pre_spend, as randomisation should make them?
2. **Primary metric.** Conversion in each arm with Wilson intervals, the difference with a 95% interval, and a test
   (two-proportion z-test or χ²). Compute the relative lift too.
3. **Secondary metric.** Revenue per visitor is very skewed (mostly zeros). Compare the arms with a Welch t-test **and** a
   bootstrap interval for the difference in means. Do they agree?
4. **Revenue per buyer.** Among converters only, did B change the order value? Explain why this comparison is *not*
   protected by randomisation in the same way (who converts may differ between arms).
5. **Subgroups.** Look at the effect on mobile and desktop separately. Apply a multiple-testing correction, and be careful
   not to over-interpret (Lesson 05.08, exercise 6).
6. **Peeking.** Suppose the team had checked the p-value every 1,000 users and stopped at the first p < 0.05. Simulate
   what that would have done to the false-positive rate for a test of this size.

## Deliverable: a one-page decision memo (`memo.md`)

Written for the product manager, not for statisticians:

- the recommendation in the first sentence;
- the effect on conversion as a number with an interval (e.g. "+0.9 points, 95% CI 0.3 to 1.6");
- what it means in money per 10,000 visitors (with its uncertainty);
- the risks and limitations, in plain words;
- a technical appendix with the tests, assumptions and code references.

## What "done" looks like

- [ ] `PLAN.md` committed before the analysis (check `git log --follow projects/m05-ab-test/PLAN.md`)
- [ ] `analysis.py` reproduces every number in the memo
- [ ] Every estimate has an interval; every test is named
- [ ] After finishing, compare your conclusion with the truth in `data/make_ab_test.py`, and write down what you'd have
      got wrong if you'd peeked
- [ ] Merged into `main`; tag `m05-done` after the checkpoint
