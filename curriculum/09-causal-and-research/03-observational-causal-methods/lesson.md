# 09.03 — Estimating causal effects from observational data

**Time:** about 6 hours · **Prerequisites:** 09.02, 06.02, 06.04, 05.04

## Why this matters

Experiments are often impossible: you can't randomise people to smoke, to go to college, or to live under a new law.
Observational causal inference estimates effects from data where treatment wasn't randomised, by **adjusting for
confounders** (regression, propensity scores, matching, weighting, doubly robust methods) or by finding **natural
experiments** (difference-in-differences, instrumental variables, regression discontinuity). Each method rests on an
assumption that the data can't fully check, so the skill is knowing which assumption you're making and how to test its
consequences.

## Learning objectives

- Estimate effects by regression adjustment, propensity-score weighting, matching and AIPW, and check covariate balance.
- Diagnose overlap (positivity) problems.
- Apply difference-in-differences and check parallel trends.
- Estimate effects with an instrumental variable (2SLS), and check instrument strength.
- Estimate effects at a cutoff with regression discontinuity.
- State the key assumption of each method in one sentence.

## Before you start

People who joined a job-training programme earn **$3,600 less** afterwards than people who didn't. Does the programme
hurt earnings?

---

## Part 1: adjusting for measured confounders

All methods in this part assume **no unmeasured confounding**: given the measured covariates X, treatment is as good as
random.

**The simulated data** (in the spirit of LaLonde, 1986): 5,000 people. Poorer, younger and less-educated people are more
likely to join the programme. **True effect: +$1,800 for everyone.** Naive difference: **−$3,605**. That answers "Before
you start": the participants were poorer to begin with. The naive comparison is confounded.

### 1. Regression adjustment

Regress the outcome on treatment and confounders: y ~ t + age + educ + prior. Coefficient on t: **+$1,859** (95% CI
+$1,649 to +$2,069). It works well *here because the outcome model is correctly specified* (linear in these covariates).
If the true relationship were non-linear, or the groups barely overlapped, regression would extrapolate silently.

### 2. Propensity scores

The **propensity score** e(x) = P(T = 1 | X = x) summarises all the confounders in one number. Treated and control units
with the same e(x) are comparable (Rosenbaum and Rubin, 1983). Estimate it with logistic regression (or any classifier).

**Check overlap first** (`observational_methods.png`, left). Here the scores range from 0.005 to 0.946 among the treated
and from 0.001 to 0.897 among controls. Some treated people have almost no comparable controls, and some controls almost
never get treated. That's a **positivity** problem.

### 3. Inverse probability weighting (IPW)

Weight treated units by 1/e(x) and controls by 1/(1 − e(x)). The weighted sample looks as if treatment had been
randomised:

$$\widehat{\text{ATE}}_{\text{IPW}} = \frac{\sum_{T=1} y_i/e_i}{\sum_{T=1} 1/e_i} - \frac{\sum_{T=0} y_i/(1 - e_i)}{\sum_{T=0} 1/(1 - e_i)}.$$

**Balance check:** standardised mean differences before → after weighting: age −0.20 → +0.07, education −0.44 → +0.05,
prior earnings **−1.27 → +0.10**. Weighting balanced the covariates (target |SMD| < 0.1).

Estimate: **+$2,232**, and the largest weight is 208: a single person counts as 208 people. **Clipping** the propensities
to [0.05, 0.95] limits the weights to 20 but gives **+$1,325**: less variance, but biased, because the estimand quietly
changes towards the region of overlap.

### 4. Matching

For each treated person, find the control with the closest propensity score (or closest covariates). That estimates the
effect on the treated (ATT). Matched ATT: **+$1,672**. Matching is transparent and easy to explain. Check balance after
matching, as with weighting.

### 5. Doubly robust estimation (AIPW)

Combine an outcome model μ_t(x) with the propensity model:

$$\widehat{\text{ATE}}_{\text{AIPW}} = \frac{1}{n}\sum_i \Big[\hat\mu_1(x_i) - \hat\mu_0(x_i) + \frac{t_i(y_i - \hat\mu_1(x_i))}{e_i} - \frac{(1 - t_i)(y_i - \hat\mu_0(x_i))}{1 - e_i}\Big].$$

It's consistent if **either** model is right. Estimate **+$1,664**. With a useless propensity model but the right outcome
model: +$1,954. With the right propensity model but a useless outcome model: +$2,445. Both are still in the right
region, though the second inherits IPW's noise from the extreme weights.

### How noisy is each estimator?

200 bootstrap resamples:

| Estimator | Bootstrap mean | SE |
| --- | --- | --- |
| regression adjustment | +$1,857 | **98** |
| IPW | +$2,227 | 353 |
| AIPW | +$1,671 | 256 |

With poor overlap, weighting estimators are noisy. Regression wins here because its model happens to be exactly right,
which you can never be sure of in real data. AIPW is the standard compromise, especially with flexible ML models for
μ and e: that's **double/debiased machine learning** (with cross-fitting).

**All of Part 1 fails if an important confounder is unmeasured.** Always report a **sensitivity analysis**: how strong
would an unmeasured confounder have to be to explain away the effect? (E-values, or Rosenbaum bounds.)

## Part 2: natural experiments

### 6. Difference-in-differences (DiD)

Compare the *change* over time in treated units with the change in untreated units:

$$\text{DiD} = (\bar{y}_{T,\text{after}} - \bar{y}_{T,\text{before}}) - (\bar{y}_{C,\text{after}} - \bar{y}_{C,\text{before}}).$$

**Key assumption: parallel trends.** Without the policy, treated and control units would have changed by the same amount.
Level differences are fine; different *trends* are not.

**Worked example 7.** 50 regions, a policy in 20 of them. Policy regions start 12 units higher; every region grows by 5;
the true policy effect is +4.

| | before | after |
| --- | --- | --- |
| other regions | 97.57 | 102.51 |
| policy regions | 114.24 | 123.10 |

The naive after-period comparison is +20.6 (it includes the pre-existing gap). DiD: (123.10 − 114.24) − (102.51 − 97.57) =
**+3.93**. As a regression, y ~ policy × post, the interaction is +3.93 with a cluster-robust SE of 0.79 (cluster by region,
Lesson 06.03). Check parallel trends with several pre-periods ("event-study" plots). Staggered adoption across many periods
needs modern estimators.

### 7. Instrumental variables (IV)

An **instrument** Z affects the treatment X, and affects Y **only through X** (the exclusion restriction), and is unrelated
to the unmeasured confounders. Then

$$\hat\beta_{\text{IV}} = \frac{\operatorname{Cov}(Y, Z)}{\operatorname{Cov}(X, Z)}, \quad\text{or two-stage least squares: regress } X \text{ on } Z, \text{ then } Y \text{ on } \hat{X}.$$

**Worked example 8: the return to education.** Unobserved ability raises both education and wages. True return: 8% per
year. The instrument is distance to the nearest college.

| Method | Estimate |
| --- | --- |
| OLS | **0.160**, biased upward (ability) |
| first stage: education on distance | slope −0.050, F = 5,292 (a strong instrument) |
| **2SLS** | **0.076** (and the Wald ratio gives the same) |

Rule of thumb: a first-stage F below about 10 means a **weak instrument** and unreliable, biased IV estimates. The
exclusion restriction can't be tested directly. Here, distance could also matter if living near a college correlates with
living in a city with higher wages, which is why real studies adjust for region. Run the second stage by hand only to
learn: the SEs from a manual second stage are wrong, so use a proper IV routine (e.g. `linearmodels`' IV2SLS) in practice.

### 8. Regression discontinuity (RD)

When treatment switches on at a **cutoff** of a running variable (a scholarship for entrance scores ≥ 70), units just
above and just below the cutoff are nearly identical, so the jump in the outcome at the cutoff is the causal effect
*there*.

**Worked example 9.** True effect on later GPA: +0.25. The naive scholars-vs-others comparison gives +0.85 (better
students get scholarships). Local linear regression on each side of the cutoff:

| Bandwidth | Jump at 70 | SE | n |
| --- | --- | --- | --- |
| ±5 | +0.244 | 0.045 | 681 |
| ±10 | +0.248 | 0.032 | 1,351 |
| ±20 | +0.261 | 0.023 | 2,682 |

A narrower window means less bias and more variance; report several bandwidths. Check that units can't manipulate the
running variable precisely (look for a jump in the *density* of scores at the cutoff), and that pre-treatment covariates
don't jump. RD identifies the effect only near the cutoff.

## Summary of assumptions

| Method | Key assumption |
| --- | --- |
| regression, matching, IPW, AIPW | no unmeasured confounding, plus overlap |
| difference-in-differences | parallel trends without the treatment |
| instrumental variables | the instrument affects Y only through X; relevant (strong); as good as random |
| regression discontinuity | no precise manipulation at the cutoff; continuity of potential outcomes there |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Skipping the overlap and balance checks | Plot propensities; report SMDs |
| Extreme IPW weights left unexamined | Report the weights; consider AIPW, overlap weights or trimming (and say what changes) |
| DiD without checking pre-trends | Show several pre-periods |
| A weak instrument (F < 10) | Find a stronger instrument, or use weak-IV-robust methods |
| Extrapolating an RD effect far from the cutoff | It's local to the cutoff |
| No sensitivity analysis | Quantify how strong a hidden confounder would need to be |

## Self-test

1. What does a propensity score of 0.95 mean, and why are such units a problem for IPW?
2. Why is AIPW called doubly robust?
3. State the parallel-trends assumption in plain words.
4. Why is distance to a college a plausible but imperfect instrument for education?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
