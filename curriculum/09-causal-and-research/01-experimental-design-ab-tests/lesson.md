# 09.01 — Experimental design and A/B tests

**Time:** about 5 hours · **Prerequisites:** 05.03–05.08, 07.01

## Why this matters

A **randomised experiment** is the most reliable way to learn what *causes* what. Randomisation makes the groups
comparable in every respect, measured or not, so any difference in outcomes can be attributed to the treatment.
Technology companies run thousands of A/B tests a year, medicine runs clinical trials, and ML research runs ablations.
But experiments are easy to ruin: wrong sample sizes, broken randomisation, peeking at results, analysing at the wrong
level. This lesson covers how to design an experiment before running it, and how to analyse it so that its conclusions
hold.

## Learning objectives

- Write an experiment plan: hypothesis, primary metric, unit, sample size, duration, stopping rule.
- Compute sample sizes for conversion tests, and understand why small lifts are expensive.
- Run sanity checks: sample ratio mismatch and covariate balance.
- Explain the peeking problem and use a sequential boundary.
- Reduce variance with CUPED and with blocking.
- Match the unit of randomisation to the unit of analysis.

## Before you start

Your team checks an A/B test's p-value every day for 20 days and stops as soon as it drops below 0.05. If the new
version is in fact *identical* to the old one, how often will they declare a winner?

---

## 1. Why randomise?

If users choose the treatment themselves (opt in to a feature), the people who choose it differ from those who don't
(Lesson 02.01: confounding). **Random assignment breaks every link between the treatment and the users'
characteristics**, measured or unmeasured, so the groups differ only by chance and by the treatment. The difference in
means is then an unbiased estimate of the causal effect (Lesson 09.02 makes this precise).

## 2. The experiment plan (write it before launching)

| Item | Example |
| --- | --- |
| hypothesis | the new checkout page increases conversion |
| **primary metric** (one!) | conversion rate per visitor |
| guardrail metrics | page load time, refund rate, support tickets |
| unit of randomisation | visitor (cookie or account) |
| minimum effect that matters | +0.5 percentage points |
| α, power → sample size | 0.05, 0.8 → n per arm |
| duration | whole weeks (weekday/weekend cycles), long enough for n |
| stopping rule | fixed horizon, or a pre-specified sequential design |
| analysis | two-proportion test with a Wilson CI; CUPED with pre-period conversion |

Commit this plan to git before launching (as in the m05 project). It's your pre-registration.

## 3. Sample size: small lifts are expensive

**Worked example 1** (α = 0.05, power 0.8):

| Baseline → target | Relative lift | Users per arm |
| --- | --- | --- |
| 10% → 12% | +20% | 3,841 |
| 10% → 11% | +10% | 14,751 |
| 10% → 10.5% | +5% | **57,763** |
| 2% → 2.2% | +10% | **80,682** |

Halving the detectable lift quadruples the sample size (Lesson 05.07), and low baseline rates make everything harder.
If you can't reach the sample size, the honest options are: test a bolder change, use a more sensitive metric, reduce
variance (sections 6–7), or don't run the test.

## 4. Sanity checks before reading results

**Sample ratio mismatch (SRM):** if you intended a 50/50 split, test the observed counts with a χ² goodness-of-fit test.
In `data/ab_test.csv`: A = 9,934 and B = 10,066, p = **0.35**, fine. A split of 10,300 vs 9,700 would give p = 2 × 10⁻⁵.
An SRM means something is broken (a redirect losing users, bots, a logging bug), and the results can't be trusted
whatever they show.

**Covariate balance:** pre-experiment variables should look the same in both arms. pre_spend has means 40.48 vs 39.81, a
standardised difference of −0.02 (anything below about 0.1 is fine). The mobile share is 61.8% vs 62.3%.

## 5. The peeking problem

**Worked example 3 (Before you start).** 4,000 simulated A/A tests (no true difference), checked daily for 20 days:

| Rule | False-positive rate |
| --- | --- |
| stop at the first daily p < 0.05 | **25.5%** |
| look once, at day 20 | 4.9% |
| daily looks with a **Pocock boundary** p < 0.0076 | 5.4% |

Every look is another chance for noise to cross 0.05 (Lesson 05.08). Answer: about a quarter of identical versions would
be declared winners. Solutions:

- **fixed horizon:** decide n in advance and analyse once;
- **group-sequential designs:** look at a few planned times with stricter thresholds. For 20 equally spaced looks, the
  Pocock constant is z = 2.67, i.e. p < 0.0076 at each look. O'Brien–Fleming boundaries are very strict early and close
  to 0.05 at the end;
- **always-valid methods** (mSPRT, confidence sequences), designed for continuous monitoring.

Peeking is fine for *safety* (stop if something breaks); it's dangerous for declaring winners.

## 6. CUPED: free power from pre-experiment data

Outcomes often correlate with users' behaviour *before* the experiment. Subtract the predictable part:

$$Y^{\text{cuped}} = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)},$$

where X is the pre-period value of the metric (or any pre-treatment covariate). Because X is measured before
randomisation, it is independent of the treatment, so the adjustment doesn't bias the estimate; it only removes noise.
Variance falls by a factor of 1 − r², where r = Corr(X, Y). It's equivalent to regression adjustment for X.

**Worked example 4.**

| Metric | r | SE raw | SE CUPED |
| --- | --- | --- | --- |
| revenue per user in `ab_test.csv`, X = pre_spend | 0.09 | 0.280 | 0.279: almost no gain |
| simulated weekly sessions, X = pre-period sessions | **0.72** | 0.0616 | **0.0424**: p falls from 0.009 to 0.0003 |

CUPED helps in proportion to r², so choose X carefully: the same metric measured before the experiment is usually the
best covariate.

## 7. Blocking (stratification)

Randomise *within* groups of similar units, for example pairs matched on a strong predictor of the outcome. **Worked
example 5:** 200 users, outcome strongly related to pre-period spend, true effect 1.0. Over 2,000 simulated experiments:

| Design | Mean estimate | SD of the estimate |
| --- | --- | --- |
| complete randomisation | 0.96 | **2.15** |
| matched-pair blocking | 1.00 | **0.76** |

Same truth, a third of the noise (`blocking.png`). "Block what you can, randomise what you cannot." In online experiments
CUPED achieves much the same after the fact. In small experiments (classrooms, clinics, cities), blocking at the design
stage is essential.

## 8. The unit of randomisation must match the unit of analysis

If you randomise **classrooms** but analyse **students** as if they were independent, you'll badly overstate certainty:
in Lesson 06.03, exercise 10, a null treatment assigned to whole classrooms was "significant" 38% of the time. Analyse at
the level of randomisation (classroom means), or use cluster-robust SEs or mixed models. Online equivalents: randomising
by session but measuring per user; marketplaces and social networks where treated and control users interact
(interference). Those need cluster or switchback designs.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Deciding n after looking | A power calculation before launch |
| Many primary metrics | One primary metric, plus guardrails |
| Peeking and stopping at p < 0.05 | Fixed horizon or sequential boundaries |
| Ignoring a sample ratio mismatch | Treat it as a bug and investigate before analysing |
| Students analysed as independent when classrooms were randomised | Analyse at the randomisation unit |
| Stopping on a Tuesday | Whole weeks, to cover weekly cycles |

## Summary

- Randomisation balances measured and unmeasured confounders.
- Plan first: metric, unit, minimum effect, sample size, stopping rule. Commit the plan.
- Check sample ratio mismatch and balance before reading results.
- Peeking inflates false positives (25% with 20 looks); use a fixed horizon or sequential boundaries.
- CUPED and blocking cut variance by about r² for free.
- Analyse at the unit of randomisation.

## Self-test

1. How many users per arm for 5% → 6% conversion at α = 0.05 and 80% power?
2. You expected a 50/50 split but got 52/48 with 100,000 users. What do you do?
3. Why doesn't CUPED bias the treatment effect estimate?
4. A school district randomises 10 schools to a new curriculum. What is the effective sample size, roughly?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
