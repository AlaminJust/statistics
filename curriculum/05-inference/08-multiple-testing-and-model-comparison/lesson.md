# 05.08 — Multiple testing and comparing ML models

**Time:** about 5 hours · **Prerequisites:** 05.04–05.07

## Why this matters

Every additional test is another chance for a false positive. Test 20 features, 20 subgroups or 20 hyperparameter
settings at α = 0.05, and you'll probably "discover" something that isn't there. That's how genomics, psychology and
ML benchmarks have produced so many results that didn't replicate. This lesson covers the corrections (Bonferroni, Holm,
Benjamini–Hochberg), the subtler forms of the problem (p-hacking and optional stopping), and the correct ways to compare
two ML models on the same data.

## Learning objectives

- Compute the family-wise error rate (FWER) and control it with Bonferroni and Holm.
- Control the false discovery rate (FDR) with Benjamini–Hochberg, and explain when FDR is the right target.
- Recognise p-hacking: optional stopping, many outcomes, flexible analysis choices.
- Compare two classifiers on one test set with McNemar's test and a paired bootstrap.
- Explain why naive t-tests across cross-validation folds are overconfident.

## Before you start

You test 20 features for association with an outcome, and none of them truly matters. What's the probability that at
least one test comes out "significant" at α = 0.05?

---

## 1. The problem: false positives add up

With m independent tests of true null hypotheses at level α,

$$P(\text{at least one false positive}) = 1 - (1 - \alpha)^m.$$

| m | 1 | 5 | 10 | **20** | 100 |
| --- | --- | --- | --- | --- | --- |
| FWER | 0.05 | 0.23 | 0.40 | **0.64** | 0.994 |

Answer to "Before you start": **64%**. In `examples.py`, 100 t-tests on pure noise produce 4 "significant" results.

## 2. Controlling the family-wise error rate (FWER)

**FWER** = P(at least one false positive among all the tests).

**Bonferroni:** test each hypothesis at α/m (equivalently, multiply each p by m, capped at 1). It guarantees FWER ≤ α
for any dependence between the tests, by the union bound: P(any false positive) ≤ Σ P(each one) = m × α/m = α.
Simple and safe, but conservative when m is large.

**Holm** (step-down; always at least as powerful as Bonferroni, with the same guarantee):

1. Sort the p-values: p₍₁₎ ≤ … ≤ p₍ₘ₎.
2. Compare p₍₁₎ with α/m, p₍₂₎ with α/(m − 1), …, p₍ₖ₎ with α/(m − k + 1).
3. Reject until the first failure, then stop.

**Use FWER control when even one false positive is costly**: confirmatory clinical trials, or a small number of
pre-planned comparisons.

## 3. Controlling the false discovery rate (FDR)

When you screen many hypotheses (genes, features, A/B metrics) and will follow up on the hits, demanding *zero* false
positives is too strict. The **FDR** is the expected share of false discoveries among everything you reject.

**Benjamini–Hochberg (BH)** at level q:

1. Sort the p-values: p₍₁₎ ≤ … ≤ p₍ₘ₎.
2. Find the largest k with **p₍ₖ₎ ≤ (k/m) q**.
3. Reject the k hypotheses with the smallest p-values.

**Worked example 3.** Ten p-values: 0.001, 0.008, 0.012, 0.039, 0.041, 0.060, 0.074, 0.205, 0.212, 0.906 (q = α = 0.05).

| Method | Thresholds | Rejected |
| --- | --- | --- |
| none | 0.05 each | 5 |
| Bonferroni | 0.005 each | **1** |
| Holm | 0.005, 0.0056, 0.0063, … | **1** (p₍₂₎ = 0.008 > 0.0056) |
| BH | 0.005, 0.010, **0.015**, 0.020, … | **3** (0.012 ≤ 0.015; 0.039 > 0.020) |

`statsmodels.stats.multitest.multipletests(p, alpha=0.05, method="fdr_bh")` returns the decisions and adjusted p-values.

**Worked example 4: what FDR control buys.** Simulate 1,000 tests, 900 true nulls and 100 real effects, 200 times:

| Rule | Share of discoveries that are false | Share of real effects found |
| --- | --- | --- |
| p < 0.05, no correction | **34%** | 86% |
| BH, q = 0.05 | **4.4%** | 50% |

Without correction, a third of the "discoveries" are false. BH keeps them near 5%, at the cost of finding fewer real
effects. That's the trade-off.

## 4. Hidden multiplicity: p-hacking

Multiple testing doesn't need many explicit tests. It also comes from flexible choices made after seeing data:

| Practice | Why it inflates false positives |
| --- | --- |
| **optional stopping**: checking the p-value repeatedly and stopping when it drops below 0.05 | many looks = many tests |
| trying several outcome measures and reporting the one that "worked" | same |
| adding or removing covariates, outliers or subgroups until p < 0.05 | the "garden of forking paths" |
| switching to a one-sided test after seeing the direction | halves p after the fact |
| hyperparameter search judged on the test set | the test set becomes a training set |

**Worked example 5: optional stopping.** With no true effect, peek after every 10 subjects per group (up to 200) and stop
as soon as p < 0.05. The false-positive rate is **24.8%**, five times the nominal 5%.

**Defences:** pre-register the hypothesis, primary outcome, sample size and analysis plan before seeing data (Module 09);
use sequential-testing methods designed for peeking; report everything you tried; keep a held-out test set you look at
once; and replicate.

## 5. Comparing two ML models on the same test set

The two models' predictions on the same examples are **paired**, so treat them that way.

**Worked example 6.** Logistic regression vs a depth-3 decision tree on the breast-cancer data, test set n = 171:
accuracies 0.959 and 0.901.

|  | tree right | tree wrong |
| --- | --- | --- |
| **logistic right** | 153 | **11** |
| **logistic wrong** | **1** | 6 |

Examples both models got right or both got wrong say nothing about which is better. Only the **12 disagreements**
matter: logistic alone was right on 11, the tree alone on 1.

**McNemar's test** asks whether the disagreements split 50/50: an exact binomial test of 11 vs 1, **p = 0.006**.
The **paired bootstrap** (resample test examples, recompute both accuracies, take the difference) gives a 95% interval for
the difference of **[0.023, 0.099]**. Both say logistic regression is genuinely better on this population of test cases.

Compare with the wrong analysis: treating the two accuracies as independent proportions (Lesson 05.03, exercise 6). Their
separate intervals overlap, which wrongly suggests the comparison is unclear.

## 6. Comparing models with cross-validation: be careful

It's tempting to run a paired t-test on the 10 fold-wise score differences from 10-fold CV. **Don't**: the folds share
about 90% of their training data, so the differences are correlated, not independent, and the naive standard error is
far too small.

**Worked example 7.** Two models with identical true accuracy. Each fold difference = a component shared by all folds
(the luck of this particular dataset) + independent fold noise. A naive t-test on the 10 differences rejects the true
null **30%** of the time.

Better options:

- the **corrected resampled t-test** (Nadeau and Bengio, 2003), which inflates the variance by (1/k + n_test/n_train);
- **5 × 2 CV** (Dietterich, 1998);
- for many datasets, the **Wilcoxon signed-rank test** across datasets, or a Friedman test with post-hoc comparisons
  (Demšar, 2006);
- the gold standard: an untouched test set, or repeating the whole experiment on fresh data.

Finally, report variability across **random seeds** (Module 09): with deep learning, seed-to-seed differences are often as
large as the improvement being claimed.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Testing 20 things at 0.05 and reporting the hits | Correct for multiplicity, or pre-register one primary test |
| Bonferroni for large-scale screening | BH (FDR) keeps much more power |
| Peeking at an A/B test daily | Fix n in advance, or use sequential methods |
| Comparing models with two separate intervals | Paired: McNemar or a paired bootstrap |
| t-test across CV folds | Corrected resampled t-test or 5 × 2 CV |

## Summary

- m tests at α give FWER = 1 − (1 − α)ᵐ: 64% for 20 tests.
- FWER control: Bonferroni (α/m) and Holm (step-down, more powerful).
- FDR control: Benjamini–Hochberg; reject p₍ₖ₎ ≤ (k/m)q.
- p-hacking (peeking, flexible choices) is hidden multiple testing; pre-register.
- Same test set: McNemar or a paired bootstrap. Across CV folds: corrected tests.

## Self-test

1. You run 50 tests at α = 0.05. Which per-test threshold does Bonferroni use?
2. Explain FWER vs FDR in one sentence each. When would you use each?
3. Why do only the disagreements matter in McNemar's test?
4. Name three forms of p-hacking.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
