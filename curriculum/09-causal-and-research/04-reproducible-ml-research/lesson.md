# 09.04 — Reproducible ML research and reporting

**Time:** about 5 hours · **Prerequisites:** 05.03–05.08, 08.02, 09.01

## Why this matters

Many published ML improvements shrink or vanish when someone else re-runs them. The usual causes are statistical, not
fraud: results from a single random seed, "best of several runs" reporting, baselines tuned less carefully than the new
method, test sets used for model selection, many comparisons with no correction, and no uncertainty estimates. This
lesson collects the practices that make results trustworthy, so that the papers, theses and reports you write hold up.
It ties together the whole course.

## Learning objectives

- Measure and report variability across random seeds, with intervals.
- Explain why "best of k runs" is misleading, and compare methods with paired designs.
- Plan the number of seeds with a power calculation.
- Design ablation studies.
- Aggregate results across tasks robustly (interquartile mean, stratified bootstrap).
- Use pre-registration, fair baselines and reporting checklists.

## Before you start

You train a small neural network once and get 98.5% test accuracy, and a colleague's version gets 97.4%. Is yours better?
What would you need to know?

---

## 1. Randomness is everywhere in ML

Weight initialisation, data shuffling, dropout, data augmentation, train/test splits, and even GPU non-determinism make
every training run a random draw. **Worked example 1:** the same 32-unit network on the digits data with 20 seeds:

| Statistic | Test accuracy |
| --- | --- |
| minimum | 0.9685 |
| maximum | 0.9852 |
| mean | 0.9750 |
| sd across seeds | 0.0041 |

The range is **1.7 points** from seeds alone. Answer to "Before you start": a single 98.5% vs 97.4% comparison is within
the seed noise of *one* method; you need several runs of both.

**Report:** "0.9750 ± 0.0041 (mean ± sd over 20 seeds); 95% CI for the mean [0.9731, 0.9769]". Say whether the ± is an sd
(how much single runs vary) or a CI (how precisely the mean is known).

## 2. "Best of k" and other optimistic reporting

**Worked example 2.** A "new method" that's really no better (mean 0.9751 vs 0.9750):

- its best of 5 seeds (0.9870) against the baseline's first seed (0.9741) looks like **+1.3 points**;
- the paired comparison over 20 seeds (both methods with the same seeds): mean difference **+0.0001**, p = 0.33.

Taking the maximum of several noisy runs is the winner's curse again (Lesson 05.07). **Rules:** report the mean (or
median) over all runs with an interval; use the same seeds and splits for every method (a *paired* design); and give the
baselines the same tuning budget as your method.

## 3. How many seeds?

The SE of a difference of two means over k seeds each is about sd√(2/k). With sd ≈ 0.004, detecting a +0.5-point gain:

| Seeds per method | SE of the difference | Power |
| --- | --- | --- |
| 3 | 0.0033 | 0.32 |
| 5 | 0.0026 | 0.49 |
| 10 | 0.0018 | 0.78 |
| 20 | 0.0013 | 0.97 |

Three seeds, common in practice, detect a half-point improvement only a third of the time. Run a pilot to estimate the
seed sd, then choose k (Lesson 05.07). For expensive models where many seeds are impossible, say so, and soften the
claims.

## 4. Ablation studies

An **ablation** removes or changes one component at a time to measure its contribution. **Worked example 4** (8 seeds each):

| Configuration | Accuracy (mean ± sd) |
| --- | --- |
| full model (input scaling + 64 units + weight decay) | 0.9775 ± 0.0036 |
| − without weight decay | 0.9769 ± 0.0030 |
| − with 16 hidden units | 0.9699 ± 0.0054 |
| − without input scaling | 0.9690 ± 0.0056 |

Weight decay contributes almost nothing here; width and scaling each matter by about 0.8 points. Good ablations change
one thing at a time, use the same seeds and data, report spread, and include the ablations that *didn't* matter. Those
are valuable information, not failures.

## 5. Comparing across datasets

**Worked example 5:** MLP vs random forest on three datasets, 10 splits each:

| Dataset | MLP | Forest | Difference (middle 95% of splits) |
| --- | --- | --- | --- |
| digits | 0.9733 | 0.9752 | +0.002 (−0.010 to +0.014) |
| breast cancer | 0.9708 | 0.9596 | −0.011 (−0.023 to +0.005) |
| wine | 0.9778 | 0.9852 | +0.007 (−0.033 to +0.033) |

No method wins everywhere, and every difference is within split-to-split noise. With many datasets, compare with a
paired test across datasets (Wilcoxon signed-rank), or a Friedman test with post-hoc corrections (Demšar, 2006), and
report how often each method wins.

## 6. Robust aggregate scores

The mean across tasks is dominated by a few tasks with large scales or outliers; the median ignores most of the data.
The **interquartile mean** (IQM, the mean of the middle 50% of all runs) is a robust compromise. Compute its uncertainty with a
**stratified bootstrap**: resample runs *within* each task, then recompute (Agarwal et al., 2021, "Deep reinforcement
learning at the edge of the statistical precipice"; the `rliable` library). **Worked example 6:** MLP IQM 0.9740 (95% CI
0.9688 to 0.9790), forest IQM 0.9731 (0.9692 to 0.9767): heavily overlapping. No evidence of a difference.

## 7. Process: what to write down before, during and after

| Stage | Practice |
| --- | --- |
| **before** | write the hypothesis, metrics, datasets, baselines, tuning budget and number of seeds; commit or tag it (**pre-registration**) |
| **during** | fixed seeds; log every run (configs, code version via the git commit, data version); never touch the test set |
| **after** | report all runs, with means, intervals and paired comparisons; correct for multiple comparisons; include failed ablations |
| **share** | code, configs, environment (`requirements.txt`), data or data access, and a script that reproduces every table and figure |

The **NeurIPS paper checklist** (and similar ones from ICML and ACL) asks exactly these questions: error bars? how many
runs? compute budget? statistical significance? data splits? Use it as a self-review before submitting anything.

Your course repository already follows these habits: pinned requirements, seeds everywhere, every figure produced by a
script, plans committed before analyses, and a git history of every change.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| One seed per method | ≥ 5–10 seeds, more for small effects |
| Best-of-k reporting | Mean or median over all runs, with intervals |
| Unpaired comparisons with different seeds or splits | The same seeds and splits for every method |
| Baselines with default hyperparameters | An equal tuning budget |
| Reporting only the ablations that helped | Report them all |
| "± 0.004" without saying what it is | State sd vs SE vs CI, and the number of runs |

## Summary

- Training is random: report the spread across seeds and an interval for the mean.
- "Best of k" inflates results; compare paired runs with the same seeds and splits.
- Choose the number of seeds by power: SE of a difference ≈ sd√(2/k).
- Ablations change one component at a time, with spread, including null results.
- Aggregate across tasks with the IQM and a stratified bootstrap.
- Pre-register, log, share code and environment, and use reporting checklists.

## Self-test

1. Your method beats the baseline by 0.3 points, and the seed sd is 0.4 points. How many seeds per method for 80% power?
2. Why should both methods use the same seeds?
3. What does "± 0.4" mean in a results table, and why does it matter?
4. Name three items from the NeurIPS checklist and say why each matters.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
