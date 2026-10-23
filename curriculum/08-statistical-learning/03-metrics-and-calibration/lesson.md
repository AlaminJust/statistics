# 08.03 — Metrics, thresholds and calibration

**Time:** about 5 hours · **Prerequisites:** 03.04 (Bayes, base rates), 04.06 (cross-entropy), 06.04, 08.02

## Why this matters

"96.5% accuracy" can describe a model that never detects a single fraud case. Choosing the right metric is choosing
what "good" means for the decision the model supports. For classifiers that output probabilities there's a second
question: **are the probabilities themselves trustworthy?** A model that says "90% risk" should be right about 90% of the
time. Medical, financial and safety applications, and anything downstream that combines model outputs, depend on this
**calibration**.

## Learning objectives

- Build a confusion matrix and compute precision, recall (sensitivity), specificity and F1.
- Choose a decision threshold from costs.
- Read ROC and precision–recall curves; know when ROC-AUC misleads.
- Use proper scoring rules (log loss, Brier score).
- Diagnose calibration with reliability diagrams and ECE, and fix it with Platt scaling, isotonic regression or
  temperature scaling.

## Before you start

3.5% of transactions are fraudulent. A model has 96.5% accuracy. Is that impressive?

---

## 1. The confusion matrix

| | predicted negative | predicted positive |
| --- | --- | --- |
| **actual negative** | TN (true negative) | FP (false positive, Type I) |
| **actual positive** | FN (false negative, Type II) | TP (true positive) |

| Metric | Formula | Question it answers |
| --- | --- | --- |
| accuracy | (TP + TN)/n | how often is it right overall? |
| **precision** | TP/(TP + FP) | when it says positive, how often is it right? |
| **recall** (sensitivity, TPR) | TP/(TP + FN) | what share of real positives does it catch? |
| specificity (TNR) | TN/(TN + FP) | what share of negatives does it clear? |
| false positive rate | FP/(FP + TN) = 1 − specificity | |
| **F1** | 2 · precision · recall/(precision + recall) | a single balance of the two (harmonic mean) |

**Worked example 1 (Before you start).** A simulated fraud problem: 10,000 test cases, 351 positives (3.5%).

| Model | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| always "not fraud" | **0.965** | — | **0** |
| logistic regression, threshold 0.5 | 0.965 | 0.52 | **0.07** |

Both have about 96.5% accuracy; neither is useful as it stands. **With imbalanced classes, accuracy is almost
meaningless.** Look at precision and recall, and at the base rate (Lesson 03.04: low prevalence crushes precision).

## 2. Thresholds are business decisions

The model outputs a probability; "positive if p ≥ t" is a separate decision. Lowering t catches more positives (higher
recall) at the cost of more false alarms (lower precision):

| Threshold | Precision | Recall | Cases flagged |
| --- | --- | --- | --- |
| 0.5 | 0.52 | 0.07 | 48 |
| 0.2 | 0.36 | 0.35 | 334 |
| 0.1 | 0.24 | 0.57 | 835 |
| 0.05 | 0.15 | 0.71 | 1,643 |

**Choose t from costs.** If a missed fraud costs 50 and a false alarm costs 1, flag a case when the expected cost of
ignoring it exceeds the cost of flagging it: p × 50 > (1 − p) × 1, i.e.

$$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{1}{51} \approx 0.02.$$

Searching thresholds on the test set agrees: **0.02**. This rule needs **calibrated** probabilities (section 5).

## 3. ROC and precision–recall curves

Sweep the threshold over all values:

- **ROC curve:** TPR (recall) against FPR. **ROC-AUC** = the area under it = **P(a random positive scores higher than a
  random negative)**, the Mann–Whitney probability of Lesson 05.06. Here 0.850, matching the direct pairwise count.
- **Precision–recall curve:** precision against recall. Its area, **average precision** (PR-AUC), has a baseline equal
  to the positive rate (0.035), not 0.5.

**Worked example 3.**

| Model | ROC-AUC | PR-AUC |
| --- | --- | --- |
| logistic regression | 0.850 | 0.279 |
| naive Bayes | 0.857 | 0.225 |
| random forest | **0.892** | **0.503** |

All three ROC curves look respectable (`roc_pr_calibration.png`). The PR curves reveal how hard the problem really is
and separate the models much more: the random forest's PR-AUC is nearly twice logistic regression's. ROC-AUC counts
true negatives, which are plentiful and easy with rare positives, so it hides poor precision. **For rare positives,
report PR-AUC (and precision at the recall you need).**

Note also that naive Bayes has a *higher* ROC-AUC than logistic regression but a *lower* PR-AUC: the metrics can rank
models differently.

## 4. Proper scoring rules

A **proper** scoring rule is optimised, in expectation, by reporting your true probabilities. Honest forecasts win.

| Score | Formula | Notes |
| --- | --- | --- |
| **log loss** (cross-entropy) | −mean[y log p + (1 − y) log(1 − p)] | Lesson 04.06; punishes confident mistakes harshly |
| **Brier score** | mean (p − y)² | squared error on probabilities; bounded, more forgiving |

Accuracy, precision and F1 aren't proper: they only see thresholded predictions. Use log loss or Brier to compare
probabilistic models.

## 5. Calibration

A model is **calibrated** if, among cases where it predicts p, the outcome happens a fraction p of the time.

- **Reliability diagram:** bin the predictions, and plot the observed frequency against the mean prediction in each bin.
  Calibrated models lie on the diagonal.
- **Expected calibration error (ECE):** the average |observed − predicted| over bins, weighted by bin size.

**Worked example 4.**

| Model | Log loss | Brier | ECE | ROC-AUC |
| --- | --- | --- | --- | --- |
| logistic regression | 0.116 | 0.029 | **0.005** | 0.850 |
| naive Bayes | 0.131 | 0.035 | **0.017** | 0.857 |
| random forest | 0.113 | 0.025 | 0.016 | 0.892 |
| naive Bayes + isotonic | 0.118 | 0.030 | **0.002** | 0.857 |
| naive Bayes + Platt (sigmoid) | 0.127 | 0.031 | 0.005 | 0.856 |

Logistic regression is naturally well calibrated (it directly maximises the likelihood). Naive Bayes is overconfident:
its independence assumption double-counts correlated evidence. **Recalibration** fits a mapping from raw scores to
calibrated probabilities on held-out data:

- **Platt scaling:** a logistic regression on the score (2 parameters);
- **isotonic regression:** a flexible monotone step function (needs more data);
- in scikit-learn, `CalibratedClassifierCV(model, method="isotonic" or "sigmoid", cv=5)`.

Calibration improves log loss, Brier and ECE, but **barely changes the AUC**, because it's monotone and so preserves the
ranking. Ranking quality (discrimination) and probability quality (calibration) are different properties.

### Temperature scaling for neural networks

Modern deep networks are often overconfident. **Temperature scaling** divides the logits by a single T > 0 fitted on a
validation set by minimising log loss: p = σ(logit/T). **Worked example 5:** logits made artificially 3 times too large
give a fitted **T = 3.10**, and the ECE falls from 0.029 to 0.009, with the AUC unchanged (0.871). One parameter, no
change to the ranking or the accuracy: the standard fix in deep learning.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Accuracy on imbalanced data | Precision, recall, PR-AUC, costs |
| Treating 0.5 as "the" threshold | Choose it from costs or capacity |
| ROC-AUC only, for rare events | Add PR-AUC and precision at the operating point |
| Comparing probabilistic models by accuracy | Log loss or Brier |
| Calibrating on the training data | Calibrate on held-out data (or within CV) |
| Assuming a high-AUC model gives good probabilities | Check a reliability diagram |

## Summary

- Confusion matrix → precision, recall, specificity, F1; accuracy misleads with imbalance.
- Threshold t* = C_FP/(C_FP + C_FN) with calibrated probabilities.
- ROC-AUC = P(positive ranked above negative); PR-AUC is more informative for rare positives.
- Log loss and Brier are proper scoring rules.
- Calibration: reliability diagram, ECE; fix with Platt, isotonic or temperature scaling; AUC is unchanged.

## Self-test

1. Precision 0.8 and recall 0.4: compute F1. What does each number mean?
2. A false negative costs 10 times a false positive. What threshold should you use?
3. Why can ROC-AUC stay high while precision is low?
4. Why doesn't calibration change the AUC?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
