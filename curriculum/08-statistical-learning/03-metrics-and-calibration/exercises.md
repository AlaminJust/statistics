# 08.03 Exercises — Metrics and calibration

`make lesson L=08-statistical-learning/03-metrics-and-calibration`

## ★ Warm-up

1. **(M)** Precision 0.8 and recall 0.4. Compute F1, and say in words what each number means for a spam filter.
2. **(M)** A false negative costs 10 times a false positive. What threshold should you use on calibrated probabilities?
3. **(M)** A test set has 1,000 cases with 20 positives. A model flags 50, of which 15 are truly positive. Fill in the
   confusion matrix and compute accuracy, precision, recall, specificity and F1.

## ★★ Core

4. **(P)** Implement `roc_curve_points(y_true, scores)` (returning FPR and TPR arrays over all thresholds), `auc(fpr, tpr)`
   (trapezoid rule), `brier_score` and `expected_calibration_error(y_true, probs, bins=10)` in `statkit/learning.py`.
   Check them against scikit-learn on the lesson's data.
5. **(P)** Show that ROC-AUC equals the Mann–Whitney probability: compute U/(n₁n₀) with `scipy.stats.mannwhitneyu` for
   the logistic scores and compare with `roc_auc_score`.
6. **(P)** Using the lesson's random forest, find the threshold that maximises F1, and the threshold that gives recall
   ≥ 0.8 with the highest precision. Report precision and recall at both.
7. **(P)** Draw reliability diagrams (10 quantile bins) for the random forest before and after isotonic calibration.
   Which way is the forest miscalibrated?
8. **(P)** **Base rates matter.** Train logistic regression on balanced data (resample to 50% positives), then evaluate
   precision at threshold 0.5 on a test set with the natural 3.5% rate. Compare with the unbalanced model. Explain with
   Bayes' theorem, and show how to correct the probabilities for the prior shift.

## ★★★ Challenge

9. **(M)** Show that the Brier score is a proper scoring rule: for an event with true probability q, the expected score
   E[(p − Y)²] is minimised at p = q.
10. **(P)** Bootstrap 95% intervals for ROC-AUC and PR-AUC of the random forest on the test set. Which metric is more
    uncertain? Why?
11. **(P)** Fit temperature scaling for a small neural network (`sklearn.neural_network.MLPClassifier`) on the digits
    dataset (`load_digits`, 10 classes; use softmax with logits from `predict_proba` → log). Report the ECE (top-label
    confidence vs accuracy) before and after.

Solutions: [solutions/solutions.md](solutions/solutions.md)
