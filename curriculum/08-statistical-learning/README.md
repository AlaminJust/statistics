# Module 08 — Statistical learning and ML evaluation

**Weeks 24–27 · about 28 hours**

This module turns statistics into ML practice: why models generalise, how to measure performance without fooling
yourself, whether predicted probabilities can be trusted, how to attach guaranteed uncertainty to every prediction, and
how to handle high-dimensional data.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 08.01 | [Generalisation and bias–variance](01-generalization/lesson.md) | Test error = bias² + variance + noise; double descent | `train_test_split_indices`, `k_fold_indices` |
| 08.02 | [Cross-validation and leakage](02-cross-validation-and-leakage/lesson.md) | Pipelines, GroupKFold, TimeSeriesSplit, nested CV | — |
| 08.03 | [Metrics and calibration](03-metrics-and-calibration/lesson.md) | Thresholds from costs; PR-AUC for rare events; reliability diagrams | `confusion_matrix`, `precision_recall_f1`, `roc_curve_points`, `auc`, `brier_score`, `expected_calibration_error` |
| 08.04 | [Uncertainty and conformal prediction](04-uncertainty-and-conformal/lesson.md) | Coverage guaranteed for any model; shift breaks it | `split_conformal_interval` |
| 08.05 | [PCA and high-dimensional data](05-pca-high-dimensional/lesson.md) | Distance concentration; eigenvectors of the covariance | `pca` |

Run `make test M=learning` as you implement each function.

## Mini-project: a trustworthy classifier

Brief in [projects/m08-trustworthy-classifier/README.md](../../projects/m08-trustworthy-classifier/README.md).

## Checkpoint (without notes)

1. Why is a test set used for model selection no longer an unbiased estimate of performance?
2. List four ways data leakage enters a project, and how to prevent each.
3. Why can ROC-AUC look good while precision is poor on a rare class? Which metric would you report instead?
4. How do you choose a decision threshold from costs?
5. What does the conformal coverage guarantee assume, and when does it fail?
6. Explain PCA as an eigenvalue problem.
7. `make test M=learning` passes.

Pass all seven → `git tag -a m08-done -m "Module 08 complete"`.

## Resources

[ISLP](https://www.statlearning.com/) chapters 2, 5, 12 ·
[The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) chapter 7 ·
[Understanding Machine Learning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) ·
[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511) ·
[scikit-learn user guide](https://scikit-learn.org/stable/) · [MAPIE](https://mapie.readthedocs.io/)
