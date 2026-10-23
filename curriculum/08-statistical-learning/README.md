# Module 08 — Statistical learning and ML evaluation

**Weeks 24–27 · about 32 hours** · Status: **outline**

This module turns statistics into ML practice: why models generalise, how to measure performance without fooling
yourself, how to know whether predicted probabilities can be trusted, and how to attach a guaranteed uncertainty to
every prediction.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 08.01 | Generalisation and the bias–variance trade-off | Explain over- and underfitting precisely | true vs empirical risk, model capacity, the bias–variance decomposition of prediction error, learning curves, double descent, intuitive VC and Rademacher bounds |
| 08.02 | Cross-validation and data leakage | Estimate performance honestly | train/validation/test splits, k-fold, stratified, grouped and time-series CV, nested CV, scikit-learn Pipelines, the five common leaks |
| 08.03 | Metrics and calibration | Choose a metric that matches the decision | confusion matrix, precision, recall, F1, ROC and PR curves, AUC, class imbalance, proper scoring rules (log loss, Brier), reliability diagrams, ECE, Platt, isotonic and temperature scaling |
| 08.04 | Uncertainty and conformal prediction | Give predictions with guaranteed coverage | aleatoric vs epistemic uncertainty, split conformal prediction, conformalised quantile regression, ensembles, distribution shift |
| 08.05 | PCA and high-dimensional data | Reduce dimensions with statistical sense | the curse of dimensionality, PCA as eigendecomposition of the covariance matrix, explained variance, probabilistic PCA, embeddings |

## statkit functions (`statkit/learning.py`)

`train_test_split(n, test_fraction, rng)`, `k_fold_indices(n, k, rng)`, `confusion_matrix`, `precision_recall_f1`,
`roc_curve_points`, `auc`, `brier_score`, `expected_calibration_error`, `split_conformal_interval`, `pca(X, k)`.

## Mini-project: a trustworthy classifier

Train a classifier on scikit-learn's breast-cancer dataset with a leakage-free pipeline and nested cross-validation.
Report metrics with bootstrap intervals, a reliability diagram before and after calibration, and conformal prediction
sets with their empirical coverage.

## Checkpoint (without notes)

1. Why is a test set used for model selection no longer an unbiased estimate of performance?
2. List four ways data leakage enters a project.
3. Why can ROC-AUC look good while precision is poor on a rare class?
4. What does the conformal coverage guarantee assume?
5. `make test M=learning` passes.

## Resources

[ISLP](https://www.statlearning.com/) chapters 2, 5, 12 ·
[The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) chapter 7 ·
[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511) ·
[scikit-learn user guide](https://scikit-learn.org/stable/)
