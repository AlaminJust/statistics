# Mini-project m08 — A trustworthy classifier

**Module 08 · week 27 · about 7 hours** · Branch: `git switch -c project/m08-trustworthy-classifier`

## Goal

Build a breast-cancer classifier (`sklearn.datasets.load_breast_cancer`) that a cautious clinician could rely on: honestly
evaluated, calibrated, with a cost-based threshold and prediction sets that say "unsure" when they should.

## Steps

1. **Split once.** Hold out 25% as a final test set (stratified, seed 0). Don't look at it until step 7. Commit after
   splitting, so the history shows the test set was untouched.
2. **Candidates.** Logistic regression (tune C), an SVM (tune C and γ), a random forest (tune depth and min_samples_leaf)
   and gradient boosting (tune the learning rate and depth). Use pipelines with scaling where needed.
3. **Honest comparison.** Nested 5 × 5 cross-validation on the training data for each candidate: report the mean and sd
   of ROC-AUC, PR-AUC and log loss. Choose one model, and justify it in two sentences.
4. **Calibration.** Plot reliability diagrams for the chosen model before and after `CalibratedClassifierCV` (sigmoid and
   isotonic). Report Brier and ECE.
5. **Threshold.** Assume missing a malignant tumour costs 20 times a false alarm. Derive the threshold (Lesson 08.03) and
   check it by minimising the cost on cross-validated predictions.
6. **Conformal sets.** On a calibration split of the training data, build 95% conformal prediction sets. How many training
   cases get the set {benign, malignant} ("unsure")?
7. **Final test, once.** Report on the test set: ROC-AUC and PR-AUC with bootstrap 95% intervals; the confusion matrix
   at your threshold; precision and recall; calibration; conformal coverage and the share of "unsure" sets.
8. **Robustness.** Simulate a shift: add Gaussian noise (sd = 0.5 × each feature's sd) to the test features. How do
   accuracy, calibration and conformal coverage change?

## Deliverable: a "model card" (`model_card.md`)

The model card should cover: intended use and users; data; evaluation (all numbers with intervals); calibration; the
threshold and its cost assumption; uncertainty (conformal sets); limitations, including the shift experiment. The idea
comes from Mitchell et al. (2019), "Model Cards for Model Reporting".

## What "done" looks like

- [ ] The test set was used once (one late commit computes all test numbers)
- [ ] Every metric has an interval
- [ ] Reliability diagrams and the cost curve appear as figures
- [ ] Merged into `main`; tag `m08-done` after the checkpoint
