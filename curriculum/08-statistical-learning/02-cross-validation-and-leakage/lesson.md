# 08.02 — Cross-validation and data leakage

**Time:** about 4 hours · **Prerequisites:** 08.01, 05.04, 06.05

## Why this matters

Every ML result rests on one number: the estimated performance on unseen data. If that estimate is wrong, nothing built
on it can be trusted. The two big threats are **noise** (one lucky split) and **leakage** (information from the test data
seeping into training). Leakage is the most common reason published ML results fail to reproduce and models fail in
deployment. This lesson shows how cross-validation reduces noise, how to choose splits that match the real use case,
how nested CV keeps tuning honest, and the main forms of leakage, each demonstrated with code.

## Learning objectives

- Explain why a single split is noisy, and run k-fold and stratified k-fold CV.
- Choose a split that matches deployment: grouped data, time series.
- Use nested CV to evaluate a tuned model honestly.
- Recognise and prevent leakage from preprocessing, feature selection, groups, time and target-derived features.
- Use scikit-learn pipelines so that every learned step happens inside the folds.

## Before you start

You have 2,000 random features and labels that are pure coin flips. You select the 20 features most correlated with the
labels, then estimate accuracy with 5-fold cross-validation. What accuracy do you expect to see?

---

## 1. Hold-out vs cross-validation

| Split | How | Pros / cons |
| --- | --- | --- |
| hold-out | one train/test split (e.g. 80/20) | simple; noisy; uses data inefficiently |
| **k-fold CV** | split into k folds; train on k − 1, validate on the remaining one; rotate; average | every point validated once; less noisy |
| stratified k-fold | k-fold keeping class proportions in every fold | the default for classification |
| repeated k-fold | k-fold several times with different shuffles | even less noise |
| leave-one-out | k = n | nearly unbiased but high variance and expensive |

**Worked example 1.** Logistic regression on the breast-cancer data, 200 random 80/20 splits: test accuracy ranges from
**0.939 to 1.000** (sd 0.013). Report "100% accuracy" from one lucky split and you'd be misleading everyone. 10-fold CV:
**0.977** (fold sd 0.019). Report the CV mean with its spread, and remember the folds aren't independent (Lesson 05.08).

## 2. Leakage through preprocessing and feature selection

Anything the model **learns from data** (scaling parameters, imputation values, selected features, PCA components,
target encodings) must be learned **inside each training fold only**, and then applied to the validation fold.

**Worked example 2 (Before you start).** 100 samples, 2,000 pure-noise features, random labels:

| Procedure | CV accuracy |
| --- | --- |
| select the 20 "best" features using **all** labels, then cross-validate | **0.850** |
| selection **inside** each fold (a pipeline) | **0.470**, chance level |

The first procedure picked features that are correlated with *these particular* random labels, including the labels of
the validation folds. The model then "predicts" noise it has already seen. Studies with many features and few samples
(genomics, neuroimaging) published results like the first number for years.

**The fix: pipelines.**

```python
model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression())
cross_val_score(model, X, y, cv=5)        # every step is refitted inside each training fold
```

Scaling leaks too, though usually only a little. Feature selection and target encoding can leak a lot.

## 3. Match the split to deployment

**Ask: what will the model face in deployment, and does my split reproduce it?**

### Grouped data

**Worked example 3.** 60 patients with 8 measurements each; each patient has a personal "fingerprint" in their features,
and the disease signal is weak.

| Split | Accuracy |
| --- | --- |
| KFold (rows from the same patient in both train and test) | **0.990** |
| GroupKFold (all of a patient's rows together) | **0.465** |

With KFold the model recognises *patients* it has already seen and copies their labels. On new patients it's at chance.
Use `GroupKFold` (or `StratifiedGroupKFold`) whenever rows share a source: patients, users, sessions, images from the
same scan, sentences from the same document, near-duplicate samples.

### Time series

**Worked example 4.** Predicting the next value of a trending random walk from its 10 previous values: shuffled KFold
gives MAE **0.791**, while `TimeSeriesSplit` (always train on the past and test on the future) gives **0.862**. Shuffling
lets the model interpolate between neighbouring future and past points, which it can't do in deployment. For any
temporal data (sales, prices, sensor logs, user activity), validate forward in time.

## 4. Tuning without fooling yourself: nested CV

If you pick the best of many hyperparameter settings by CV score, that best score is **optimistic**: it's the maximum of
many noisy estimates (the winner's curse, Lesson 05.07).

**Worked example 5.** An SVM tuned over 20 settings (C × γ):

| Estimate | Accuracy |
| --- | --- |
| best inner-CV score | 0.9824 |
| **nested CV** (an outer loop around the entire tuning procedure) | **0.9737** (sd 0.024) |

```python
inner = GridSearchCV(pipe, grid, cv=5)              # tunes on the training part of each outer fold
cross_val_score(inner, X, y, cv=5)                  # evaluates the tuning *procedure* honestly
```

The equivalent with a fixed test set: tune by CV on the training data only, then evaluate the final model **once** on
the test set. If you look at the test score, change something and evaluate again, the test set has become a validation
set.

## 5. Target leakage

A feature that is **caused by** the outcome, or only known after it, will look predictive and be useless at prediction
time.

**Worked example 6.** Add a feature equal to the label plus noise (in reality, e.g. "treatment given", recorded after
the diagnosis): CV accuracy **0.998**. Warning signs: a single feature with near-perfect importance; performance that
seems too good; a feature whose timestamp is after the prediction time.

Common real examples: "number of follow-up visits" when predicting a disease; "account closed" when predicting churn;
IDs or file paths that encode the label; for images, hospital-specific markings that correlate with the diagnosis.

## 6. The leakage checklist

| Leak | Prevention |
| --- | --- |
| preprocessing fitted on all data (scaling, imputation, PCA) | pipeline inside CV |
| feature selection or target encoding on all data | pipeline inside CV |
| same entity in train and test | GroupKFold |
| future data in training | TimeSeriesSplit; split by time |
| duplicates or near-duplicates across the split | deduplicate before splitting |
| tuning on the test set | nested CV, or a test set used once |
| features unavailable at prediction time | audit each feature's timestamp and origin |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Reporting one train/test split | CV mean ± sd, or a repeated hold-out |
| `scaler.fit(X)` before splitting | Fit inside a pipeline |
| Plain KFold on grouped or temporal data | GroupKFold / TimeSeriesSplit |
| Reporting the best grid-search score as performance | Nested CV or an untouched test set |
| Celebrating suspiciously good results | Hunt for leakage first |

## Summary

- k-fold CV reduces split noise; stratify for classification.
- Everything learned from data goes inside a pipeline, refitted in each fold.
- The split must mimic deployment: groups → GroupKFold; time → TimeSeriesSplit.
- Tuning makes the best score optimistic: use nested CV or a test set used once.
- Target leakage: features that depend on the label or the future.

## Self-test

1. Why does selecting features before cross-validation inflate accuracy, even with random labels?
2. You're predicting next month's sales for each store. Which CV scheme?
3. What does nested CV estimate that ordinary CV with tuning doesn't?
4. Give an example of target leakage from a field you know.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
