# 08.02 Exercises — Cross-validation and leakage

`make lesson L=08-statistical-learning/02-cross-validation-and-leakage`

## ★ Warm-up

1. **(C)** Why does selecting features before cross-validation inflate accuracy, even with random labels?
2. **(C)** Which CV scheme for each? (a) Predicting next month's sales per store; (b) diagnosing disease from several
   X-rays per patient; (c) classifying independent customer emails; (d) predicting a user's next click from their
   session history.
3. **(C)** List three features that would be target leakage when predicting whether a loan will default.

## ★★ Core

4. **(P)** Reproduce the noise-feature leakage example for p = 100, 500, 2,000 and 10,000 features (n = 100). Plot the
   "wrong" and "right" CV accuracies against p. Why does the wrong one get *better* as p grows?
5. **(P)** Show scaling leakage: fit a k-NN classifier on the breast-cancer data with the scaler fitted (a) on all data
   before CV and (b) inside a pipeline. How big is the difference here? When could it be large?
6. **(P)** Use nested CV to compare logistic regression (tuning C) with an SVM (tuning C and γ) on the breast-cancer data.
   Report both nested estimates with their sds. Is the difference meaningful?
7. **(P)** Simulate **duplicate leakage**: duplicate 30% of the breast-cancer rows (with tiny noise) before a random
   train/test split, and compare the test accuracy of a 1-NN classifier with and without the duplicates. Why does 1-NN
   suffer most?
8. **(P)** Implement `confusion_matrix(y_true, y_pred)` (returning [[TN, FP], [FN, TP]]) and
   `precision_recall_f1(y_true, y_pred)` in `statkit/learning.py` (they're used in Lesson 08.03).

## ★★★ Challenge

9. **(P)** **Group leakage in disguise.** Using `data/ab_test.csv`, build a model predicting `converted` from `pre_spend`,
   `device` and `user_id` (as a number). Is `user_id` leaky here? Explain why IDs are sometimes leaky in real data (think
   of IDs assigned in time order).
10. **(P)** For a time series of your choice (simulate an AR(1) with a trend), compare the error estimates of shuffled
    KFold, TimeSeriesSplit and a single final 20% hold-out. Which matches the true forecasting error on the next 200
    simulated points?
11. **(C)** Read the Kaggle Leakage guide or any public post-mortem of a leakage incident (search "data leakage machine
    learning case study"). Summarise how the leak entered and how it was caught.

Solutions: [solutions/solutions.md](solutions/solutions.md)
