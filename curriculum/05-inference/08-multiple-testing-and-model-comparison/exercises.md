# 05.08 Exercises — Multiple testing and comparing models

`make lesson L=05-inference/08-multiple-testing-and-model-comparison`

## ★ Warm-up

1. **(M)** You run 50 tests at α = 0.05. What is the Bonferroni per-test threshold? If all 50 nulls are true and
   independent, what's the chance of at least one false positive without correction?
2. **(M)** Apply Bonferroni, Holm and BH (q = 0.05) by hand to p = (0.0002, 0.004, 0.019, 0.03, 0.05, 0.12, 0.31, 0.48).
   How many does each reject? Check with `multipletests`.
3. **(C)** Explain FWER vs FDR in one sentence each. Which would you use for (a) choosing which 3 of 200 candidate genes
   to study further, (b) approving a new drug on its single primary outcome?

## ★★ Core

4. **(M)** Two classifiers disagree on 20 test examples: A is right on 15 of them and B on 5. Run McNemar's exact test.
   What if they disagree on only 12, with a 9–3 split?
5. **(P)** Implement `bonferroni(pvalues, alpha=0.05)` and `benjamini_hochberg(pvalues, q=0.05)` in
   `statkit/inference.py`. Each returns a list of True/False (reject) in the **original** order. Check against
   `multipletests`.
6. **(P)** **Subgroup fishing.** In `students.csv`, test the tutoring effect separately within every program × year
   subgroup that has at least 3 students in each arm (Welch t-tests). How many subgroups show p < 0.05? Apply BH. Then
   explain why "tutoring only works for 4th-year Science students" would be a bad conclusion, given that the overall
   effect is clear.
7. **(P)** Simulate the optional-stopping experiment for peeks every 1, 10 and 50 subjects (up to 200). How does the
   false-positive rate depend on how often you peek?
8. **(P)** Train two different models (for example logistic regression with C = 1 and C = 0.01) on the breast-cancer data.
   Compare them with McNemar's test and a paired bootstrap interval for the accuracy difference.

## ★★★ Challenge

9. **(P)** Implement the **Nadeau–Bengio corrected resampled t-test**: for k fold differences d₁…d_k with training size
   n_train and test size n_test, t = d̄ / √((1/k + n_test/n_train) s_d²), with k − 1 df. Re-run worked example 7 with this
   correction (assume 10 folds of a 1,000-example dataset). Is the false-positive rate closer to 5%?
10. **(M)** Show that Bonferroni controls the FWER using the union bound P(A₁ ∪ … ∪ A_m) ≤ Σ P(Aᵢ).
11. **(C)** Find an ML paper (any recent arXiv paper with a results table) that compares many methods on many datasets.
    List every comparison it makes, say whether it corrects for multiplicity or reports variance across seeds, and
    suggest one concrete improvement.

Solutions: [solutions/solutions.md](solutions/solutions.md)
