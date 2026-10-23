# 08.01 Exercises — Generalisation and bias–variance

`make lesson L=08-statistical-learning/01-generalization`

## ★ Warm-up

1. **(C)** Training accuracy is 99% and validation accuracy 75%. What's the diagnosis? Name two remedies.
2. **(C)** Training accuracy is 71% and validation accuracy 70%, while a human expert gets 95%. Diagnosis and remedies?
3. **(M)** Using n ≥ ln(2M/δ)/(2ε²), how many validation examples do you need to pick the best of 50 hyperparameter
   settings and trust its accuracy to ±0.03 with 95% confidence?

## ★★ Core

4. **(P)** Repeat worked example 2 for every degree from 0 to 10, and at three points (x₀ = 0.2, 1.5 and 2.9). Plot bias²,
   variance and their sum against the degree for each point. Where is variance largest, and why?
5. **(P)** Repeat worked example 1 with 300 instead of 30 training points. How do the best degree and the size of the
   overfitting gap change?
6. **(P)** Fit decision trees (`DecisionTreeRegressor`) of depth 1 to 15 to the diabetes data. Plot training and 10-fold CV
   RMSE against depth. Which depth is best? Then compare with a random forest (`RandomForestRegressor`, 300 trees): why
   does averaging many deep trees reduce variance without adding much bias?
7. **(P)** Plot learning curves for logistic regression and for a depth-unlimited decision tree on the breast-cancer data.
   Which is high-variance? Which benefits more from extra data?
8. **(P)** Implement `train_test_split_indices(n, test_fraction, rng)` (returns shuffled train and test index arrays) and
   `k_fold_indices(n, k, rng)` (returns a list of (train, validation) index pairs covering every point exactly once as
   validation) in `statkit/learning.py`.

## ★★★ Challenge

9. **(P)** Reproduce double descent with ridge regression on the random features: add a small penalty λ = 10⁻³ and then
   λ = 10⁻¹ to the minimum-norm fit. What happens to the peak at p = n? What does this suggest about tuning
   regularisation near the threshold?
10. **(M)** Derive the bias–variance decomposition for squared error, starting from E[(y₀ − f̂(x₀))²] with
    y₀ = f(x₀) + ε, E[ε] = 0, Var(ε) = σ², and ε independent of the training set.
11. **(C)** Your colleague says "our network has 10 million parameters and only 50,000 training images, so it must be
    overfitting". Write a short reply using this lesson.

Solutions: [solutions/solutions.md](solutions/solutions.md)
