# 08.01 Solutions

1. **High variance (overfitting).** Remedies: more training data, regularisation (weight decay, dropout, a smaller depth
   or larger k), a simpler model, data augmentation, or an ensemble.
2. **High bias (underfitting)**: both errors are close together and far below what's achievable. Remedies: a more
   flexible model, better features, less regularisation, longer training. More data alone won't help much.
3. n ≥ ln(2 × 50/0.05)/(2 × 0.03²) = ln 2000/0.0018 ≈ **4,223** examples.
4. Variance is largest at the edges (x₀ = 0.2 and 2.9) and grows fast with the degree: high-degree polynomials swing
   wildly where data are sparse. In the middle (x₀ = 1.5) variance stays small for longer. Bias² is largest for low
   degrees and roughly flat from degree 3 on.
5. With 300 points the best degree is about 4 (test MSE 0.172, close to the 0.16 floor), and even degree 15 barely
   overfits (training 0.171, test 0.179). More data shrinks variance, so flexible models become safer.
6. | depth | training RMSE | 10-fold CV RMSE |
   | --- | --- | --- |
   | 1 | 64.8 | 68.1 |
   | 2 | 58.0 | 62.7 |
   | **3** | 54.4 | **62.4** |
   | 5 | 44.9 | 69.1 |
   | 15 | 1.5 | 83.9 |

   Depth 3 is best. A random forest of 300 deep trees gets CV RMSE **58.1**, better than any single tree. Each deep tree
   has low bias but high variance; averaging decorrelated trees (bootstrap samples plus random feature subsets) cancels
   much of the variance (Lesson 03.06, exercise 12) while keeping the low bias.
7. Logistic regression: training about 0.99, validation rising from 0.968 to 0.981, a small gap (well-balanced). Tree:
   training **1.000** at every size, validation only 0.89–0.92: high variance. The tree's validation score improves only
   slowly with more data; logistic regression is already close to its best.
8. See `../solutions/statkit_learning_reference.py`.
9. Test MSE at p = 50 / 100 / 400 / 2000: λ = 0 → 0.72 / **784** / 0.52 / 0.50; λ = 10⁻³ → 0.72 / **2.9** / 0.52 / 0.50;
   λ = 0.1 → 0.61 / **0.51** / 0.51 / 0.50. Even a little ridge penalty removes the spike: it comes from the
   unregularised fit being forced through every noisy point. With a well-tuned penalty, test error falls almost
   monotonically with p.
10. Write y₀ − f̂ = (f(x₀) − E f̂) + (E f̂ − f̂) + ε. Square it and take expectations. The cross terms vanish, because
    E[E f̂ − f̂] = 0 and ε is independent with mean 0. What's left is (f − E f̂)² + Var(f̂) + σ². ∎
11. "Parameter count alone doesn't determine overfitting. Over-parameterised networks trained by gradient descent tend
    to find simple solutions (implicit regularisation), and their test error often keeps falling as size grows, past
    the interpolation threshold (double descent). The only way to know whether we're overfitting is to measure: compare
    training and held-out performance, and look at learning curves."
