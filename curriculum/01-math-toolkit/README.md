# Module 01 — Math toolkit

**Weeks 1–2 · about 16 hours**

Statistics and ML are written in a small amount of mathematics: summation notation, logarithms, derivatives,
integrals and matrices. This module teaches exactly that toolkit, always with a statistics or ML reason attached.
You don't need to have enjoyed maths at school. Every step is shown.

## Lessons

| # | Lesson | Key idea | Used later in |
| --- | --- | --- | --- |
| 01.01 | [Notation and summation](01-notation-and-summation/lesson.md) | Σ is a `for` loop that adds | Every formula in the course |
| 01.02 | [Functions, exponents and logarithms](02-functions-exponents-logs/lesson.md) | Logs turn products into sums | Likelihood (05), logistic regression (06), softmax |
| 01.03 | [Derivatives and optimisation](03-derivatives-and-optimization/lesson.md) | Minimum ⇒ slope is zero; gradient descent | MLE (05), regression (06), training any ML model |
| 01.04 | [Integrals and area](04-integrals-and-area/lesson.md) | Area under a curve = probability | Continuous distributions (04), expectations |
| 01.05 | [Vectors and matrices](05-vectors-and-matrices/lesson.md) | A dataset is a matrix | Regression (06), PCA (08), all of ML |

## statkit functions for this module

`statkit/mathkit.py`: `summation`, `product`, `sigmoid`, `log_sum_exp`, `numerical_derivative`, `gradient_descent`,
`integrate`, `dot`, `norm`, `cosine_similarity`, `transpose`, `matmul`. Test with `make test M=mathkit`.

## Mini-project: fit a line with gradient descent

Brief in [projects/m01-gradient-descent/README.md](../../projects/m01-gradient-descent/README.md). You will fit
`exam_score ≈ w · study_hours + b` to `data/students.csv` using only the tools from this module, then check your answer
against NumPy. The data were generated with a true slope of 2.2, so you'll know if you got it right.

## Checkpoint (without notes)

1. Expand and compute Σ from i = 1 to 4 of (2i + 1).
2. Explain why we add log-probabilities instead of multiplying probabilities.
3. Show with a derivative that the number m minimising Σ(xᵢ − m)² is the mean.
4. Compute ∫₀¹ 2x dx and explain why that makes 2x a valid probability density on [0, 1].
5. Multiply [[1, 2], [3, 4]] by [[5, 6], [7, 8]] by hand.

Pass all five → `git tag -a m01-done -m "Module 01 complete"`.
