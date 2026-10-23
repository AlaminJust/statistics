# 01.05 Exercises — Vectors and matrices

`make lesson L=01-math-toolkit/05-vectors-and-matrices`

## ★ Warm-up

1. **(M)** For **a** = (2, −1, 3) and **b** = (1, 4, 0): compute **a** + **b**, 3**a**, **a**·**b**, ‖**a**‖, ‖**b**‖.
2. **(M)** Compute (1, 0, −1)·(2, 5, 2). What does the result tell you about the angle?
3. **(M)** Give the shape of each product, or say it's impossible: (5 × 3)(3 × 4); (3 × 4)(5 × 3); (2 × 2)(2);
   (1 × 3)(3 × 1); (3 × 1)(1 × 3).

## ★★ Core

4. **(M)** Compute by hand [[2, 0], [1, 3]] · [[1, 4], [2, 1]] and the product in the other order.
5. **(M)** Invert [[4, 2], [1, 1]] and check that A⁻¹A = I.
6. **(M)** Solve 3x + y = 9 and x + 2y = 8 by writing it as A**x** = **b** and using the 2 × 2 inverse formula.
7. **(M)** Explain why [[1, 2], [2, 4]] has no inverse, both with the determinant and by looking at its columns.
8. **(M)** Students' (study hours, sleep hours) are **s₁** = (10, 7), **s₂** = (12, 8), **s₃** = (2, 9). Which two
   students are most similar by cosine similarity? Is cosine similarity a good measure here? (Think about what it
   ignores.)
9. **(P)** Implement in `statkit/mathkit.py` with plain loops (no NumPy): `dot(u, v)`, `norm(u)`,
   `cosine_similarity(u, v)`, `transpose(A)` and `matmul(A, B)` (matrices as lists of lists). Raise `ValueError` when
   shapes don't match. Run `make test M=mathkit`.

## ★★★ Challenge

10. **(M)** Show that for any data matrix X, the matrix XᵀX is symmetric: (XᵀX)ᵀ = XᵀX.
11. **(M)** Using the normal equations, fit y = β₀ + β₁x to (1, 2), (2, 3), (3, 5) by hand. Check with NumPy.
12. **(P)** For `students.csv`, build X with columns (1, study_hours, sleep_hours) and solve the normal equations for
    the three coefficients. Compare with how the data were generated in `data/make_students.py`.
13. **(C)** A neural-network layer computes **h** = σ(W**x** + **b**). If **x** has 784 entries (a 28 × 28 image) and
    the layer has 128 units, what are the shapes of W, **b** and **h**? How many numbers does the layer learn?

Solutions: [solutions/solutions.md](solutions/solutions.md)
