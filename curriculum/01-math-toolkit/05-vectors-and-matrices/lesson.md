# 01.05 — Vectors and matrices

**Time:** about 4 hours · **Prerequisites:** 01.01–01.03

## Why this matters

A dataset with n rows and p columns *is* a matrix. Every statistical and ML method works on that matrix:

- regression solves a matrix equation for its coefficients (Module 06);
- PCA finds the directions in which the data matrix varies most (Module 08);
- a neural-network layer is a matrix multiplication followed by a function;
- embeddings (words, images) are vectors compared with dot products and cosine similarity.

This lesson covers the operations you need, each linked to where it's used.

## Learning objectives

- Add vectors, scale them, and compute dot products, lengths and cosine similarity.
- Multiply a matrix by a vector and by another matrix, and check that the shapes are compatible.
- Transpose a matrix and explain the identity and the inverse.
- Solve a small linear system by hand and with NumPy.
- Represent a dataset as a matrix X and compute XᵀX.

## Before you start

Guess: is A·B always equal to B·A for matrices, the way 3 × 5 = 5 × 3 for numbers?

---

## 1. Vectors

A **vector** is an ordered list of numbers: **u** = (1, 2, 3). It can be read as a point, an arrow from the origin, or
one row of data (a student's study hours, sleep and score).

| Operation | Rule | Example with **u** = (1, 2, 3), **v** = (4, 5, 6) |
| --- | --- | --- |
| Addition | add matching entries | **u** + **v** = (5, 7, 9) |
| Scalar multiple | multiply every entry | 2**u** = (2, 4, 6) |
| **Dot product** | multiply matching entries, then add | **u**·**v** = 1·4 + 2·5 + 3·6 = **32** |
| Length (norm) | √(**u**·**u**) | ‖**u**‖ = √(1 + 4 + 9) = √14 ≈ 3.742 |

In Σ notation the dot product is **u**·**v** = Σᵢ uᵢvᵢ. You've already used it: a **weighted sum** is a dot product
of weights and values, and Σxᵢ² is **x**·**x**.

### The dot product measures alignment

$$\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta,$$

where θ is the angle between the arrows. Rearranged, this gives **cosine similarity**:

$$\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}.$$

| cos θ | Meaning |
| --- | --- |
| 1 | same direction |
| 0 | perpendicular (no alignment) |
| −1 | opposite directions |

**Worked example 1.** ‖**v**‖ = √(16 + 25 + 36) = √77 ≈ 8.775. cos θ = 32 / (3.742 × 8.775) = 32 / 32.83 ≈ **0.975**.
The two vectors point in almost the same direction. Search engines and recommendation systems rank items by exactly
this number on embedding vectors. In Module 02 you'll see that the **correlation coefficient is a cosine similarity
of centred data**.

## 2. Matrices

A **matrix** is a rectangular table of numbers with a shape (rows × columns). A is m × n with entry aᵢⱼ in row i,
column j.

$$A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}\quad(2\times 2), \qquad X = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}\quad(3\times 2).$$

### Matrix × vector

Each output entry is the dot product of one **row** of the matrix with the vector.

**Worked example 2.** A**x** with **x** = (1, 2):

- row 1: (2, 1)·(1, 2) = 2 + 2 = **4**
- row 2: (1, 3)·(1, 2) = 1 + 6 = **7**

So A**x** = (4, 7). Shapes: (2 × 2)(2) → (2).

This is what a **linear model** does to every row of data at once: predictions ŷ = X**β**, one dot product per row.

### Matrix × matrix

Entry (i, j) of AB = (row i of A) · (column j of B). **The inner sizes must match:** (m × n)(n × p) → (m × p).

**Worked example 3.**

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1\cdot5+2\cdot7 & 1\cdot6+2\cdot8 \\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}.$$

In the other order:

$$\begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 23 & 34 \\ 31 & 46 \end{pmatrix}.$$

That answers "Before you start": **AB ≠ BA in general.** Order matters.

In NumPy, `@` is matrix multiplication; `*` multiplies entry by entry (a different operation).

## 3. Transpose, identity, inverse

| Concept | Definition | Example |
| --- | --- | --- |
| **Transpose** Aᵀ | swap rows and columns: (Aᵀ)ᵢⱼ = aⱼᵢ | the 3 × 2 X becomes 2 × 3 |
| **Identity** I | 1s on the diagonal, 0 elsewhere; IA = AI = A | like multiplying by 1 |
| **Inverse** A⁻¹ | the matrix with A⁻¹A = I; "dividing" by A | only square matrices with det ≠ 0 have one |

For a 2 × 2 matrix there is a formula:

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}.$$

ad − bc is the **determinant**. If it is 0, the matrix has no inverse (its columns point in the same direction).

**Worked example 4.** For A = [[2, 1], [1, 3]]: det = 2·3 − 1·1 = 5.
A⁻¹ = (1/5)[[3, −1], [−1, 2]] = [[0.6, −0.2], [−0.2, 0.4]].
Check the top-left entry of A⁻¹A: 0.6·2 + (−0.2)·1 = 1.0 ✓.

## 4. Solving linear systems

The equations 2x₁ + x₂ = 4 and x₁ + 3x₂ = 7 are the single matrix equation A**x** = **b** with **b** = (4, 7).
Multiply both sides by A⁻¹: **x** = A⁻¹**b**.

**Worked example 5.** **x** = [[0.6, −0.2], [−0.2, 0.4]] (4, 7) = (2.4 − 1.4, −0.8 + 2.8) = **(1, 2)**.
That matches worked example 2, where A(1, 2) = (4, 7). ✓

In code, use `np.linalg.solve(A, b)`, not `np.linalg.inv(A) @ b`: it's faster and more accurate.

## 5. A dataset as a matrix, and a preview of regression

Rows are observations, columns are variables. To fit a line y = β₀ + β₁x we add a column of 1s (for the intercept).
Three students studied 2, 4 and 6 hours and scored 3, 5 and 7 points:

$$X = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 3 \\ 5 \\ 7 \end{pmatrix}.$$

In Module 06 you'll derive that the least-squares coefficients solve the **normal equations** (XᵀX)**β** = Xᵀ**y**.

**Worked example 6.**

- XᵀX = [[1+1+1, 2+4+6], [2+4+6, 4+16+36]] = [[3, 12], [12, 56]]
- Xᵀ**y** = (3 + 5 + 7, 2·3 + 4·5 + 6·7) = (15, 68)
- det(XᵀX) = 3·56 − 12·12 = 24, so (XᵀX)⁻¹ = (1/24)[[56, −12], [−12, 3]]
- **β** = (1/24)(56·15 − 12·68, −12·15 + 3·68) = (1/24)(24, 24) = **(1, 1)**

The fitted line is y = 1 + 1·x. Check: 1 + 2 = 3, 1 + 4 = 5, 1 + 6 = 7 ✓ (these three points lie exactly on it).

Look at what XᵀX contains: n = 3, Σxᵢ = 12 and Σxᵢ² = 56. **The summaries from Lesson 01.01 sit inside these
matrices.** That's why matrix notation is so compact.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| AB = BA | Not in general; order matters |
| Multiplying (3 × 2)(3 × 2) | Inner sizes must match: (3 × 2)(2 × 3) works |
| `A * B` in NumPy for matrix multiplication | `A @ B`; `*` is entry-wise |
| (AB)ᵀ = AᵀBᵀ | **(AB)ᵀ = BᵀAᵀ** (order reverses) |
| Inverting with `inv` to solve a system | `np.linalg.solve(A, b)` |

## Summary

- Dot product **u**·**v** = Σuᵢvᵢ; length ‖**u**‖ = √(**u**·**u**); cosine similarity = **u**·**v** / (‖**u**‖‖**v**‖).
- Matrix × vector = dot product of each row with the vector; (m × n)(n × p) → (m × p).
- AB ≠ BA; (AB)ᵀ = BᵀAᵀ; A⁻¹ exists only if det ≠ 0.
- A dataset is a matrix; regression solves (XᵀX)**β** = Xᵀ**y**.

## Self-test

1. Compute (1, 0, −1)·(2, 5, 2). What does the answer say about the angle between them?
2. What is the shape of the product of a 5 × 3 and a 3 × 4 matrix? Can you multiply them in the other order?
3. Invert [[4, 2], [1, 1]].
4. Why does a dataset matrix get a column of 1s before fitting a line?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
