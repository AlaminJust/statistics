"""01.05 Vectors and matrices — worked examples in code.

Run:  python curriculum/01-math-toolkit/05-vectors-and-matrices/examples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

print("1. Vector operations")
print("   u + v =", u + v, "  2u =", 2 * u)
print("   u · v =", u @ v, "  (same as np.dot(u, v) =", np.dot(u, v), ")")
print("   ‖u‖   =", round(np.linalg.norm(u), 3), "  ‖v‖ =", round(np.linalg.norm(v), 3))
cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print("   cosine similarity =", round(cos, 3))
print("   perpendicular example: (1, 0)·(0, 1) =", np.array([1, 0]) @ np.array([0, 1]))

print("\n2. Matrix × vector")
A = np.array([[2, 1], [1, 3]])
x = np.array([1, 2])
print("   A @ x =", A @ x)
print("   row by row:", [int(row @ x) for row in A])

print("\n3. Matrix × matrix (order matters)")
P = np.array([[1, 2], [3, 4]])
Q = np.array([[5, 6], [7, 8]])
print("   P @ Q =\n", P @ Q)
print("   Q @ P =\n", Q @ P)
print("   P * Q (entry-wise, NOT matrix multiplication) =\n", P * Q)

print("\n   Shapes: (3x2) @ (2x4) ->", (np.ones((3, 2)) @ np.ones((2, 4))).shape)
try:
    np.ones((3, 2)) @ np.ones((3, 2))
except ValueError as error:
    print("   (3x2) @ (3x2) -> ValueError:", str(error)[:60], "...")

print("\n4. Transpose, identity, inverse")
X = np.array([[1, 2], [1, 4], [1, 6]])
print("   X shape", X.shape, "-> X.T shape", X.T.shape)
print("   det(A) =", round(np.linalg.det(A), 6))
A_inv = np.linalg.inv(A)
print("   A^-1 =\n", A_inv)
print("   A^-1 @ A =\n", (A_inv @ A).round(10) + 0.0)   # + 0.0 turns -0.0 into 0.0
print("   (PQ)^T == Q^T P^T ?", np.array_equal((P @ Q).T, Q.T @ P.T))

print("\n5. Solving A x = b")
b = np.array([4, 7])
print("   np.linalg.solve(A, b) =", np.linalg.solve(A, b))

print("\n6. Normal equations for a line through (2,3), (4,5), (6,7)")
y = np.array([3, 5, 7])
XtX = X.T @ X
Xty = X.T @ y
print("   X^T X =\n", XtX)
print("   X^T y =", Xty)
print("   beta  =", np.linalg.solve(XtX, Xty), "  -> y = 1 + 1·x")

print("\n7. The same idea on real-looking data: students.csv")
df = pd.read_csv(ROOT / "data" / "students.csv")
X_students = np.column_stack([np.ones(len(df)), df["study_hours"]])
beta = np.linalg.solve(X_students.T @ X_students, X_students.T @ df["exam_score"].to_numpy())
print(f"   exam_score ≈ {beta[0]:.2f} + {beta[1]:.2f} · study_hours")
print("   (the data were simulated with slope 2.2 — Module 06 explains why the estimate isn't exactly 2.2)")
