"""01.03 Derivatives and optimisation — worked examples in code.

Run:  python curriculum/01-math-toolkit/03-derivatives-and-optimization/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)


def forward_difference(f, x, h=1e-3):
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-3):
    return (f(x + h) - f(x - h)) / (2 * h)


print("Worked example 1: slope of x^2 at x = 3 (true answer 6)")
square = lambda x: x ** 2
for h in [0.1, 0.01, 0.001]:
    print(f"  h = {h:<6} forward = {forward_difference(square, 3, h):.6f}   central = {central_difference(square, 3, h):.6f}")

print("\nWorked examples 2-4: rules checked against numerical slopes")
checks = [
    ("(3x+1)^2 at x=0", lambda x: (3 * x + 1) ** 2, lambda x: 6 * (3 * x + 1), 0.0),
    ("exp(-x^2) at x=1", lambda x: math.exp(-x ** 2), lambda x: -2 * x * math.exp(-x ** 2), 1.0),
    ("sigmoid at z=0", lambda z: 1 / (1 + math.exp(-z)), lambda z: (1 / (1 + math.exp(-z))) * (1 - 1 / (1 + math.exp(-z))), 0.0),
]
for name, f, rule, x in checks:
    print(f"  {name:<18} rule = {rule(x):.6f}   numerical = {central_difference(f, x):.6f}")

print("\nThe mean minimises the sum of squared deviations")
data = np.array([4, 7, 1, 8])
S = lambda m: ((data - m) ** 2).sum()
for m in [4, 4.5, 5, 5.5, 6]:
    print(f"  S({m}) = {S(m)}")
grid = np.linspace(0, 10, 1001)
best_sq = grid[np.argmin([S(m) for m in grid])]
A = lambda m: np.abs(data - m).sum()
best_abs = grid[np.argmin([A(m) for m in grid])]
print("  grid search: best m for squared error  =", best_sq, " (mean   =", data.mean(), ")")
print("  grid search: best m for absolute error =", best_abs, " (median =", np.median(data), ")")
print("  note: absolute error is flat for every m between 4 and 7, so any of them is a minimiser")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(grid, [S(m) for m in grid], label="Σ (x − m)²")
ax.axvline(data.mean(), color="grey", ls="--", lw=0.8)
ax.annotate("minimum at the mean, m = 5", (5, 30), xytext=(5.6, 90), arrowprops={"arrowstyle": "->"})
ax.set_title("Squared error is smallest at the mean")
ax.set_xlabel("candidate summary value m")
ax.set_ylabel("total squared error")
fig.tight_layout()
fig.savefig(FIGURES / "mean_minimises_squared_error.png", dpi=120)
plt.close(fig)

print("\nWorked example 6: partial derivatives of g(x, y) = x^2 + 3xy + y^2 at (1, 2)")
g = lambda x, y: x ** 2 + 3 * x * y + y ** 2
dgdx = central_difference(lambda x: g(x, 2), 1)
dgdy = central_difference(lambda y: g(1, y), 2)
print(f"  numerical gradient = ({dgdx:.4f}, {dgdy:.4f})   rules give (8, 7)")

print("\nWorked example 8: gradient descent on f(x) = (x - 3)^2 from x = 0")


def gradient_descent(grad, x0, learning_rate, steps):
    path = [x0]
    x = x0
    for _ in range(steps):
        x = x - learning_rate * grad(x)
        path.append(x)
    return path


grad = lambda x: 2 * (x - 3)
for lr in [0.1, 0.5, 1.0, 1.1]:
    path = gradient_descent(grad, 0.0, lr, 20)
    first = ", ".join(f"{v:.3f}" for v in path[1:5])
    print(f"  lr = {lr:<4} first steps: {first} ... after 20 steps: {path[-1]:.4g}")

fig, ax = plt.subplots(figsize=(6, 3.8))
for lr in [0.1, 0.5, 1.0]:
    ax.plot(gradient_descent(grad, 0.0, lr, 15), marker="o", ms=3, label=f"learning rate {lr}")
ax.axhline(3, color="grey", ls="--", lw=0.8)
ax.set_title("Gradient descent on (x − 3)²")
ax.set_xlabel("step")
ax.set_ylabel("x")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "gradient_descent_learning_rates.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
