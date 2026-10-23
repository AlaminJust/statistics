"""01.04 Integrals and area — worked examples in code.

Run:  python curriculum/01-math-toolkit/04-integrals-and-area/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)


def riemann(f, a, b, n, where="left"):
    dx = (b - a) / n
    offset = {"left": 0.0, "mid": 0.5, "right": 1.0}[where]
    return sum(f(a + (i + offset) * dx) for i in range(n)) * dx


def trapezoid(f, a, b, n=1000):
    dx = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * dx)
    return total * dx


square = lambda x: x ** 2
print("Worked example 1: area under x^2 from 0 to 2 (exact 8/3 = 2.6667)")
for n in [4, 10, 100]:
    print(f"  n = {n:<4} left = {riemann(square, 0, 2, n, 'left'):.4f}  "
          f"mid = {riemann(square, 0, 2, n, 'mid'):.4f}  right = {riemann(square, 0, 2, n, 'right'):.4f}  "
          f"trapezoid = {trapezoid(square, 0, 2, n):.4f}")

print("\nWorked examples 2-4: exact vs numerical")
print("  ∫0^2 x^2 dx     exact", round(8 / 3, 4), " numerical", round(trapezoid(square, 0, 2), 4))
print("  ∫0^1 2x dx      exact 1        numerical", round(trapezoid(lambda x: 2 * x, 0, 1), 4))
print("  ∫0^inf e^-x dx  exact 1        numerical (0 to 50)", round(trapezoid(lambda x: math.exp(-x), 0, 50, 10000), 4))

print("\nWorked example 5: P(X <= 0.5) for density 2x on [0, 1]")
print("  exact 0.25   numerical", round(trapezoid(lambda x: 2 * x, 0, 0.5), 4))

print("\nWorked example 6: P(wait > 2) for density e^-x")
print("  exact e^-2 =", round(math.exp(-2), 4), "  numerical", round(trapezoid(lambda x: math.exp(-x), 2, 50, 10000), 4))

print("\nWorked example 7: E[X] for density 2x on [0, 1]")
print("  exact 2/3 =", round(2 / 3, 4), "  numerical", round(trapezoid(lambda x: x * 2 * x, 0, 1), 4))

# The same expectation by simulation: draw many X values and average them.
rng = np.random.default_rng(0)
u = rng.uniform(size=200_000)
samples = np.sqrt(u)       # if U is uniform on [0, 1], sqrt(U) has density 2x (you'll see why in Module 04)
print("  simulation mean of 200,000 draws =", round(samples.mean(), 4))

print("\nThe standard normal density integrates to 1")
normal_pdf = lambda x: math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
print("  ∫ from -10 to 10 =", round(trapezoid(normal_pdf, -10, 10, 10000), 6))
print("  P(-1 <= Z <= 1)  =", round(trapezoid(normal_pdf, -1, 1, 10000), 4), " (the famous 68%)")
print("  P(-2 <= Z <= 2)  =", round(trapezoid(normal_pdf, -2, 2, 10000), 4), " (about 95%; exactly 95% needs ±1.96)")

# Figure: rectangles under x^2
xs = np.linspace(0, 2, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, n in zip(axes, [4, 16]):
    dx = 2 / n
    lefts = np.arange(n) * dx
    ax.bar(lefts, (lefts + dx / 2) ** 2, width=dx, align="edge", alpha=0.35, edgecolor="black")
    ax.plot(xs, xs ** 2, color="black")
    ax.set_title(f"{n} midpoint rectangles: area ≈ {riemann(square, 0, 2, n, 'mid'):.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("x²")
fig.suptitle("More rectangles → closer to the exact area 8/3 ≈ 2.667")
fig.tight_layout()
fig.savefig(FIGURES / "riemann_rectangles.png", dpi=120)
plt.close(fig)

# Figure: probability as shaded area
fig, ax = plt.subplots(figsize=(6, 3.8))
grid = np.linspace(-4, 4, 400)
ax.plot(grid, [normal_pdf(g) for g in grid], color="black")
inner = grid[(grid >= -1) & (grid <= 1)]
ax.fill_between(inner, [normal_pdf(g) for g in inner], alpha=0.4)
ax.set_title("P(−1 ≤ Z ≤ 1) is the shaded area ≈ 0.683")
ax.set_xlabel("z")
ax.set_ylabel("density")
fig.tight_layout()
fig.savefig(FIGURES / "probability_as_area.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
