"""01.01 Notation and summation — every worked example, in code.

Run:  python curriculum/01-math-toolkit/01-notation-and-summation/examples.py
"""

import math

import numpy as np

print("Worked example 1: sum over i = 1..4 of (2i + 1)")
terms = [2 * i + 1 for i in range(1, 5)]
print("  terms:", terms, " sum:", sum(terms))          # [3, 5, 7, 9]  24

print("\nWorked example 2: sums over data x = (4, 7, 1, 8)")
x = np.array([4, 7, 1, 8])
n = len(x)
print("  sum x_i       =", x.sum())                    # 20
print("  sum x_i^2     =", (x ** 2).sum())             # 130  square first
print("  (sum x_i)^2   =", x.sum() ** 2)               # 400  add first
print("  mean x-bar    =", x.mean())                   # 5.0

print("\nThe same sum written as the explicit loop that Σ stands for:")
total = 0
for i in range(1, n + 1):
    total += x[i - 1]
print("  loop total    =", total)

print("\nWorked example 3: deviations from the mean sum to zero")
deviations = x - x.mean()
print("  deviations    =", deviations, " sum:", deviations.sum())

# It works for ANY data, not just this example:
rng = np.random.default_rng(1)
random_data = rng.normal(50, 10, size=1000)
print("  random data   : sum of deviations =", round((random_data - random_data.mean()).sum(), 10))

print("\nWorked example 4: sum (x_i - x-bar)^2 = sum x_i^2 - n * x-bar^2")
left = ((x - x.mean()) ** 2).sum()
right = (x ** 2).sum() - n * x.mean() ** 2
print("  left =", left, " right =", right)               # 30.0  30.0

print("\nRules of summation, checked numerically")
y = np.array([1, 0, 2, 5])
c = 3
print("  sum c*x_i == c * sum x_i       :", (c * x).sum() == c * x.sum())
print("  sum (x_i + y_i) == sum x + sum y:", (x + y).sum() == x.sum() + y.sum())
print("  sum of constant c over n terms  :", sum(c for _ in range(n)), "== n*c =", n * c)
print("  sum x_i*y_i =", (x * y).sum(), " but (sum x)(sum y) =", x.sum() * y.sum(), " -> no product rule")

print("\nDouble sum: sum over i = 1..2, j = 1..3 of i*j")
double = sum(i * j for i in range(1, 3) for j in range(1, 4))
print("  nested loops:", double, "  factored (1+2)(1+2+3):", (1 + 2) * (1 + 2 + 3))   # 18

print("\nProduct notation")
print("  prod i for i = 1..4 =", math.prod(range(1, 5)))           # 24
print("  P(all three independent events) =", round(math.prod([0.5, 0.2, 0.1]), 10))   # 0.01
