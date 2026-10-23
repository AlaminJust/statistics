"""03.02 Counting — worked examples, with brute-force checks.

Run:  python curriculum/03-probability/02-counting/examples.py
"""

import math
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

print("Worked example 1: multiplication rule")
print("  outfits, 3 shirts x 4 trousers         =", 3 * 4)
print("  4-letter passwords (a-z)              =", f"{26 ** 4:,}")
print("  4-character passwords (a-z, 0-9)      =", f"{36 ** 4:,}")
print("  8-character passwords (62 symbols)    =", f"{62 ** 8:,}")

print("\nWorked example 2: permutations — gold, silver, bronze among 8 runners")
print("  8 * 7 * 6 =", 8 * 7 * 6, "  math.perm(8, 3) =", math.perm(8, 3),
      "  brute force =", len(list(permutations(range(8), 3))))
print("  ways to order 5 books = 5! =", math.factorial(5))

print("\nWorked example 3: combinations — choose 3 of 8 runners for a team")
print("  C(8, 3) = 8!/(3! 5!) =", math.comb(8, 3), "  brute force =", len(list(combinations(range(8), 3))))
print("  each team can be ordered 3! = 6 ways: 336 / 6 =", 336 // 6)
print(f"  lottery 6 of 49: C(49, 6) = {math.comb(49, 6):,}  -> P(win) = {1 / math.comb(49, 6):.2e}")

print("\nPascal's triangle (row n lists C(n, 0) ... C(n, n))")
for n in range(7):
    print("  ", " ".join(f"{math.comb(n, k):>3}" for k in range(n + 1)))

print("\nWorked example 4: exactly 2 heads in 5 fair flips")
exact = Fraction(math.comb(5, 2), 2 ** 5)
brute = sum(1 for seq in product("HT", repeat=5) if seq.count("H") == 2)
print(f"  C(5, 2) / 2^5 = {exact} = {float(exact)}   (brute force: {brute} of 32 sequences)")

print("\nWorked example 5: 4 items from a box of 10 with 3 defective — P(exactly 2 defective)")
p = Fraction(math.comb(3, 2) * math.comb(7, 2), math.comb(10, 4))
print(f"  C(3,2) C(7,2) / C(10,4) = {math.comb(3, 2)} * {math.comb(7, 2)} / {math.comb(10, 4)} = {p} = {float(p)}")
box = np.array([1] * 3 + [0] * 7)
draws = np.array([rng.choice(box, size=4, replace=False).sum() for _ in range(50_000)])
print(f"  simulated: {(draws == 2).mean():.4f}")

print("\nWorked example 6: the birthday problem")


def p_shared(n, days=365):
    p_all_different = 1.0
    for i in range(n):
        p_all_different *= (days - i) / days
    return 1 - p_all_different


for n in [10, 23, 30, 50, 70]:
    print(f"  {n:>2} people: P(at least two share a birthday) = {p_shared(n):.4f}")
sims = rng.integers(0, 365, size=(20_000, 23))
shared = np.array([len(set(row)) < 23 for row in sims])
print(f"  simulated for 23 people: {shared.mean():.4f}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ns = np.arange(1, 81)
ax.plot(ns, [p_shared(n) for n in ns])
ax.axhline(0.5, color="grey", ls="--", lw=0.8)
ax.axvline(23, color="grey", ls="--", lw=0.8)
ax.set_title("With 23 people a shared birthday is more likely than not")
ax.set_xlabel("number of people")
ax.set_ylabel("P(at least one shared birthday)")
fig.tight_layout()
fig.savefig(FIGURES / "birthday_problem.png", dpi=120)
plt.close(fig)

print("\nCombinatorial explosion in ML")
for p_features in [10, 20, 40]:
    print(f"  {p_features} features -> {2 ** p_features:,} possible feature subsets")
grid = {"learning_rate": 5, "depth": 6, "n_estimators": 4, "subsample": 3}
print("  hyperparameter grid", grid, "->", math.prod(grid.values()), "combinations; x 5-fold CV =",
      5 * math.prod(grid.values()), "model fits")
print("\nFigure saved:", FIGURES / "birthday_problem.png")
