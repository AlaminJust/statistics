"""03.01 Sample spaces and axioms — worked examples, each checked by simulation.

Run:  python curriculum/03-probability/01-sample-spaces-and-axioms/examples.py
"""

from fractions import Fraction
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def prob(event, space):
    """Classical probability: favourable outcomes / all outcomes (equally likely)."""
    return Fraction(len(event & space), len(space))


print("Worked example 1: one die")
S = {1, 2, 3, 4, 5, 6}
even = {2, 4, 6}
big = {5, 6}
print("  P(even)            =", prob(even, S))
print("  P(> 4)             =", prob(big, S))
print("  P(even and > 4)    =", prob(even & big, S), "  outcomes:", even & big)
print("  P(even or > 4)     =", prob(even | big, S), "  outcomes:", even | big)
print("  addition rule      =", prob(even, S) + prob(big, S) - prob(even & big, S))
print("  P(not even)        =", prob(S - even, S), "= 1 - P(even)")

rolls = rng.integers(1, 7, size=100_000)
print(f"  simulated P(even or > 4) with 100,000 rolls: {np.isin(rolls, list(even | big)).mean():.4f}"
      f"  (exact {float(prob(even | big, S)):.4f})")

print("\nWorked example 2: two dice, the sample space of 36 pairs")
S2 = set(product(range(1, 7), repeat=2))
sums = {}
for a, b in S2:
    sums[a + b] = sums.get(a + b, 0) + 1
for total in range(2, 13):
    print(f"  sum {total:>2}: {sums[total]} outcomes, P = {Fraction(sums[total], 36)}")
print("  P(sum = 7) =", Fraction(sums[7], 36), " P(doubles) =", prob({(a, a) for a in range(1, 7)}, S2))

two = rng.integers(1, 7, size=(100_000, 2)).sum(axis=1)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
values, counts = np.unique(two, return_counts=True)
ax.bar(values - 0.2, counts / len(two), width=0.4, label="simulated (100,000 throws)")
ax.bar(np.arange(2, 13) + 0.2, [sums[t] / 36 for t in range(2, 13)], width=0.4, label="exact (counting)")
ax.set_title("Sum of two dice: 7 is the most likely total")
ax.set_xlabel("sum")
ax.set_ylabel("probability")
ax.set_xticks(range(2, 13))
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "two_dice_sum.png", dpi=120)
plt.close(fig)

print("\nWorked example 3: complement rule — at least one six in four rolls")
exact = 1 - Fraction(5, 6) ** 4
four = rng.integers(1, 7, size=(200_000, 4))
print(f"  exact 1 - (5/6)^4 = {float(exact):.4f}   simulated {(four == 6).any(axis=1).mean():.4f}")

print("\nWorked example 4: cards — P(heart or king)")
deck = set(product(["hearts", "diamonds", "clubs", "spades"], list(range(2, 11)) + ["J", "Q", "K", "A"]))
hearts = {c for c in deck if c[0] == "hearts"}
kings = {c for c in deck if c[1] == "K"}
print("  P(heart) =", prob(hearts, deck), " P(king) =", prob(kings, deck), " P(heart and king) =", prob(hearts & kings, deck))
print("  P(heart or king) =", prob(hearts | kings, deck), f"≈ {float(prob(hearts | kings, deck)):.4f}")

print("\nRelative frequency settles down as the number of trials grows (law of large numbers preview)")
flips = rng.integers(0, 2, size=10_000)
running = np.cumsum(flips) / np.arange(1, len(flips) + 1)
for n in [10, 100, 1000, 10_000]:
    print(f"  after {n:>6} flips: share of heads = {running[n - 1]:.4f}")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(np.arange(1, len(flips) + 1), running)
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_title("The share of heads settles near 0.5")
ax.set_xlabel("number of flips (log scale)")
ax.set_ylabel("running share of heads")
fig.tight_layout()
fig.savefig(FIGURES / "running_frequency.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
