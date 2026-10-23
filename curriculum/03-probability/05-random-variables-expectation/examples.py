"""03.05 Random variables and expectation — worked examples with simulation checks.

Run:  python curriculum/03-probability/05-random-variables-expectation/examples.py
"""

from fractions import Fraction
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5)


def expectation(values, probs):
    return sum(v * p for v, p in zip(values, probs))


def variance(values, probs):
    mu = expectation(values, probs)
    return sum((v - mu) ** 2 * p for v, p in zip(values, probs))


print("Worked example 1: a fair die")
faces = list(range(1, 7))
probs = [Fraction(1, 6)] * 6
mu = expectation(faces, probs)
ex2 = expectation([f * f for f in faces], probs)
var = variance(faces, probs)
print(f"  E[X] = {mu} = {float(mu)}")
print(f"  E[X^2] = {ex2} = {float(ex2):.4f}   (E[X])^2 = {mu ** 2} = {float(mu ** 2)}   -> E[X^2] != (E[X])^2")
print(f"  Var(X) = E[X^2] - (E[X])^2 = {var} = {float(var):.4f}   sd = {float(var) ** 0.5:.4f}")
print("  CDF:", {x: str(sum(probs[:x])) for x in faces})
rolls = rng.integers(1, 7, size=500_000)
print(f"  simulated mean {rolls.mean():.4f}, variance {rolls.var():.4f}")

print("\nWorked example 2: is this game worth playing?")
values = [10 - 2, 0 - 2]          # win 10 (pay 2) or win nothing (pay 2)
p_win = [0.1, 0.9]
print(f"  E[profit] = 8*0.1 + (-2)*0.9 = {expectation(values, p_win):.2f} per game")
games = np.where(rng.random(1_000_000) < 0.1, 8, -2)
print(f"  1,000,000 simulated games: average profit {games.mean():.4f}")

print("\nWorked example 3: linearity of expectation")
two = rng.integers(1, 7, size=(500_000, 2))
print(f"  E[sum of two dice] = 3.5 + 3.5 = 7;  simulated {two.sum(axis=1).mean():.4f}")
print(f"  E[3X + 2] = 3*3.5 + 2 = {3 * 3.5 + 2};  simulated {(3 * rolls + 2).mean():.4f}")
print(f"  Var(3X + 2) = 9 * Var(X) = {9 * float(var):.4f};  simulated {(3 * rolls + 2).var():.4f}")

print("\nWorked example 4: the indicator trick — fixed points of a random shuffle")
for n in [3, 4, 5, 6]:
    perms = list(permutations(range(n)))
    avg = Fraction(sum(sum(p[i] == i for i in range(n)) for p in perms), len(perms))
    print(f"  n = {n}: average number of letters in the right envelope = {avg}")
shuffles = np.array([np.sum(rng.permutation(52) == np.arange(52)) for _ in range(50_000)])
print(f"  52 cards, 50,000 shuffles: average fixed points {shuffles.mean():.4f}  (theory: exactly 1)")

print("\nWorked example 5: Bernoulli and binomial means by indicators")
p = 0.3
print(f"  Bernoulli(0.3): E = {p}, Var = p(1-p) = {p * (1 - p):.2f}")
print(f"  number of heads in 20 flips with p = 0.3: E = 20*0.3 = {20 * p}, Var = 20*0.21 = {20 * p * (1 - p):.1f}")
heads = (rng.random((200_000, 20)) < p).sum(axis=1)
print(f"  simulated: mean {heads.mean():.3f}, variance {heads.var():.3f}")

print("\nWorked example 6: E[g(X)] vs g(E[X]) for g(x) = 1/x on a die")
print(f"  E[1/X] = {float(expectation([1 / f for f in faces], [1 / 6] * 6)):.4f}   1/E[X] = {1 / 3.5:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(faces, [1 / 6] * 6, width=0.6)
axes[0].axvline(3.5, color="black", ls="--", label="E[X] = 3.5")
axes[0].set_title("PMF of a fair die")
axes[0].set_xlabel("x")
axes[0].set_ylabel("P(X = x)")
axes[0].legend()
axes[1].step([0] + faces + [7], [0] + [i / 6 for i in range(1, 7)] + [1], where="post")
axes[1].set_title("CDF: P(X ≤ x) jumps by 1/6 at each face")
axes[1].set_xlabel("x")
axes[1].set_ylabel("P(X ≤ x)")
fig.tight_layout()
fig.savefig(FIGURES / "die_pmf_cdf.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "die_pmf_cdf.png")
