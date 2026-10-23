"""03.03 Conditional probability and independence — worked examples with simulation checks.

Run:  python curriculum/03-probability/03-conditional-probability-independence/examples.py
"""

from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
rng = np.random.default_rng(3)
S2 = list(product(range(1, 7), repeat=2))


def conditional(a, b, space):
    """P(A | B) by counting: |A and B| / |B|."""
    in_b = [o for o in space if b(o)]
    return Fraction(sum(1 for o in in_b if a(o)), len(in_b))


print("Worked example 1: two dice")
print("  P(sum = 8 | first die = 3) =", conditional(lambda o: sum(o) == 8, lambda o: o[0] == 3, S2))
print("  P(first die = 3 | sum = 8) =", conditional(lambda o: o[0] == 3, lambda o: sum(o) == 8, S2))
print("  P(sum = 8)                 =", Fraction(sum(1 for o in S2 if sum(o) == 8), 36))
dice = rng.integers(1, 7, size=(300_000, 2))
given = dice[:, 0] == 3
print(f"  simulated P(sum = 8 | first = 3) = {(dice[given].sum(axis=1) == 8).mean():.4f}")

print("\nWorked example 2: conditional proportions from students.csv")
df = pd.read_csv(ROOT / "data" / "students.csv")
science = df["program"] == "Science"
tutoring = df["tutoring"] == "yes"
print(f"  P(tutoring | Science) = {(science & tutoring).sum()}/{science.sum()} = {(science & tutoring).sum() / science.sum():.3f}")
print(f"  P(Science | tutoring) = {(science & tutoring).sum()}/{tutoring.sum()} = {(science & tutoring).sum() / tutoring.sum():.3f}")

print("\nWorked example 3: multiplication rule — two aces without replacement")
exact = Fraction(4, 52) * Fraction(3, 51)
print(f"  4/52 * 3/51 = {exact} ≈ {float(exact):.5f}")
deck = np.array([1] * 4 + [0] * 48)
two = np.array([rng.choice(deck, 2, replace=False).sum() == 2 for _ in range(200_000)])
print(f"  simulated: {two.mean():.5f}")
print(f"  WITH replacement it would be (4/52)^2 = {float(Fraction(4, 52) ** 2):.5f}")

print("\nWorked example 4: independence checks on one die")
S = range(1, 7)
P = lambda e: Fraction(sum(1 for o in S if e(o)), 6)
even = lambda o: o % 2 == 0
low = lambda o: o <= 2
big = lambda o: o >= 4
for name, b in [("{1, 2}", low), ("{4, 5, 6}", big)]:
    joint = P(lambda o: even(o) and b(o))
    print(f"  A = even, B = {name}: P(A and B) = {joint}, P(A)P(B) = {P(even) * P(b)} -> "
          f"{'independent' if joint == P(even) * P(b) else 'dependent'}")
print("  disjoint events A = even, B = odd: P(A and B) = 0 but P(A)P(B) =", P(even) * P(lambda o: o % 2 == 1),
      "-> disjoint events are DEPENDENT")

print("\nWorked example 5: law of total probability — two machines")
p_m1, p_m2 = 0.60, 0.40
defect_m1, defect_m2 = 0.02, 0.05
p_defect = p_m1 * defect_m1 + p_m2 * defect_m2
print(f"  P(defect) = 0.6*0.02 + 0.4*0.05 = {p_m1 * defect_m1:.3f} + {p_m2 * defect_m2:.3f} = {p_defect:.3f}")
machine = rng.random(500_000) < p_m1
defect = np.where(machine, rng.random(500_000) < defect_m1, rng.random(500_000) < defect_m2)
print(f"  simulated: {defect.mean():.4f}")

print("\nWorked example 6: independence across repeated trials (gambler's fallacy)")
flips = rng.integers(0, 2, size=1_000_000)
after_five_heads = []
run = 0
for i, f in enumerate(flips[:-1]):
    run = run + 1 if f == 1 else 0
    if run >= 5:
        after_five_heads.append(flips[i + 1])
print(f"  after a run of 5+ heads, the next flip was heads {np.mean(after_five_heads):.4f} of the time "
      f"({len(after_five_heads):,} cases)")

print("\nWorked example 7: conditional independence")
# Two children's test scores depend on a shared factor (school quality) but not on each other within a school.
school_good = rng.random(200_000) < 0.5
p_pass = np.where(school_good, 0.9, 0.4)
a = rng.random(200_000) < p_pass
b = rng.random(200_000) < p_pass
print(f"  P(B passes)               = {b.mean():.3f}")
print(f"  P(B passes | A passes)    = {b[a].mean():.3f}   <- dependent overall")
for label, s in [("good", school_good), ("weak", ~school_good)]:
    print(f"  within {label} schools: P(B) = {b[s].mean():.3f}, P(B | A) = {b[s & a].mean():.3f}  <- independent given school")
