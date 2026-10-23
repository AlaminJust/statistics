"""01.02 Functions, exponents and logarithms — worked examples in code.

Run:  python curriculum/01-math-toolkit/02-functions-exponents-logs/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)

print("1. Linear function f(x) = 40 + 2.2x at x = 10:", 40 + 2.2 * 10)

print("\n2. Exponent rules")
print("   2**3 * 2**2 =", 2**3 * 2**2, "= 2**5 =", 2**5)
print("   2**-3       =", 2**-3)
print("   9**0.5      =", 9**0.5)
print("   1e-6        =", 1e-6)

print("\n3. e and exp")
print("   e        =", math.e)
print("   exp(-1)  =", round(math.exp(-1), 3))

print("\n4. Logarithms and their rules")
print("   log10(1000) =", math.log10(1000), " log2(8) =", math.log2(8), " ln(e) =", math.log(math.e))
print("   ln(0.5)     =", round(math.log(0.5), 3), "(negative: input below 1)")
print("   ln2 + ln3   =", round(math.log(2) + math.log(3), 4), " ln6 =", round(math.log(6), 4))
print("   ln(2**10)   =", round(math.log(2**10), 4), " 10*ln2 =", round(10 * math.log(2), 4))

print("\n5a. Underflow: product of 200 probabilities of 0.01")
print("   0.01 ** 200        =", 0.01 ** 200, "  <- wrong! true value is 1e-400")
print("   200 * ln(0.01)     =", round(200 * math.log(0.01), 2), " <- the log of the true value")
print("   smallest float > 0 ≈", np.nextafter(0, 1))

print("\n5b. Logs tame right-skewed data (commute_minutes)")
df = pd.read_csv(ROOT / "data" / "students.csv")
commute = df["commute_minutes"]
log_commute = np.log1p(commute)          # log(1 + x): safe when x = 0
print("   skewness raw        :", round(commute.skew(), 2))
print("   skewness log(1 + x) :", round(log_commute.skew(), 2), " (closer to 0 = more symmetric)")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].hist(commute, bins=20, edgecolor="white")
axes[0].set_title("Commute minutes: long right tail")
axes[0].set_xlabel("minutes")
axes[1].hist(log_commute, bins=20, edgecolor="white", color="darkorange")
axes[1].set_title("log(1 + minutes): less skewed")
axes[1].set_xlabel("log(1 + minutes)")
for ax in axes:
    ax.set_ylabel("students")
fig.tight_layout()
fig.savefig(FIGURES / "log_transform.png", dpi=120)
plt.close(fig)

print("\n5c. Odds and log-odds for p = 0.8")
p = 0.8
odds = p / (1 - p)
print("   odds =", round(odds, 3), " log-odds =", round(math.log(odds), 4))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("\n6. Sigmoid")
for z in [-2, 0, 2, math.log(4)]:
    print(f"   sigmoid({z:.3f}) = {sigmoid(z):.4f}")

z = np.linspace(-6, 6, 200)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(z, sigmoid(z))
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_title("Sigmoid maps any score to a probability")
ax.set_xlabel("score z (log-odds)")
ax.set_ylabel("probability σ(z)")
fig.tight_layout()
fig.savefig(FIGURES / "sigmoid.png", dpi=120)
plt.close(fig)

print("\n7. Softmax and log-sum-exp")
scores = np.array([1.0, 2.0, 3.0])
softmax = np.exp(scores) / np.exp(scores).sum()
print("   softmax(1, 2, 3) =", softmax.round(4), " sums to", softmax.sum())

big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("   naive softmax(1000, 1001, 1002) =", naive, "  <- overflow")
m = big.max()
lse = m + np.log(np.exp(big - m).sum())
print("   log-sum-exp =", round(lse, 4))
print("   stable softmax =", np.exp(big - lse).round(4), " (same as softmax(1, 2, 3)!)")

print("\nFigures saved in", FIGURES.relative_to(ROOT))
