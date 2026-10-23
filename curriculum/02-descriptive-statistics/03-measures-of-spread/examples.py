"""02.03 Measures of spread — worked examples in code.

Run:  python curriculum/02-descriptive-statistics/03-measures-of-spread/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")

A = np.array([70, 72, 75, 78, 80])
B = np.array([50, 60, 75, 90, 100])

print("1. Same mean, different spread")
for name, x in [("A", A), ("B", B)]:
    print(f"   class {name}: mean {x.mean():.0f}  range {np.ptp(x)}  variance {x.var(ddof=1):.1f}  sd {x.std(ddof=1):.2f}")

print("\n2. Worked example 1, step by step (class A)")
deviations = A - A.mean()
print("   deviations        :", deviations, " sum =", deviations.sum())
print("   squared deviations:", deviations ** 2, " sum =", (deviations ** 2).sum())
print("   s^2 = 68 / 4 =", (deviations ** 2).sum() / (len(A) - 1))

print("\n3. ddof: NumPy and pandas disagree by default")
x = np.array([4, 7, 1, 8])
print("   np.var(x)            =", np.var(x), "  (divides by n)")
print("   np.var(x, ddof=1)    =", np.var(x, ddof=1), " (divides by n - 1)")
print("   pd.Series(x).var()   =", pd.Series(x).var(), " (pandas default ddof=1)")
print("   shortcut: (sum x^2 - n*mean^2)/(n-1) =", ((x ** 2).sum() - len(x) * x.mean() ** 2) / (len(x) - 1))

print("\n4. Why n - 1: 100,000 samples of size 5 from a population with variance 4")
rng = np.random.default_rng(0)
samples = rng.normal(loc=0, scale=2, size=(100_000, 5))
var_n = samples.var(axis=1, ddof=0)
var_n1 = samples.var(axis=1, ddof=1)
print(f"   average estimate dividing by n    : {var_n.mean():.3f}  (expected 4 * 4/5 = 3.2)")
print(f"   average estimate dividing by n - 1: {var_n1.mean():.3f}  (expected 4)")

print("   Bias shrinks as n grows:")
for n in [2, 5, 10, 30, 100]:
    s = rng.normal(0, 2, size=(20_000, n))
    print(f"     n = {n:<4} ddof=0 avg {s.var(axis=1, ddof=0).mean():.3f}   ddof=1 avg {s.var(axis=1, ddof=1).mean():.3f}")

print("\n5. Robust spread: MAD vs sd when an outlier appears")
salaries = np.array([32, 35, 38, 40, 41, 45])
with_outlier = np.append(salaries, 250)
for label, data in [("without 250", salaries), ("with 250", with_outlier)]:
    print(f"   {label:<12} sd {data.std(ddof=1):6.2f}   MAD {stats.median_abs_deviation(data):.2f}")
print("   MAD of (4, 7, 1, 8) =", stats.median_abs_deviation(x))

print("\n6. Coefficient of variation in students.csv")
for column in ["study_hours", "sleep_hours", "exam_score"]:
    s, m = df[column].std(), df[column].mean()
    print(f"   {column:<12} mean {m:6.2f}  sd {s:5.2f}  CV {s / m:.3f}")

print("\n7. Transformations")
minutes = df["commute_minutes"]
hours = minutes / 60
print(f"   commute sd: {minutes.std():.2f} minutes = {hours.std():.4f} hours (= {minutes.std() / 60:.4f})")
print(f"   adding 10 minutes to everyone: sd {(minutes + 10).std():.2f} (unchanged)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, (name, data) in zip(axes, [("A", A), ("B", B)]):
    ax.scatter(data, np.zeros_like(data), s=80, zorder=3)
    ax.axvline(data.mean(), color="black", lw=1)
    ax.axvspan(data.mean() - data.std(ddof=1), data.mean() + data.std(ddof=1), alpha=0.15)
    ax.set_xlim(40, 110)
    ax.set_yticks([])
    ax.set_title(f"Class {name}: mean 75, sd {data.std(ddof=1):.1f}")
    ax.set_xlabel("score")
fig.suptitle("Shaded band = mean ± 1 standard deviation")
fig.tight_layout()
fig.savefig(FIGURES / "same_mean_different_spread.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(var_n, bins=80, range=(0, 15), alpha=0.5, label=f"÷ n   (mean {var_n.mean():.2f})")
ax.hist(var_n1, bins=80, range=(0, 15), alpha=0.5, label=f"÷ n−1 (mean {var_n1.mean():.2f})")
ax.axvline(4, color="black", ls="--", label="true variance 4")
ax.set_title("Variance estimates from samples of size 5")
ax.set_xlabel("estimated variance")
ax.set_ylabel("number of samples")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "bessel_correction.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
