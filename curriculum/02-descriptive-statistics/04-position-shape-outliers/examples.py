"""02.04 Position, shape and outliers — worked examples in code.

Run:  python curriculum/02-descriptive-statistics/04-position-shape-outliers/examples.py
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

d = np.array([2, 4, 4, 5, 7, 9, 10, 12, 15, 30])

print("Worked example 1: percentiles by linear interpolation")
for p in [0.25, 0.50, 0.75, 0.90]:
    h = (len(d) - 1) * p
    lo = int(np.floor(h))
    frac = h - lo
    by_hand = d[lo] + frac * (d[min(lo + 1, len(d) - 1)] - d[lo])
    print(f"  p = {p:.2f}: h = {h:.2f} -> by hand {by_hand:.3f}   np.percentile {np.percentile(d, p * 100):.3f}")
print("  'median of halves' textbook method: Q1 =", np.median(d[:5]), " Q3 =", np.median(d[5:]))
print("  other NumPy methods for Q1:",
      {m: float(np.percentile(d, 25, method=m)) for m in ["linear", "lower", "higher", "midpoint", "weibull"]})

print("\nWorked example 2: IQR, five-number summary and fences")
q1, med, q3 = np.percentile(d, [25, 50, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"  five-number summary: {d.min()}, {q1}, {med}, {q3}, {d.max()}")
print(f"  IQR = {iqr}   fences = ({lower}, {upper})   flagged: {d[(d < lower) | (d > upper)]}")

print("\nOutliers in commute_minutes by the 1.5 x IQR rule")
c = df["commute_minutes"]
cq1, cq3 = c.quantile([0.25, 0.75])
c_upper = cq3 + 1.5 * (cq3 - cq1)
print(f"  Q1 {cq1}, Q3 {cq3}, upper fence {c_upper:.2f}, flagged {(c > c_upper).sum()} students:",
      sorted(c[c > c_upper].tolist()))

print("\nWorked example 3: z-scores compare across scales")
print(f"  you    : z = (90 - 64) / 14 = {(90 - 64) / 14:.2f}")
print(f"  friend : z = (80 - 60) / 8  = {(80 - 60) / 8:.2f}")

z_d = (d - d.mean()) / d.std(ddof=1)
print(f"\n  z-scores of the example data (mean {d.mean()}, sd {d.std(ddof=1):.2f}):", z_d.round(2))
print("  |z| > 3 flags:", d[np.abs(z_d) > 3], " <- the outlier inflated the sd and hid itself")
robust_z = (d - np.median(d)) / (1.4826 * stats.median_abs_deviation(d))
print("  robust z (median, MAD):", robust_z.round(2), " flags:", d[np.abs(robust_z) > 3])

print("\nStandardising gives mean 0, sd 1")
z = (df["exam_score"] - df["exam_score"].mean()) / df["exam_score"].std()
print(f"  standardised exam_score: mean {z.mean():.2e}, sd {z.std():.3f}")

print("\nSkewness")
print(f"  example data g1 = {stats.skew(d):.3f}   (pandas adjusted G1 = {pd.Series(d).skew():.3f})")
for column in ["exam_score", "sleep_hours", "study_hours", "commute_minutes"]:
    print(f"  {column:<16} g1 = {stats.skew(df[column]):.3f}")
g1_by_hand = np.mean(((d - d.mean()) / d.std(ddof=0)) ** 3)
print(f"  by hand (mean of cubed z with population sd) = {g1_by_hand:.3f}")

print("\n68-95-99.7 rule check and Chebyshev's bound")
for column in ["sleep_hours", "exam_score", "commute_minutes"]:
    v = df[column]
    zz = (v - v.mean()) / v.std()
    shares = [f"{(zz.abs() <= k).mean():.1%}" for k in (1, 2, 3)]
    print(f"  {column:<16} within 1, 2, 3 sd: {shares}")
print("  Chebyshev guarantees at least:", [f"{1 - 1 / k**2:.1%}" for k in (2, 3)], "within 2 and 3 sd")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].boxplot([df["exam_score"], df["study_hours"] * 4], tick_labels=["exam_score", "study_hours × 4"])
axes[0].set_title("Box plots: symmetric vs right-skewed")
axes[0].set_ylabel("value")
axes[1].boxplot(df["commute_minutes"], orientation="horizontal")
axes[1].set_yticks([])
axes[1].set_title("commute_minutes: 11 points beyond the upper fence")
axes[1].set_xlabel("minutes")
fig.tight_layout()
fig.savefig(FIGURES / "box_plots.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "box_plots.png")
