"""02.06 Two variables and correlation — worked examples in code.

Run:  python curriculum/02-descriptive-statistics/06-two-variables-correlation/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("Worked example 1: contingency table program x tutoring")
print(pd.crosstab(df["program"], df["tutoring"], margins=True, margins_name="total"))
print("\n  joint proportions:")
print(pd.crosstab(df["program"], df["tutoring"], normalize="all").round(3))
print("\n  conditional on program (rows sum to 1):")
print(pd.crosstab(df["program"], df["tutoring"], normalize="index").round(3))
print("\n  P(Science | tutoring) =", round(((df.program == "Science") & (df.tutoring == "yes")).sum() / (df.tutoring == "yes").sum(), 3))

print("\nNumeric by group")
print(df.groupby("tutoring")["exam_score"].agg(["count", "mean", "median", "std"]).round(2))

print("\nWorked example 2: covariance and correlation by hand")
x = np.array([2, 4, 6, 8, 10])
y = np.array([50, 60, 65, 80, 85])
dx, dy = x - x.mean(), y - y.mean()
print("  products of deviations:", dx * dy, " sum =", (dx * dy).sum())
cov = (dx * dy).sum() / (len(x) - 1)
r = cov / (x.std(ddof=1) * y.std(ddof=1))
print(f"  cov = {cov}   r = {r:.4f}   np.corrcoef = {np.corrcoef(x, y)[0, 1]:.4f}")
print(f"  in minutes: cov = {np.cov(x * 60, y)[0, 1]:.0f}, r = {np.corrcoef(x * 60, y)[0, 1]:.4f} (unchanged)")
cosine = dx @ dy / (np.linalg.norm(dx) * np.linalg.norm(dy))
print(f"  cosine similarity of centred vectors = {cosine:.4f} (same as r)")

print("\nCorrelation matrix of students.csv")
numeric = df[["study_hours", "sleep_hours", "commute_minutes", "exam_score"]]
print(numeric.corr().round(3))
r_study = df["study_hours"].corr(df["exam_score"])
print(f"  r(study, score) = {r_study:.3f}, r^2 = {r_study ** 2:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", square=True, ax=ax)
ax.set_title("Only study hours correlates strongly with score")
fig.tight_layout()
fig.savefig(FIGURES / "correlation_heatmap.png", dpi=120)
plt.close(fig)

print("\nAnscombe's quartet: identical summaries, different data")
ax_x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
quartet = {
    "I": (ax_x, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (ax_x, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ax_x, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True, sharey=True)
for ax, (name, (qx, qy)) in zip(axes, quartet.items()):
    qx, qy = np.array(qx), np.array(qy)
    slope, intercept = np.polyfit(qx, qy, 1)
    rq = np.corrcoef(qx, qy)[0, 1]
    print(f"  {name:<4} mean x {qx.mean():.1f}  mean y {qy.mean():.2f}  var y {qy.var(ddof=1):.2f}  "
          f"r {rq:.3f}  line y = {intercept:.2f} + {slope:.2f}x")
    ax.scatter(qx, qy)
    grid = np.linspace(3, 20, 10)
    ax.plot(grid, intercept + slope * grid, color="black", lw=1)
    ax.set_title(f"Set {name}: r = {rq:.2f}")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.suptitle("Anscombe's quartet: same mean, variance, r and line")
fig.tight_layout()
fig.savefig(FIGURES / "anscombe_quartet.png", dpi=120)
plt.close(fig)

print("\nr only measures linear association")
u = np.arange(-3, 4)
print("  y = x^2 on -3..3: r =", round(np.corrcoef(u, u ** 2)[0, 1], 3))
high = df[df["study_hours"] > 12]
print(f"  restricted range (study > 12 h, n = {len(high)}): r = {high['study_hours'].corr(high['exam_score']):.3f} "
      f"vs all students {r_study:.3f}")

print("\nWorked examples 3-4: Spearman rank correlation")
print("  ranks of (10, 20, 20, 40):", stats.rankdata([10, 20, 20, 40]))
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 100])
print(f"  Pearson {np.corrcoef(a, b)[0, 1]:.3f}   Spearman {stats.spearmanr(a, b).statistic:.3f}")
ra, rb = stats.rankdata(a), stats.rankdata(b)
print(f"  Spearman = Pearson on ranks: {np.corrcoef(ra, rb)[0, 1]:.3f}")

print("\nConfounding simulation: temperature drives ice cream AND drownings")
rng = np.random.default_rng(4)
temperature = rng.uniform(10, 35, 365)
ice_cream = 20 * temperature + rng.normal(0, 60, 365)
drownings = 0.3 * temperature + rng.normal(0, 1.5, 365)
print(f"  r(ice cream, drownings) = {np.corrcoef(ice_cream, drownings)[0, 1]:.2f}  <- strong, but no causal link")
hot = temperature > 30
print(f"  within hot days only (temperature held roughly fixed): r = {np.corrcoef(ice_cream[hot], drownings[hot])[0, 1]:.2f}")
print("\nFigures saved in", FIGURES)
