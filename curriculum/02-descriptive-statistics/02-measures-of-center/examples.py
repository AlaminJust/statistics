"""02.02 Measures of centre — worked examples in code.

Run:  python curriculum/02-descriptive-statistics/02-measures-of-center/examples.py
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

salaries = np.array([32, 35, 38, 40, 41, 45, 250])

print("Worked examples 1-2: mean vs median, with and without the founder")
print(f"  with founder   : mean {salaries.mean():.2f}  median {np.median(salaries):.1f}")
print(f"  without founder: mean {salaries[:-1].mean():.2f}  median {np.median(salaries[:-1]):.1f}")
print("  employees earning less than the mean:", (salaries < salaries.mean()).sum(), "of", len(salaries))

print("\nWorked example 3: mode")
values = [2, 3, 3, 5, 7, 7, 7, 9]
print("  mode of", values, "=", stats.mode(values, keepdims=False).mode)
print("  modes of commute_minutes (ties):", df["commute_minutes"].mode().tolist()[:6], "...")

print("\nMean vs median in students.csv")
summary = df[["exam_score", "sleep_hours", "study_hours", "commute_minutes"]].agg(["mean", "median"]).T.round(2)
summary["mean - median"] = (summary["mean"] - summary["median"]).round(2)
print(summary)

print("\nWorked example 4: weighted mean (GPA)")
marks = np.array([80, 70, 95])
credits = np.array([3, 4, 1])
print("  weighted:", np.average(marks, weights=credits), "  unweighted:", round(marks.mean(), 2))

print("\nCombining group means correctly")
groups = df.groupby("program")["exam_score"].agg(["count", "mean"])
print(groups.round(2))
weighted_overall = np.average(groups["mean"], weights=groups["count"])
print(f"  weighted overall {weighted_overall:.2f} = true overall {df['exam_score'].mean():.2f};"
      f" unweighted average of group means {groups['mean'].mean():.2f} (wrong)")

print("\nWorked example 5: 20% trimmed mean")
print("  scipy trim_mean:", stats.trim_mean(salaries, 0.2))
trimmed = np.sort(salaries)[1:-1]
print("  by hand: drop one from each end ->", trimmed, "mean", trimmed.mean())

print("\nWorked example 6: geometric mean of growth factors 1.5 and 0.5")
factors = np.array([1.5, 0.5])
gm = np.exp(np.log(factors).mean())
print(f"  arithmetic mean of rates: {(factors - 1).mean():+.0%}   geometric mean factor: {gm:.4f} ({gm - 1:+.1%} per year)")
print("  100 -> 150 -> 75; check 100 * gm**2 =", round(100 * gm ** 2, 2))

print("\nHow one outlier moves each measure: add a growing value to the salaries")
for extra in [50, 250, 1000, 10000]:
    data = np.append(salaries[:-1], extra)
    print(f"  top salary {extra:>6}: mean {data.mean():>8.2f}  median {np.median(data):>5.1f}  "
          f"20% trimmed {stats.trim_mean(data, 0.2):>5.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, column in zip(axes, ["exam_score", "commute_minutes"]):
    ax.hist(df[column], bins=25, color="lightgrey", edgecolor="white")
    ax.axvline(df[column].mean(), color="tab:blue", lw=2, label=f"mean {df[column].mean():.1f}")
    ax.axvline(df[column].median(), color="tab:orange", lw=2, ls="--", label=f"median {df[column].median():.1f}")
    ax.set_xlabel(column)
    ax.set_ylabel("students")
    ax.legend()
axes[0].set_title("Symmetric: mean ≈ median")
axes[1].set_title("Right-skewed: mean pulled toward the tail")
fig.tight_layout()
fig.savefig(FIGURES / "mean_vs_median.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "mean_vs_median.png")
