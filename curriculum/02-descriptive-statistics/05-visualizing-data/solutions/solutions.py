"""02.05 Solutions. Run: python curriculum/02-descriptive-statistics/05-visualizing-data/solutions/solutions.py

Concept answers:
 1. (a) histogram, box plot or ECDF (incomes are right-skewed: consider a log axis); (b) bar chart in year order;
    (c) scatter plot; (d) line chart, perhaps with a 7-day moving average; (e) strip plot, or box plot with points.
 2. Steps of 0.2 at 2, then 0.4 at 5 (two values), then 0.2 at 7 and 0.2 at 10: heights 0.2, 0.6, 0.8, 1.0.
    The median is where the curve first reaches 0.5: x = 5.
 3. About 15 bins: 8 hides detail and 40 is noisy.
 4. The histogram and box plot show the skew most clearly. The ECDF answers the "70% commute less than ___" question:
    read across from 0.7 to the curve (about 27 minutes).
 6. Misleading features: a truncated y-axis, counts instead of proportions (groups differ in size), unsorted bars,
    no labels.
 7. Weak: the trend lines are nearly flat (correlation about 0.13).
 9. The two groups average about 6 and 16 hours. The overall mean (about 8.4) lies between the peaks, where few students
    actually are; the median (about 6.4) only describes the larger group.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")
sns.set_palette("colorblind")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print("saved", name)


# 3
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, bins in zip(axes, [8, 15, 40]):
    ax.hist(df["sleep_hours"], bins=bins, edgecolor="white")
    ax.set_title(f"{bins} bins")
    ax.set_xlabel("sleep hours per night")
    ax.set_ylabel("students")
save(fig, "ex3_sleep_bins.png")

# 4
c = df["commute_minutes"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(c, bins=20, edgecolor="white")
axes[0].set_title("Histogram: long right tail")
axes[1].boxplot(c, orientation="horizontal")
axes[1].set_yticks([])
axes[1].set_title("Box plot: many upper outliers")
x = np.sort(c)
axes[2].step(x, np.arange(1, len(x) + 1) / len(x), where="post")
p70 = np.percentile(c, 70)
axes[2].axhline(0.7, color="grey", ls="--", lw=0.8)
axes[2].axvline(p70, color="grey", ls="--", lw=0.8)
axes[2].set_title(f"ECDF: 70% commute ≤ {p70:.0f} min")
for ax in axes:
    ax.set_xlabel("commute minutes")
save(fig, "ex4_commute_three_views.png")

# 5
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="year", y="study_hours", color="white", ax=ax, showfliers=False)
sns.stripplot(data=df, x="year", y="study_hours", alpha=0.5, jitter=0.25, ax=ax)
ax.set_title("Study hours are similar in every year")
ax.set_xlabel("year of study")
ax.set_ylabel("study hours per week")
save(fig, "ex5_study_by_year.png")
print(df.groupby("year")["study_hours"].median().round(2).to_dict())

# 6
share = (df.assign(tut=df["tutoring"] == "yes").groupby("program")["tut"].mean().sort_values(ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(share.index, share.values)
axes[0].set_ylim(0, 1)
axes[0].set_title("Honest: share attending tutoring")
axes[0].set_ylabel("proportion of students")
for i, v in enumerate(share.values):
    axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center")
counts = df[df["tutoring"] == "yes"]["program"].value_counts().sort_index()
axes[1].bar(counts.index, counts.values, color="tab:red")
axes[1].set_ylim(counts.min() - 3, counts.max() + 1)
axes[1].set_title("Misleading: counts on a truncated axis")
save(fig, "ex6_tutoring_share.png")
print(share.round(3).to_dict())

# 7
fig, ax = plt.subplots(figsize=(6.5, 4))
for label, group in df.groupby("tutoring"):
    ax.scatter(group["sleep_hours"], group["exam_score"], alpha=0.5, label=f"tutoring = {label}")
    b1, b0 = np.polyfit(group["sleep_hours"], group["exam_score"], 1)
    xs = np.linspace(4, 10, 20)
    ax.plot(xs, b0 + b1 * xs)
ax.set_title("Sleep is only weakly related to exam score")
ax.set_xlabel("sleep hours per night")
ax.set_ylabel("exam score")
ax.legend()
save(fig, "ex7_sleep_vs_score.png")
print("corr sleep-score:", round(df["sleep_hours"].corr(df["exam_score"]), 3))

# 9
rng = np.random.default_rng(5)
bimodal = np.concatenate([rng.normal(6, 1.5, 150), rng.normal(16, 2, 50)])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(bimodal, bins=30, edgecolor="white")
ax.axvline(bimodal.mean(), color="tab:blue", lw=2, label=f"mean {bimodal.mean():.1f}")
ax.axvline(np.median(bimodal), color="tab:orange", lw=2, ls="--", label=f"median {np.median(bimodal):.1f}")
ax.set_title("Two groups: the mean describes neither")
ax.set_xlabel("study hours per week")
ax.set_ylabel("students")
ax.legend()
save(fig, "ex9_bimodal.png")


# 12
def quick_eda(data, column, out_dir=OUT):
    values = data[column].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(values, bins="fd", edgecolor="white")
    axes[0, 0].set_title(f"{column}: histogram")
    axes[0, 1].boxplot(values, orientation="horizontal")
    axes[0, 1].set_yticks([])
    axes[0, 1].set_title("box plot")
    xs = np.sort(values)
    axes[1, 0].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post")
    axes[1, 0].set_title("ECDF")
    stats_table = values.describe().round(2)
    stats_table["skew"] = round(values.skew(), 2)
    axes[1, 1].axis("off")
    axes[1, 1].table(cellText=[[k, v] for k, v in stats_table.items()], colLabels=["statistic", "value"],
                     loc="center", cellLoc="left")
    fig.suptitle(f"Quick EDA — {column}")
    fig.tight_layout()
    fig.savefig(out_dir / f"ex12_quick_eda_{column}.png", dpi=110)
    plt.close(fig)


for col in ["study_hours", "sleep_hours", "commute_minutes", "exam_score"]:
    quick_eda(df, col)
print("saved ex12 quick EDA figures")
