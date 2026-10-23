"""02.05 Visualising data — builds a gallery of the charts in the lesson.

Run:  python curriculum/02-descriptive-statistics/05-visualizing-data/examples.py
Then open the PNG files in curriculum/02-descriptive-statistics/05-visualizing-data/figures/generated/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=120)
    plt.close(fig)
    print("saved", name)


# 1. Histograms: the bin count changes the story ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
for ax, bins in zip(axes, [5, 20, 80]):
    ax.hist(df["study_hours"], bins=bins, edgecolor="white")
    ax.set_title(f"{bins} bins")
    ax.set_xlabel("study hours per week")
    ax.set_ylabel("students")
fig.suptitle("Too few bins hide the shape; too many show only noise")
save(fig, "01_histogram_bins.png")
edges = np.histogram_bin_edges(df["study_hours"], bins="fd")
print(f"   Freedman-Diaconis rule suggests {len(edges) - 1} bins for study_hours")

# 2. Four shapes of distributions ----------------------------------------------------------------
rng = np.random.default_rng(0)
shapes = {
    "symmetric (normal)": rng.normal(50, 10, 2000),
    "right-skewed (exponential)": rng.exponential(10, 2000),
    "left-skewed": 100 - rng.exponential(10, 2000),
    "bimodal (two groups mixed)": np.concatenate([rng.normal(35, 5, 1000), rng.normal(65, 5, 1000)]),
}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, (title, data) in zip(axes, shapes.items()):
    ax.hist(data, bins=40, edgecolor="white")
    ax.axvline(np.mean(data), color="tab:blue", lw=2, label="mean")
    ax.axvline(np.median(data), color="tab:orange", lw=2, ls="--", label="median")
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])
axes[0].legend(fontsize=8)
save(fig, "02_distribution_shapes.png")

# 3. Comparing a numeric variable across groups ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
order = ["Arts", "Commerce", "Science"]
sns.boxplot(data=df, x="program", y="exam_score", order=order, ax=axes[0], color="lightsteelblue")
axes[0].set_title("Box plot")
sns.violinplot(data=df, x="program", y="exam_score", order=order, ax=axes[1], color="lightsteelblue", inner="quartile")
axes[1].set_title("Violin plot (shape + quartiles)")
sns.stripplot(data=df, x="program", y="exam_score", order=order, ax=axes[2], alpha=0.5, jitter=0.25)
axes[2].set_title("Strip plot (every student)")
for ax in axes:
    ax.set_xlabel("")
axes[0].set_ylabel("exam score")
save(fig, "03_groups_box_violin_strip.png")

# 4. Bar charts: honest vs misleading ------------------------------------------------------------
means = df.groupby("program")["exam_score"].mean().reindex(order)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(means.index, means.values, color="tab:red")
axes[0].set_ylim(62.5, 65.5)
axes[0].set_title("Misleading: axis starts at 62.5")
axes[1].bar(means.index, means.values, color="tab:green")
axes[1].set_ylim(0, 100)
axes[1].set_title("Honest: bars start at 0")
for ax in axes:
    ax.set_ylabel("mean exam score")
fig.suptitle(f"The same three means ({', '.join(f'{m:.1f}' for m in means)})")
save(fig, "04_bar_truncated_axis.png")

counts = df["program"].value_counts()
fig, ax = plt.subplots(figsize=(5.5, 3.3))
ax.barh(counts.index[::-1], counts.values[::-1])
for y, value in enumerate(counts.values[::-1]):
    ax.text(value + 1, y, str(value), va="center")
ax.set_title("Students per program (sorted, labelled)")
ax.set_xlabel("students")
save(fig, "05_bar_categorical.png")

# 5. Scatter plots: overplotting, trend, third variable ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(df["study_hours"], df["exam_score"], alpha=0.5)
slope, intercept = np.polyfit(df["study_hours"], df["exam_score"], 1)
xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 50)
axes[0].plot(xs, intercept + slope * xs, color="black", label=f"trend: {intercept:.1f} + {slope:.2f}·hours")
axes[0].legend()
axes[0].set_title("Scatter with a trend line")
for label, group in df.groupby("tutoring"):
    axes[1].scatter(group["study_hours"], group["exam_score"], alpha=0.6, label=f"tutoring = {label}")
axes[1].legend()
axes[1].set_title("Colour adds a third variable")
for ax in axes:
    ax.set_xlabel("study hours per week")
    ax.set_ylabel("exam score")
save(fig, "06_scatter.png")

# 6. ECDF ----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
for label, group in df.groupby("tutoring"):
    x = np.sort(group["exam_score"])
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=f"tutoring = {label} (n = {len(x)})")
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_title("ECDF: read any percentile; no bins to choose")
ax.set_xlabel("exam score")
ax.set_ylabel("proportion of students ≤ score")
ax.legend()
save(fig, "07_ecdf.png")

small = np.array([3, 1, 4, 1, 5])
xs_small = np.sort(small)
print("   ECDF of (3, 1, 4, 1, 5):", list(zip(xs_small.tolist(), (np.arange(1, 6) / 5).tolist())))

# 7. Line chart for a time series ----------------------------------------------------------------
weeks = np.arange(1, 35)
hours = 6 + 0.1 * weeks + rng.normal(0, 1.2, len(weeks))
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(weeks, hours, marker="o", ms=3, label="weekly hours")
ax.plot(weeks, pd.Series(hours).rolling(4).mean(), lw=2.5, label="4-week moving average")
ax.set_title("Line chart: a quantity over time (simulated study log)")
ax.set_xlabel("course week")
ax.set_ylabel("hours studied")
ax.legend()
save(fig, "08_line_time_series.png")

# 8. Pair plot: quick look at every pair of numeric variables -------------------------------------
grid = sns.pairplot(df[["study_hours", "sleep_hours", "commute_minutes", "exam_score"]], corner=True,
                    plot_kws={"alpha": 0.4, "s": 12})
grid.figure.suptitle("Pair plot: every pair at a glance", y=1.02)
grid.figure.savefig(FIGURES / "09_pairplot.png", dpi=100, bbox_inches="tight")
plt.close(grid.figure)
print("saved 09_pairplot.png")

print("\nAll figures are in", FIGURES)
