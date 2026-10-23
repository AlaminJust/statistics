"""02.01 Solutions (code). Run: python curriculum/02-descriptive-statistics/01-data-and-variables/solutions/solutions.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("--- 3")
print(pd.DataFrame({"frequency": df["tutoring"].value_counts(),
                    "relative": df["tutoring"].value_counts(normalize=True)}))

print("--- 7")
table = df["program"].value_counts().sort_index().to_frame("frequency")
table["relative"] = table["frequency"] / table["frequency"].sum()
table["percent"] = 100 * table["relative"]
table.loc["Total"] = table.sum()
print(table.round(3))

print("--- 8")
years = df["year"].value_counts().sort_index().to_frame("frequency")
years["cumulative_relative"] = (years["frequency"] / len(df)).cumsum()
print(years.round(3))
print("year 3 or above:", round(1 - years.loc[2, "cumulative_relative"], 3))

print("--- 10")
fig, ax = plt.subplots(figsize=(7, 4))
for n in [10, 40, 160]:
    means = [df["exam_score"].sample(n, random_state=i).mean() for i in range(1000)]
    print(f"n = {n:<4} sd of sample means = {np.std(means):.2f}")
    ax.hist(means, bins=40, alpha=0.5, label=f"n = {n}")
ax.set_title("Bigger samples give less variable means")
ax.set_xlabel("sample mean exam score")
ax.set_ylabel("number of samples")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "sample_size_spread.png", dpi=120)
plt.close(fig)

print("--- 11")
wide = pd.DataFrame({"student": [1, 2, 3], "math_2024": [60, 70, 80], "math_2025": [65, 72, 90]})
tidy = wide.melt(id_vars="student", var_name="year", value_name="score")
tidy["year"] = tidy["year"].str.replace("math_", "").astype(int)
print(tidy)

print("--- 12")
true_mean = df["exam_score"].mean()
for label, scale in [("steep (/4)", 4), ("original (/8)", 8)]:
    rng = np.random.default_rng(11)
    p = 1 / (1 + np.exp(-(df["exam_score"] - 64) / scale))
    replied = rng.random(len(df)) < p
    print(f"{label:<14} repliers mean {df.loc[replied, 'exam_score'].mean():.2f} (true {true_mean:.2f})")
rng = np.random.default_rng(11)
replied = rng.random(len(df)) < 0.5
print(f"{'no dependence':<14} repliers mean {df.loc[replied, 'exam_score'].mean():.2f} (true {true_mean:.2f})")
