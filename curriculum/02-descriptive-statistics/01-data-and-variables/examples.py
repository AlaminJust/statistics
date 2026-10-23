"""02.01 Data and variables — worked examples in code.

Run:  python curriculum/02-descriptive-statistics/01-data-and-variables/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data" / "students.csv")

print("1. Column types as pandas sees them (it can't know which are categorical!)")
print(df.dtypes)

# Tell pandas what the variables really are.
typed = df.astype({"program": "category", "tutoring": "category"})
typed["year"] = pd.Categorical(df["year"], categories=[1, 2, 3, 4], ordered=True)
print("\nAfter declaring categorical types:")
print(typed.dtypes)

print("\n2. Frequency table for program")
table = pd.DataFrame({
    "frequency": df["program"].value_counts(),
    "relative": df["program"].value_counts(normalize=True).round(3),
})
table["percent"] = (table["relative"] * 100).round(1)
print(table)
print("relative frequencies sum to", table["relative"].sum())

print("\n3. Ordinal variable: year with cumulative relative frequency")
year_table = df["year"].value_counts().sort_index().to_frame("frequency")
year_table["relative"] = year_table["frequency"] / len(df)
year_table["cumulative"] = year_table["relative"].cumsum()
print(year_table.round(3))

print("\n4. Mode of a nominal variable:", df["program"].mode().tolist())

print("\n5. Sample vs population: treat the 200 students as the population")
population_mean = df["exam_score"].mean()
one_sample = df.sample(20, random_state=3)
print(f"   population mean (parameter): {population_mean:.2f}")
print(f"   one sample of 20 (statistic): {one_sample['exam_score'].mean():.2f}")
sample_means = np.array([df["exam_score"].sample(20, random_state=i).mean() for i in range(1000)])
print(f"   1000 samples of 20: means range from {sample_means.min():.1f} to {sample_means.max():.1f}, "
      f"centre {sample_means.mean():.2f}, spread (sd) {sample_means.std():.2f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(sample_means, bins=30, edgecolor="white")
ax.axvline(population_mean, color="black", ls="--", label=f"population mean {population_mean:.2f}")
ax.set_title("Means of 1,000 random samples of 20 students")
ax.set_xlabel("sample mean exam score")
ax.set_ylabel("number of samples")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "sampling_variability.png", dpi=120)
plt.close(fig)

print("\n6. Non-response bias in action")
# Suppose we email a survey asking for exam scores, and students who did well are more likely to reply.
rng = np.random.default_rng(11)
reply_probability = 1 / (1 + np.exp(-(df["exam_score"] - 64) / 8))     # sigmoid: higher score, likelier reply
replied = rng.random(len(df)) < reply_probability
print(f"   {replied.sum()} of {len(df)} students replied")
print(f"   mean score of repliers: {df.loc[replied, 'exam_score'].mean():.2f}   true mean: {population_mean:.2f}")
print("   A random sample of the same size has no such systematic gap:")
random_same_size = df["exam_score"].sample(int(replied.sum()), random_state=11).mean()
print(f"   random sample mean: {random_same_size:.2f}")

print("\n7. Observational association: tutoring vs exam score")
print(df.groupby("tutoring")["exam_score"].agg(["count", "mean"]).round(2))
print("   tutoring students also study more on average?",
      df.groupby("tutoring")["study_hours"].mean().round(2).to_dict())

print("\n8. Tidy vs untidy")
wide = pd.DataFrame({"student": [1, 2], "score_jan": [60, 72], "score_feb": [65, 70]})
print("   untidy (wide):\n", wide)
tidy = wide.melt(id_vars="student", var_name="month", value_name="score")
tidy["month"] = tidy["month"].str.replace("score_", "")
print("   tidy (long):\n", tidy)
