"""00.02 Solutions — run: python curriculum/00-setup/02-python-for-data/solutions/solutions.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "students.csv"
OUT = Path(__file__).parent / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)

print("--- 1")
x = np.array([4, 8, 15, 16, 23, 42])
print(x * 10)                 # [ 40  80 150 160 230 420]
print(x - x.mean())           # mean is 18 -> [-14 -10 -3 -2 5 24]
print(x[x > 10])              # [15 16 23 42]

print("--- 2")
print((x % 2 == 0).mean())    # 4 evens of 6 -> 0.666...

print("--- 3")
print(df.shape, list(df.columns))

print("--- 4")
print(len(df[(df["year"] == 4) & (df["study_hours"] > 10)]))

print("--- 5")
print(df["tutoring"].value_counts(normalize=True)["yes"])   # 0.3
print((df["tutoring"] == "yes").mean())                      # same: 0.3

print("--- 6")
means = df.groupby("tutoring")["exam_score"].mean()
print(means.round(2))                                        # no 61.46, yes 69.68
print("difference:", round(means["yes"] - means["no"], 2))   # about 8.2 points

print("--- 7")
print(df.groupby("year")["sleep_hours"].mean().sort_values(ascending=False).round(2))

print("--- 8")
tests = np.array([[70, 80, 90], [60, 65, 70], [85, 95, 75]])
print("per test   :", tests.mean(axis=0).round(2))   # [71.67 80. 78.33]
print("per student:", tests.mean(axis=1).round(2))   # [80. 65. 85.]

print("--- 9")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["commute_minutes"], bins=20, edgecolor="white")
ax.set_title("Commute time (200 students)")
ax.set_xlabel("Minutes")
ax.set_ylabel("Number of students")
fig.savefig(OUT / "commute_hist.png", dpi=120)
plt.close(fig)
print("Strongly right-skewed: most commutes are short, a few are very long (up to 108 min).")

print("--- 10")
print("Much weaker than study_hours: correlation",
      round(df["sleep_hours"].corr(df["exam_score"]), 3), "vs",
      round(df["study_hours"].corr(df["exam_score"]), 3))

print("--- 11")
scores = np.loadtxt(DATA, delimiter=",", skiprows=1, usecols=7)
print(scores.mean(), df["exam_score"].mean())

print("--- 12")
table = (
    df.assign(tut=df["tutoring"] == "yes")
    .groupby("program")
    .agg(students=("student_id", "count"),
         mean_study=("study_hours", "mean"),
         mean_score=("exam_score", "mean"),
         share_tutoring=("tut", "mean"))
    .round(2)
)
print(table)

print("--- 13: see curriculum/00-setup/solutions/statkit_basics_reference.py")
