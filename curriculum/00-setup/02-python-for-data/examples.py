"""00.02 Python for data — runnable examples.

Run:  python curriculum/00-setup/02-python-for-data/examples.py
Figures are saved to curriculum/00-setup/02-python-for-data/figures/generated/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# ----------------------------------------------------------------------------
section("1. Lists vs NumPy arrays")
scores_list = [62, 75, 75, 80, 84, 88, 91, 97]
scores = np.array(scores_list)

print("list + 5      ->", "TypeError (lists can't add a number)")
print("array + 5     ->", scores + 5)
print("array * 2     ->", scores * 2)
print("list * 2      ->", scores_list * 2, " (repeats the list!)")
print("dtype, shape  ->", scores.dtype, scores.shape)

# ----------------------------------------------------------------------------
section("2. Indexing, slicing and boolean masks")
print("first         ->", scores[0])
print("last          ->", scores[-1])
print("first three   ->", scores[:3])
mask = scores >= 80
print("mask >= 80    ->", mask)
print("scores[mask]  ->", scores[mask])
print("how many >=80 ->", mask.sum(), "(True counts as 1)")
print("share >= 80   ->", mask.mean())

# ----------------------------------------------------------------------------
section("3. Aggregations")
print("sum  ->", scores.sum())
print("mean ->", scores.mean())
print("min, max ->", scores.min(), scores.max())
print("sorted descending ->", np.sort(scores)[::-1])

# ----------------------------------------------------------------------------
section("4. 2-D arrays (rows = students, columns = tests)")
tests = np.array([
    [70, 80, 90],
    [60, 65, 70],
    [85, 95, 75],
])
print("shape            ->", tests.shape)
print("mean per student ->", tests.mean(axis=1).round(2), " (axis=1: across columns)")
print("mean per test    ->", tests.mean(axis=0).round(2), " (axis=0: down rows)")

# ----------------------------------------------------------------------------
section("5. pandas: load and inspect a table")
df = pd.read_csv(ROOT / "data" / "students.csv")
print("shape (rows, columns):", df.shape)
print(df.head())
print("\nColumn types:")
print(df.dtypes)

# ----------------------------------------------------------------------------
section("6. Select, filter, sort")
print(df[["study_hours", "exam_score"]].head(3))
science = df[df["program"] == "Science"]
print("\nScience students:", len(science))
hard_workers = df[(df["study_hours"] > 15) & (df["sleep_hours"] >= 7)]
print("Study > 15 h AND sleep >= 7 h:", len(hard_workers))
print("\nTop 3 exam scores:")
print(df.sort_values("exam_score", ascending=False).head(3)[["student_id", "program", "exam_score"]])

# ----------------------------------------------------------------------------
section("7. Summaries and group-by")
print(df["exam_score"].describe().round(2))
print("\nMean exam score by program:")
print(df.groupby("program")["exam_score"].mean().round(2))
print("\nCount by tutoring:")
print(df["tutoring"].value_counts())

# New column from existing ones
df["sleep_deficit"] = (8 - df["sleep_hours"]).clip(lower=0)
print("\nNew column sleep_deficit, first 3 rows:", df["sleep_deficit"].head(3).tolist())

# ----------------------------------------------------------------------------
section("8. matplotlib: two basic plots")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["study_hours"], bins=20, color="steelblue", edgecolor="white")
ax.set_title("Weekly study hours (200 students)")
ax.set_xlabel("Study hours per week")
ax.set_ylabel("Number of students")
fig.tight_layout()
fig.savefig(FIGURES / "study_hours_hist.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["study_hours"], df["exam_score"], alpha=0.6)
ax.set_title("More study, higher scores")
ax.set_xlabel("Study hours per week")
ax.set_ylabel("Exam score")
fig.tight_layout()
fig.savefig(FIGURES / "study_vs_score.png", dpi=120)
plt.close(fig)
print("Saved figures to", FIGURES.relative_to(ROOT))

# ----------------------------------------------------------------------------
section("9. Why vectorise? Loop vs NumPy on 1,000,000 numbers")
import time

rng = np.random.default_rng(0)
big = rng.normal(size=1_000_000)
start = time.perf_counter()
total = 0.0
for value in big:
    total += value
loop_seconds = time.perf_counter() - start
start = time.perf_counter()
total_np = big.sum()
numpy_seconds = time.perf_counter() - start
print(f"loop : {loop_seconds:.4f} s")
print(f"numpy: {numpy_seconds:.4f} s  (about {loop_seconds / numpy_seconds:.0f}x faster)")
