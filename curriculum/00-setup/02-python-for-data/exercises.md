# 00.02 Exercises — Python for data

Start with `make lesson L=00-setup/02-python-for-data`. Write code in
`workspace/00-setup/02-python-for-data/exercises.py`. For each exercise, write your **expected** answer in a comment
before running.

Load the data with:

```python
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / "data" / "students.csv")
```

## ★ Warm-up

1. **(P)** Create `x = np.array([4, 8, 15, 16, 23, 42])`. Compute `x * 10`, `x - x.mean()`, and `x[x > 10]`.
2. **(P)** Using a mask, compute the proportion of values of `x` that are even. (Hint: `x % 2 == 0`.)
3. **(P)** Print the number of rows and columns of `df`, and the names of all columns.

## ★★ Core

4. **(P)** How many students are in their 4th year *and* study more than 10 hours a week?
5. **(P)** What proportion of students attend tutoring? Compute it two ways: with `value_counts` and with a boolean mask.
6. **(P)** Compute the mean exam score for students who attend tutoring and for those who don't. Which group scores
   higher, and by how many points? (Don't conclude that tutoring *causes* this yet — Module 09 explains why.)
7. **(P)** Mean `sleep_hours` for each `year`, sorted from highest to lowest.
8. **(M)** By hand: for `tests = [[70, 80, 90], [60, 65, 70], [85, 95, 75]]`, compute the mean of each test (column)
   and of each student (row). Then check with NumPy using the correct `axis`.
9. **(P)** Make a histogram of `commute_minutes` with a title and axis labels, save it to
   `workspace/00-setup/02-python-for-data/commute_hist.png`, and describe its shape in one sentence.
10. **(P)** Make a scatter plot of `sleep_hours` (x) vs `exam_score` (y). Is the relationship stronger or weaker than
    for `study_hours`? Answer by eye.

## ★★★ Challenge

11. **(P)** Without pandas, using only NumPy: load `exam_score` with
    `np.loadtxt(path, delimiter=",", skiprows=1, usecols=7)` and compute its mean. Confirm it matches pandas.
12. **(P)** Build a table with one row per `program` and columns: number of students, mean study hours, mean exam
    score, share attending tutoring. (Hint: `df.assign(tut=df.tutoring == "yes").groupby("program").agg(...)`.)
13. **(P)** Implement `statkit/basics.py` (`count`, `total`, `minimum`, `maximum`) using only `for` loops — no `len`,
    `sum`, `min`, `max`. Run `make test M=basics` until all tests pass. Commit with `statkit(00.02): ...`.

Solutions: [solutions/solutions.py](solutions/solutions.py) — run it after committing your attempt.
