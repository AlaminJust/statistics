"""Generate data/students.csv — a SYNTHETIC practice dataset used throughout Modules 00–02.

The numbers are simulated (fixed seed), not real students. That is deliberate: because we know exactly how the data
was made, we can check whether our statistics recover the truth. For example, exam_score was built to rise by about
2.2 points per weekly study hour; Module 06 asks you to estimate that number back from the data.

Run:  python data/make_students.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 200

program = rng.choice(["Science", "Arts", "Commerce"], size=n, p=[0.40, 0.35, 0.25])
year = rng.integers(1, 5, size=n)                                  # 1st to 4th year
study_hours = rng.gamma(shape=4, scale=2.5, size=n).round(1)       # weekly; right-skewed, mean about 10
sleep_hours = rng.normal(7, 1, size=n).clip(4, 10).round(1)        # nightly
commute_minutes = rng.exponential(25, size=n).round(0)             # strongly right-skewed
tutoring = rng.random(n) < 0.30                                    # about 30% attend tutoring

noise = rng.normal(0, 8, size=n)
exam_score = 40 + 2.2 * study_hours + 2.0 * (sleep_hours - 7) + 5.0 * tutoring + noise
exam_score = exam_score.clip(0, 100).round(0).astype(int)

df = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "program": program,
    "year": year,
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "commute_minutes": commute_minutes.astype(int),
    "tutoring": np.where(tutoring, "yes", "no"),
    "exam_score": exam_score,
})

out = Path(__file__).parent / "students.csv"
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df)} rows")
