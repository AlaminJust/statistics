"""Generate data/job_training.csv — a SYNTHETIC observational study used in the m09 project.

8,000 adults; some chose to join a job-training programme. Participation depends on age, education, prior earnings,
unemployment and region, so a naive comparison is badly confounded. Earnings are recorded before (year 0) and after
(year 2) the programme. The true effects are hidden in this script — read it only after finishing your analysis.

Run:  python data/make_job_training.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2025)
n = 8000

age = rng.integers(18, 60, n)
educ = rng.integers(8, 19, n)
female = rng.random(n) < 0.5
region = rng.choice(["north", "south", "east", "west"], n, p=[0.3, 0.3, 0.2, 0.2])
unemployed = rng.random(n) < 0.25
base = 9_000 + 900 * (educ - 12) + 300 * (age - 18) - 6 * (age - 18) ** 2 + np.where(region == "north", 2_000, 0)
earn0 = np.maximum(0, base * np.where(unemployed, 0.35, 1.0) + rng.normal(0, 4_000, n)).round(0)

logit = (-0.6 + 1.3 * unemployed - 0.00009 * earn0 - 0.025 * (age - 35) - 0.12 * (educ - 12)
         + np.where(region == "south", 0.5, 0.0))
treated = rng.random(n) < 1 / (1 + np.exp(-logit))

# Hidden truth: the effect is larger for the unemployed and shrinks with education.
effect = 1_500 + 1_500 * unemployed - 150 * (educ - 12)
growth = 1_200 + 0.05 * earn0                                         # everyone's earnings grow over two years
earn2 = np.maximum(0, earn0 + growth + effect * treated + 2_500 * unemployed * rng.random(n)
                   + rng.normal(0, 4_000, n)).round(0)

df = pd.DataFrame({
    "person_id": np.arange(1, n + 1),
    "age": age,
    "educ": educ,
    "female": female.astype(int),
    "region": region,
    "unemployed": unemployed.astype(int),
    "earnings_before": earn0.astype(int),
    "trained": treated.astype(int),
    "earnings_after": earn2.astype(int),
})
out = Path(__file__).parent / "job_training.csv"
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df):,} rows; {df.trained.mean():.1%} trained")
