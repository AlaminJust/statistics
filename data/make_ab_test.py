"""Generate data/ab_test.csv — a SYNTHETIC A/B test used in the m05 project and in Module 09.

20,000 users were randomly split between the current checkout page (A) and a new one (B). The truth, hidden in this
script: B raises conversion from 10.0% to 10.8% and does not change spend per buyer. Users who spent more before the
experiment also spend more during it (useful for variance reduction in Module 09).

Run:  python data/make_ab_test.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
n = 20_000

variant = rng.choice(["A", "B"], size=n)
device = rng.choice(["mobile", "desktop"], size=n, p=[0.62, 0.38])
pre_spend = np.round(rng.gamma(shape=2.0, scale=20.0, size=n), 2)            # spend in the 30 days before

base = np.where(device == "desktop", 0.115, 0.091)                            # desktop converts better
lift = np.where(variant == "B", 0.008, 0.0)
p_convert = np.clip(base + lift + 0.0004 * (pre_spend - pre_spend.mean()), 0.001, 0.999)
converted = rng.random(n) < p_convert

spend_if_buy = np.exp(rng.normal(3.6, 0.6, size=n)) + 0.3 * pre_spend         # right-skewed order values
revenue = np.round(np.where(converted, spend_if_buy, 0.0), 2)

df = pd.DataFrame({
    "user_id": np.arange(1, n + 1),
    "variant": variant,
    "device": device,
    "pre_spend": pre_spend,
    "converted": converted.astype(int),
    "revenue": revenue,
})
out = Path(__file__).parent / "ab_test.csv"
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df):,} rows")
