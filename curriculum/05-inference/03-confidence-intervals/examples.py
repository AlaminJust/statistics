"""05.03 Confidence intervals — construction, interpretation and coverage simulations.

Run:  python curriculum/05-inference/03-confidence-intervals/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(53)
df = pd.read_csv(ROOT / "data" / "students.csv")


def t_interval(x, level=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return x.mean() - t * se, x.mean() + t * se


def wald(k, n, level=0.95):
    p = k / n
    z = stats.norm.ppf(0.5 + level / 2)
    half = z * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(k, n, level=0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


print("Worked example 1: 95% t-interval for the mean exam score")
x = df["exam_score"]
n = len(x)
se = x.std() / math.sqrt(n)
t_crit = stats.t.ppf(0.975, n - 1)
lo, hi = t_interval(x)
print(f"  x̄ = {x.mean():.2f}, s = {x.std():.2f}, n = {n}, SE = {se:.3f}, t(0.975, {n - 1}) = {t_crit:.3f}")
print(f"  interval: {x.mean():.2f} ± {t_crit * se:.2f} = [{lo:.2f}, {hi:.2f}]")
a_sp, b_sp = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=se)
print(f"  scipy: [{a_sp:.2f}, {b_sp:.2f}]")
for level in [0.80, 0.90, 0.95, 0.99]:
    a, b = t_interval(x, level)
    print(f"  {level:.0%} interval: [{a:.2f}, {b:.2f}]  width {b - a:.2f}")

print("\nWorked example 2: what '95%' means — coverage of 10,000 intervals from samples of 15")
mu, sigma = 64, 14
covered_t = covered_z = 0
intervals = []
for i in range(10_000):
    sample = rng.normal(mu, sigma, 15)
    a, b = t_interval(sample)
    covered_t += a <= mu <= b
    z_half = 1.96 * sample.std(ddof=1) / math.sqrt(15)
    covered_z += sample.mean() - z_half <= mu <= sample.mean() + z_half
    if i < 50:
        intervals.append((a, b))
print(f"  t-intervals contain μ: {covered_t / 10_000:.4f};  intervals using 1.96 instead: {covered_z / 10_000:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
for i, (a, b) in enumerate(intervals):
    hit = a <= mu <= b
    ax.plot([a, b], [i, i], color="tab:blue" if hit else "tab:red", lw=2)
ax.axvline(mu, color="black", ls="--")
ax.set_title("50 95% intervals from 50 samples: most (blue) contain μ")
ax.set_xlabel("exam score")
ax.set_ylabel("sample number")
fig.tight_layout()
fig.savefig(FIGURES / "coverage.png", dpi=120)
plt.close(fig)

print("\nWorked example 3: intervals for a proportion — 2 successes in 20")
print(f"  Wald   [{wald(2, 20)[0]:.4f}, {wald(2, 20)[1]:.4f}]")
print(f"  Wilson [{wilson(2, 20)[0]:.4f}, {wilson(2, 20)[1]:.4f}]")
print(f"  0 successes in 20: Wald {[round(float(v), 4) for v in wald(0, 20)]}   "
      f"Wilson {[round(float(v), 4) for v in wilson(0, 20)]}")
print("  coverage when the true p = 0.1 and n = 20 (simulated 20,000 times):")
ks = rng.binomial(20, 0.1, size=20_000)
for name, f in [("Wald", wald), ("Wilson", wilson)]:
    cover = np.mean([f(k, 20)[0] <= 0.1 <= f(k, 20)[1] for k in ks])
    print(f"    {name:<6} {cover:.4f}")
tut = (df["tutoring"] == "yes").sum()
a, b = wilson(tut, len(df))
print(f"  share of students in tutoring: {tut}/{len(df)} = {tut / len(df):.3f}, Wilson 95% [{a:.3f}, {b:.3f}]")

print("\nWorked example 4: difference between two means (Welch interval) — tutoring vs not")
y_yes = df.loc[df["tutoring"] == "yes", "exam_score"]
y_no = df.loc[df["tutoring"] == "no", "exam_score"]
diff = y_yes.mean() - y_no.mean()
v1, v2 = y_yes.var() / len(y_yes), y_no.var() / len(y_no)
se_diff = math.sqrt(v1 + v2)
df_welch = (v1 + v2) ** 2 / (v1 ** 2 / (len(y_yes) - 1) + v2 ** 2 / (len(y_no) - 1))
t_w = stats.t.ppf(0.975, df_welch)
print(f"  difference {diff:.2f}, SE {se_diff:.3f}, Welch df {df_welch:.1f}, interval [{diff - t_w * se_diff:.2f}, {diff + t_w * se_diff:.2f}]")
res = stats.ttest_ind(y_yes, y_no, equal_var=False)
ci = res.confidence_interval()
print(f"  scipy ttest_ind(...).confidence_interval(): [{ci.low:.2f}, {ci.high:.2f}]")

print("\nWorked example 5: sample size for a target margin of error")
for margin in [5, 2, 1]:
    n_needed = math.ceil((1.96 * 14 / margin) ** 2)
    print(f"  margin ±{margin} points with σ ≈ 14 -> n = (1.96 · 14 / {margin})² ≈ {n_needed}")
print(f"  a poll with margin ±3 points for a proportion near 0.5 needs n = (1.96 · 0.5 / 0.03)² ≈ {math.ceil((1.96 * 0.5 / 0.03) ** 2)}")
print("\nFigure saved:", FIGURES / "coverage.png")
