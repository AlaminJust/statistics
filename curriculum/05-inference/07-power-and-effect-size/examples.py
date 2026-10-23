"""05.07 Power and effect size — planning studies and reporting magnitudes.

Run:  python curriculum/05-inference/07-power-and-effect-size/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(57)
df = pd.read_csv(ROOT / "data" / "students.csv")
z = stats.norm.ppf

print("Worked example 1: Cohen's d for tutoring")
yes = df.loc[df["tutoring"] == "yes", "exam_score"]
no = df.loc[df["tutoring"] == "no", "exam_score"]
pooled_sd = math.sqrt(((len(yes) - 1) * yes.var() + (len(no) - 1) * no.var()) / (len(yes) + len(no) - 2))
d = (yes.mean() - no.mean()) / pooled_sd
print(f"  difference {yes.mean() - no.mean():.2f} points, pooled sd {pooled_sd:.2f}, d = {d:.3f}")
print(f"  for comparison, the study-hours slope: one sd of study ({df.study_hours.std():.2f} h) moves the score by "
      f"{2.05 * df.study_hours.std():.1f} points = {2.05 * df.study_hours.std() / df.exam_score.std():.2f} sd")

print("\nWorked example 2: sample size per group for 80% power, α = 0.05, two-sided")
for effect in [0.2, 0.5, 0.8]:
    n_formula = 2 * (z(0.975) + z(0.80)) ** 2 / effect ** 2
    n_t = TTestIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.80)
    print(f"  d = {effect}: formula 2(z_α/2 + z_β)²/d² = {n_formula:.1f}  -> t-test exact {math.ceil(n_t)} per group")

print("\nWorked example 3: power by simulation for d = 0.5, n = 64 per group")
hits = sum(stats.ttest_ind(rng.normal(0.5, 1, 64), rng.normal(0, 1, 64)).pvalue < 0.05 for _ in range(10_000))
print(f"  simulated power {hits / 10_000:.4f}  (statsmodels: {TTestIndPower().power(0.5, 64, 0.05):.4f})")

print("\nWorked example 4: power curves")
ns = np.arange(5, 201)
fig, ax = plt.subplots(figsize=(6.5, 4))
for effect in [0.2, 0.5, 0.8]:
    ax.plot(ns, TTestIndPower().power(effect, ns, 0.05), label=f"d = {effect}")
ax.axhline(0.8, color="grey", ls="--", lw=0.8)
ax.set_title("Power of a two-sample t-test (α = 0.05)")
ax.set_xlabel("sample size per group")
ax.set_ylabel("power")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "power_curves.png", dpi=120)
plt.close(fig)
print("  power at n = 20 per group:", {e: round(float(TTestIndPower().power(e, 20, 0.05)), 3) for e in [0.2, 0.5, 0.8]})

print("\nWorked example 5: A/B test on conversion, 10% -> 11%")
h = proportion_effectsize(0.11, 0.10)
n_ab = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.80)
p_bar = 0.105
n_simple = (z(0.975) * math.sqrt(2 * p_bar * (1 - p_bar)) + z(0.80) * math.sqrt(0.10 * 0.90 + 0.11 * 0.89)) ** 2 / 0.01 ** 2
print(f"  Cohen's h = {h:.4f};  n per arm = {math.ceil(n_ab):,}  (classic formula {math.ceil(n_simple):,})")
print(f"  for 10% -> 10.5%: n per arm = {math.ceil(NormalIndPower().solve_power(proportion_effectsize(0.105, 0.10), alpha=0.05, power=0.8)):,}")

print("\nWorked example 6: minimum detectable effect with a fixed budget (1,000 users per arm)")
mde_h = NormalIndPower().solve_power(nobs1=1000, alpha=0.05, power=0.80)
for p_new in np.arange(0.100, 0.200, 0.001):
    if proportion_effectsize(p_new, 0.10) >= mde_h:
        print(f"  MDE: the conversion would have to rise from 10.0% to about {p_new:.1%}")
        break

print("\nWorked example 7: statistical vs practical significance")
big_a = rng.normal(100.0, 15, 500_000)
big_b = rng.normal(100.3, 15, 500_000)
res = stats.ttest_ind(big_b, big_a)
d_big = (big_b.mean() - big_a.mean()) / 15
print(f"  n = 500,000 per group, true difference 0.3 points: p = {res.pvalue:.1e}, d = {d_big:.3f}")
print("  'highly significant', yet 0.02 sd is negligible for almost any decision")

print("\nWorked example 8: low power exaggerates published effects (the 'winner's curse')")
true_d, n = 0.2, 20
est = []
for _ in range(20_000):
    a, b = rng.normal(true_d, 1, n), rng.normal(0, 1, n)
    if stats.ttest_ind(a, b).pvalue < 0.05 and a.mean() > b.mean():
        est.append(a.mean() - b.mean())
print(f"  true d = {true_d}, n = {n} per group: power {len(est) / 20_000:.3f}")
print(f"  among significant results, the average estimated effect is {np.mean(est):.3f}, "
      f"{np.mean(est) / true_d:.1f}x the truth")
print("\nFigure saved:", FIGURES / "power_curves.png")
