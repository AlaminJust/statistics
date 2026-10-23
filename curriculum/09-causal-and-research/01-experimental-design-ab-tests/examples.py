"""09.01 Experimental design and A/B tests — planning, sanity checks, peeking, variance reduction, blocking.

Run:  python curriculum/09-causal-and-research/01-experimental-design-ab-tests/examples.py
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
rng = np.random.default_rng(91)
ab = pd.read_csv(ROOT / "data" / "ab_test.csv")
z = stats.norm.ppf


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    p_bar = (p1 + p2) / 2
    num = (z(1 - alpha / 2) * math.sqrt(2 * p_bar * (1 - p_bar)) + z(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


print("Worked example 1: planning — sample size per arm for a conversion test (α = 0.05, power 0.8)")
for p1, p2 in [(0.10, 0.12), (0.10, 0.11), (0.10, 0.105), (0.02, 0.022)]:
    print(f"  {p1:.1%} -> {p2:.1%} (relative lift {p2 / p1 - 1:+.0%}): {n_per_arm(p1, p2):>8,} users per arm")

print("\nWorked example 2: sanity checks on data/ab_test.csv")
counts = ab["variant"].value_counts().reindex(["A", "B"])
srm = stats.chisquare(counts, f_exp=[len(ab) / 2] * 2)
print(f"  sample ratio: A {counts['A']:,}, B {counts['B']:,}; chi-square p = {srm.pvalue:.3f} (no sample ratio mismatch if p is large)")
for col in ["pre_spend"]:
    a, b = ab.loc[ab.variant == "A", col], ab.loc[ab.variant == "B", col]
    smd = (b.mean() - a.mean()) / math.sqrt((a.var() + b.var()) / 2)
    print(f"  balance on {col}: A {a.mean():.2f}, B {b.mean():.2f}, standardised difference {smd:+.3f}")
print("  device share mobile:", ab.groupby("variant")["device"].apply(lambda s: round((s == 'mobile').mean(), 3)).to_dict())
broken = stats.chisquare([10_300, 9_700], f_exp=[10_000, 10_000])
print(f"  example of a broken split (10,300 vs 9,700 when 50/50 was intended): p = {broken.pvalue:.1e} -> investigate before analysing!")

print("\nWorked example 3: the peeking problem — A/A tests checked daily for 20 days")


def daily_p_values(rng):
    """One A/A test: 500 users per arm per day for 20 days, no true difference. Returns the p-value after each day."""
    a = (rng.random((20, 500)) < 0.10).sum(axis=1).cumsum()
    b = (rng.random((20, 500)) < 0.10).sum(axis=1).cumsum()
    n = 500 * np.arange(1, 21)
    pooled = (a + b) / (2 * n)
    zstat = (b - a) / n / np.sqrt(2 * pooled * (1 - pooled) / n)
    return 2 * stats.norm.sf(np.abs(zstat))


pvals = np.array([daily_p_values(rng) for _ in range(4000)])
pocock = 2 * stats.norm.sf(2.67)          # Pocock constant for 20 equally spaced looks at overall α = 0.05
print(f"  stop at the first daily p < 0.05: false-positive rate {(pvals < 0.05).any(axis=1).mean():.3f}")
print(f"  look once, at day 20:             false-positive rate {(pvals[:, -1] < 0.05).mean():.3f}")
print(f"  daily looks with the Pocock threshold p < {pocock:.4f}: false-positive rate {(pvals < pocock).any(axis=1).mean():.3f}")

print("\nWorked example 4: CUPED — use pre-experiment data to reduce variance (revenue per user)")
y = ab["revenue"].to_numpy()
x = ab["pre_spend"].to_numpy()
treat = (ab["variant"] == "B").to_numpy()
theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
y_cuped = y - theta * (x - x.mean())
for label, outcome in [("raw revenue", y), ("CUPED-adjusted", y_cuped)]:
    diff = outcome[treat].mean() - outcome[~treat].mean()
    se = math.sqrt(outcome[treat].var(ddof=1) / treat.sum() + outcome[~treat].var(ddof=1) / (~treat).sum())
    print(f"  {label:<15}: difference {diff:.3f}, SE {se:.4f}, 95% CI [{diff - 1.96 * se:.3f}, {diff + 1.96 * se:.3f}], "
          f"p = {2 * stats.norm.sf(abs(diff / se)):.4f}")
r = np.corrcoef(x, y)[0, 1]
print(f"  correlation of pre_spend with revenue r = {r:.3f}: variance reduced by about r² = {r ** 2:.1%}"
      f" -> equivalent to {1 / (1 - r ** 2) - 1:.1%} more users for free")

print("  a metric strongly tied to pre-period behaviour (simulated weekly sessions, r ≈ 0.7):")
pre_sessions = rng.gamma(3, 2, 20_000)
treat_sim = rng.random(20_000) < 0.5
sessions = 0.9 * pre_sessions + rng.normal(0, 3, 20_000) + 0.15 * treat_sim
theta_s = np.cov(pre_sessions, sessions)[0, 1] / pre_sessions.var(ddof=1)
adj = sessions - theta_s * (pre_sessions - pre_sessions.mean())
for label, outcome in [("raw sessions", sessions), ("CUPED-adjusted", adj)]:
    diff = outcome[treat_sim].mean() - outcome[~treat_sim].mean()
    se = math.sqrt(outcome[treat_sim].var(ddof=1) / treat_sim.sum() + outcome[~treat_sim].var(ddof=1) / (~treat_sim).sum())
    print(f"    {label:<15}: difference {diff:.3f} (true 0.15), SE {se:.4f}, p = {2 * stats.norm.sf(abs(diff / se)):.4f}")
print(f"    r = {np.corrcoef(pre_sessions, sessions)[0, 1]:.3f}: the CUPED SE is about sqrt(1 - r²) of the raw SE")

print("\nWorked example 5: blocking — randomise within strata of a strong predictor")
pop_pre = rng.gamma(2, 20, 4000)
true_effect = 1.0
complete, blocked = [], []
for _ in range(2000):
    idx = rng.choice(4000, 200, replace=False)
    pre = pop_pre[idx]
    outcome0 = 10 + 0.5 * pre + rng.normal(0, 5, 200)
    t_c = rng.permutation(np.r_[np.ones(100), np.zeros(100)]).astype(bool)          # complete randomisation
    order = np.argsort(pre)                                                          # pairs of similar users
    t_b = np.zeros(200, bool)
    for pair in order.reshape(-1, 2):
        t_b[pair[rng.integers(0, 2)]] = True                                         # coin flip within each pair
    for t, store in [(t_c, complete), (t_b, blocked)]:
        yy = outcome0 + true_effect * t
        store.append(yy[t].mean() - yy[~t].mean())
print(f"  complete randomisation: estimate mean {np.mean(complete):.3f}, sd {np.std(complete):.3f}")
print(f"  matched-pair blocking:  estimate mean {np.mean(blocked):.3f}, sd {np.std(blocked):.3f}  (same truth, far less noise)")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(complete, bins=50, alpha=0.5, label="complete randomisation")
ax.hist(blocked, bins=50, alpha=0.5, label="matched-pair blocking")
ax.axvline(true_effect, color="black", ls="--", label="true effect")
ax.set_title("Blocking on a strong predictor shrinks the noise")
ax.set_xlabel("estimated treatment effect")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "blocking.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "blocking.png")
