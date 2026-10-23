"""04.02 Continuous distributions — PDFs, CDFs, quantiles and inverse-CDF sampling.

Run:  python curriculum/04-distributions/02-continuous-distributions/examples.py
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
rng = np.random.default_rng(42)

print("Worked example 1: uniform waiting time, bus every 10 minutes")
u = stats.uniform(loc=0, scale=10)
print(f"  P(wait > 7) = (10 - 7)/10 = {u.sf(7):.2f}")
print(f"  mean (a+b)/2 = {u.mean():.1f}, variance (b-a)^2/12 = {u.var():.3f}, sd {u.std():.3f}")
print(f"  density is 1/10 = {u.pdf(3):.2f} everywhere on [0, 10]")

print("\nWorked example 2: exponential commute times with mean 25 minutes (rate 1/25)")
lam = 1 / 25
e = stats.expon(scale=25)
print(f"  P(X > 60) = e^(-60/25) = {math.exp(-60 * lam):.4f}   scipy {e.sf(60):.4f}")
print(f"  P(X <= 10) = 1 - e^(-10/25) = {e.cdf(10):.4f}")
print(f"  median = 25 ln 2 = {25 * math.log(2):.2f}   scipy ppf(0.5) = {e.ppf(0.5):.2f}")
print(f"  90th percentile = -25 ln(0.1) = {e.ppf(0.9):.2f}")
df = pd.read_csv(ROOT / "data" / "students.csv")
c = df["commute_minutes"]
print(f"  students.csv (generated as exponential(25)): mean {c.mean():.2f}, median {c.median():.1f}, "
      f"share > 60: {(c > 60).mean():.3f}")

print("\nWorked example 3: memorylessness")
print(f"  P(X > 40 | X > 20) = P(X > 40)/P(X > 20) = {e.sf(40) / e.sf(20):.4f};  P(X > 20) = {e.sf(20):.4f}")
x = rng.exponential(25, size=1_000_000)
print(f"  simulated: P(X > 40 | X > 20) = {(x[x > 20] > 40).mean():.4f}")

print("\nWorked example 4: inverse-CDF sampling")
U = rng.uniform(size=200_000)
by_hand = -np.log(1 - U) / lam
print(f"  exponential via -ln(1-U)/λ: mean {by_hand.mean():.2f} (expect 25), median {np.median(by_hand):.2f} (expect 17.33)")
sqrt_u = np.sqrt(U)
print(f"  sqrt(U) has density 2x on [0, 1] (Lesson 01.04): mean {sqrt_u.mean():.4f} (expect 0.6667), "
      f"P(<= 0.5) {(sqrt_u <= 0.5).mean():.4f} (expect 0.25)")

print("\nWorked example 5: gamma and beta in one line each")
g = stats.gamma(a=4, scale=2.5)
print(f"  Gamma(shape 4, scale 2.5) — how study_hours was generated: mean {g.mean():.1f}, sd {g.std():.2f}")
print(f"  students.csv study_hours: mean {df['study_hours'].mean():.2f}, sd {df['study_hours'].std():.2f}")
be = stats.beta(8, 4)
print(f"  Beta(8, 4) — a belief about a probability: mean {be.mean():.3f}, 95% of mass in "
      f"[{be.ppf(0.025):.3f}, {be.ppf(0.975):.3f}]")

print("\nPDF vs CDF vs quantile (ppf) for the standard normal")
z = stats.norm()
print(f"  pdf(0) = {z.pdf(0):.4f} (a density, not a probability)")
print(f"  cdf(1.96) = {z.cdf(1.96):.4f};  ppf(0.975) = {z.ppf(0.975):.4f}")

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
grid = np.linspace(0, 100, 400)
dists = [("Uniform(0, 10)", u, np.linspace(-1, 11, 400)), ("Exponential(mean 25)", e, grid),
         ("Gamma(4, 2.5)", g, np.linspace(0, 35, 400))]
for col, (name, dist, xs) in enumerate(dists):
    axes[0, col].plot(xs, dist.pdf(xs))
    axes[0, col].set_title(f"{name}: PDF")
    axes[1, col].plot(xs, dist.cdf(xs), color="tab:orange")
    axes[1, col].set_title("CDF")
    axes[1, col].set_xlabel("x")
axes[0, 1].hist(c, bins=25, density=True, alpha=0.4, label="students.csv commute")
axes[0, 1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "continuous_pdf_cdf.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "continuous_pdf_cdf.png")
