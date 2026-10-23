"""03.06 Joint distributions and covariance — worked examples with simulation checks.

Run:  python curriculum/03-probability/06-joint-distributions-covariance/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(6)

print("Worked example 1: a joint PMF (X = studied, Y = passed)")
joint = np.array([[0.30, 0.10],     # X = 0: Y = 0, Y = 1
                  [0.20, 0.40]])    # X = 1
x_vals = np.array([0, 1])
y_vals = np.array([0, 1])
p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)
print("  joint:\n", joint)
print("  marginal of X (row sums):", p_x, "  marginal of Y (column sums):", p_y)
print("  P(Y=1 | X=1) =", round(joint[1, 1] / p_x[1], 4), "  P(Y=1 | X=0) =", round(joint[0, 1] / p_x[0], 4))
print("  independent? P(X=0,Y=0) =", joint[0, 0], "vs P(X=0)P(Y=0) =", round(p_x[0] * p_y[0], 4))

e_x = (x_vals * p_x).sum()
e_y = (y_vals * p_y).sum()
e_xy = sum(x * y * joint[i, j] for i, x in enumerate(x_vals) for j, y in enumerate(y_vals))
var_x = ((x_vals - e_x) ** 2 * p_x).sum()
var_y = ((y_vals - e_y) ** 2 * p_y).sum()
cov = e_xy - e_x * e_y
corr = cov / np.sqrt(var_x * var_y)
print(f"  E[X] = {e_x:.2f}, E[Y] = {e_y:.2f}, E[XY] = {e_xy:.2f}")
print(f"  Cov(X, Y) = E[XY] - E[X]E[Y] = {e_xy} - {e_x * e_y:.2f} = {cov:.2f}")
print(f"  Var(X) = {var_x:.2f}, Var(Y) = {var_y:.2f}, Corr = {corr:.4f}")
print(f"  Var(X + Y) = {var_x:.2f} + {var_y:.2f} + 2*{cov:.2f} = {var_x + var_y + 2 * cov:.2f}")

cells = rng.choice(4, size=500_000, p=joint.ravel())
xs, ys = cells // 2, cells % 2
print(f"  simulated: Cov {np.cov(xs, ys, ddof=0)[0, 1]:.4f}, Corr {np.corrcoef(xs, ys)[0, 1]:.4f}, "
      f"Var(X+Y) {(xs + ys).var():.4f}")

print("\nWorked example 2: zero covariance does NOT mean independence")
x = rng.choice([-1, 0, 1], size=300_000)
y = x ** 2
print(f"  X uniform on (-1, 0, 1), Y = X^2:  Cov ≈ {np.cov(x, y)[0, 1]:.4f}, yet Y is a function of X")
print("  exact: E[XY] = E[X^3] = 0 and E[X] = 0, so Cov = 0")

print("\nWorked example 3: Var of a sum of independent variables — two dice")
dice = rng.integers(1, 7, size=(500_000, 2))
print(f"  Var(die) = 35/12 = {35 / 12:.4f}; Var(sum) = 2*35/12 = {70 / 12:.4f}; simulated {dice.sum(axis=1).var():.4f}")

print("\nWorked example 4: the variance of an average falls like 1/n")
sigma2 = 35 / 12
for n in [1, 4, 16, 64]:
    means = rng.integers(1, 7, size=(100_000, n)).mean(axis=1)
    print(f"  n = {n:>2}: Var(mean) theory {sigma2 / n:.4f}  simulated {means.var():.4f}   sd {np.sqrt(sigma2 / n):.4f}")

print("\nWorked example 5: diversification — two assets with sd 20% each")
for rho in [1.0, 0.5, 0.0, -0.5]:
    var_portfolio = 0.25 * 0.2 ** 2 + 0.25 * 0.2 ** 2 + 2 * 0.25 * rho * 0.2 * 0.2
    print(f"  correlation {rho:+.1f}: sd of the 50/50 portfolio = {np.sqrt(var_portfolio):.4f}")

print("\nWorked example 6: covariance matrix of (study, sleep, score) in students.csv")
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / "data" / "students.csv")
cov_matrix = df[["study_hours", "sleep_hours", "exam_score"]].cov()
print(cov_matrix.round(3))
w = np.array([1.0, 1.0, 0.0])
print(f"  Var(study + sleep) via w^T Σ w = {w @ cov_matrix.to_numpy() @ w:.3f}; "
      f"directly = {(df.study_hours + df.sleep_hours).var():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].imshow(joint, cmap="Blues", vmin=0, vmax=0.5)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{joint[i, j]:.2f}", ha="center", va="center", fontsize=14)
axes[0].set_xticks([0, 1], ["Y = 0 (fail)", "Y = 1 (pass)"])
axes[0].set_yticks([0, 1], ["X = 0", "X = 1 (studied)"])
axes[0].set_title("Joint PMF: mass on the diagonal → Cov > 0")
for n in [1, 4, 16]:
    means = rng.integers(1, 7, size=(50_000, n)).mean(axis=1)
    axes[1].hist(means, bins=np.arange(0.95, 6.1, 1 / (n * 1.0) if n < 16 else 0.0625), alpha=0.5,
                 density=True, label=f"mean of {n} dice")
axes[1].set_title("Averages concentrate: Var = σ²/n")
axes[1].set_xlabel("average roll")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "joint_and_averages.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "joint_and_averages.png")
