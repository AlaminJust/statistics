"""06.01 Simple linear regression — fit, interpret, evaluate.

Run:  python curriculum/06-regression/01-simple-linear-regression/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(61)
df = pd.read_csv(ROOT / "data" / "students.csv")
x = df["study_hours"].to_numpy()
y = df["exam_score"].to_numpy(dtype=float)

print("Worked example 1: a line through five points, by hand")
xs = np.array([2, 4, 6, 8, 10.])
ys = np.array([50, 60, 65, 80, 85.])
sxy = ((xs - xs.mean()) * (ys - ys.mean())).sum()
sxx = ((xs - xs.mean()) ** 2).sum()
b1 = sxy / sxx
b0 = ys.mean() - b1 * xs.mean()
print(f"  Sxy = {sxy}, Sxx = {sxx}, slope b1 = Sxy/Sxx = {b1}, intercept b0 = ȳ - b1 x̄ = {b0}")
fitted = b0 + b1 * xs
resid = ys - fitted
print("  fitted:", fitted, " residuals:", resid, " sum of residuals:", round(resid.sum(), 10))
sse = (resid ** 2).sum()
sst = ((ys - ys.mean()) ** 2).sum()
print(f"  SSE = {sse}, SST = {sst}, R² = 1 - SSE/SST = {1 - sse / sst:.4f}, r² = {np.corrcoef(xs, ys)[0, 1] ** 2:.4f}")

print("\nWorked example 2: exam score on study hours (students.csv)")
r = np.corrcoef(x, y)[0, 1]
slope = r * y.std(ddof=1) / x.std(ddof=1)
intercept = y.mean() - slope * x.mean()
print(f"  slope = r · s_y/s_x = {r:.4f} · {y.std(ddof=1):.3f}/{x.std(ddof=1):.3f} = {slope:.4f}")
print(f"  intercept = {intercept:.3f}")
model = smf.ols("exam_score ~ study_hours", data=df).fit()
print(model.summary().tables[1])
resid_sd = np.sqrt(model.ssr / (len(df) - 2))
print(f"  R² = {model.rsquared:.4f}; residual standard error = {resid_sd:.3f} points")
print(f"  95% CI for the slope: [{model.conf_int().loc['study_hours', 0]:.3f}, {model.conf_int().loc['study_hours', 1]:.3f}]"
      "  (the data were generated with 2.2)")

print("\nWorked example 3: prediction vs confidence intervals at 15 study hours")
pred = model.get_prediction(pd.DataFrame({"study_hours": [15]})).summary_frame(alpha=0.05)
print(f"  predicted mean score {pred['mean'].iloc[0]:.2f}")
print(f"  95% CI for the MEAN score of all 15-hour students: [{pred['mean_ci_lower'].iloc[0]:.2f}, {pred['mean_ci_upper'].iloc[0]:.2f}]")
print(f"  95% PREDICTION interval for ONE such student:     [{pred['obs_ci_lower'].iloc[0]:.2f}, {pred['obs_ci_upper'].iloc[0]:.2f}]")
print(f"  extrapolation to 60 hours a week: {model.predict(pd.DataFrame({'study_hours': [60]})).iloc[0]:.1f} points  <- impossible (max 100)")

print("\nWorked example 4: regression in the other direction is NOT the inverse line")
rev = smf.ols("study_hours ~ exam_score", data=df).fit()
print(f"  score on hours: slope {model.params['study_hours']:.4f}  -> implied hours-per-point {1 / model.params['study_hours']:.4f}")
print(f"  hours on score: slope {rev.params['exam_score']:.4f}  (= r · s_x/s_y = {r * x.std(ddof=1) / y.std(ddof=1):.4f})")

print("\nWorked example 5: regression to the mean in test-retest data")
ability = rng.normal(60, 10, 5000)
test1 = ability + rng.normal(0, 8, 5000)
test2 = ability + rng.normal(0, 8, 5000)
top = test1 > np.percentile(test1, 90)
print(f"  top 10% on test 1 averaged {test1[top].mean():.1f}; the same students averaged {test2[top].mean():.1f} on test 2")
bottom = test1 < np.percentile(test1, 10)
print(f"  bottom 10% on test 1 averaged {test1[bottom].mean():.1f}; on test 2 {test2[bottom].mean():.1f}  (no intervention!)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
grid = np.linspace(0, 28, 100)
band = model.get_prediction(pd.DataFrame({"study_hours": grid})).summary_frame(alpha=0.05)
axes[0].scatter(x, y, alpha=0.5, s=15)
axes[0].plot(grid, band["mean"], color="black", label="fitted line")
axes[0].fill_between(grid, band["mean_ci_lower"], band["mean_ci_upper"], alpha=0.3, label="95% CI for the mean")
axes[0].fill_between(grid, band["obs_ci_lower"], band["obs_ci_upper"], alpha=0.12, label="95% prediction interval")
axes[0].set_title(f"Score ≈ {intercept:.1f} + {slope:.2f} × hours  (R² = {model.rsquared:.2f})")
axes[0].set_xlabel("study hours per week")
axes[0].set_ylabel("exam score")
axes[0].legend(fontsize=8)
axes[1].scatter(model.fittedvalues, model.resid, alpha=0.5, s=15)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Residuals vs fitted: no pattern is good")
axes[1].set_xlabel("fitted value")
axes[1].set_ylabel("residual")
fig.tight_layout()
fig.savefig(FIGURES / "simple_regression.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "simple_regression.png")
