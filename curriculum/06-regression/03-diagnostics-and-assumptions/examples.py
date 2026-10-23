"""06.03 Diagnostics and assumptions — residual plots, heteroscedasticity, influence, multicollinearity.

Run:  python curriculum/06-regression/03-diagnostics-and-assumptions/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(63)
df = pd.read_csv(ROOT / "data" / "students.csv")
df["tutored"] = (df["tutoring"] == "yes").astype(int)
good = smf.ols("exam_score ~ study_hours + sleep_hours + tutored", data=df).fit()

print("1. The four standard diagnostic plots for the students model")
influence = good.get_influence()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].scatter(good.fittedvalues, good.resid, s=12, alpha=0.6)
axes[0, 0].axhline(0, color="black", lw=0.8)
axes[0, 0].set_title("Residuals vs fitted (look for curves, funnels)")
sm.qqplot(good.resid, line="s", ax=axes[0, 1], markersize=3)
axes[0, 1].set_title("Normal Q–Q of residuals")
std_resid = influence.resid_studentized_internal
axes[1, 0].scatter(good.fittedvalues, np.sqrt(np.abs(std_resid)), s=12, alpha=0.6)
axes[1, 0].set_title("Scale–location (flat = constant variance)")
axes[1, 1].scatter(influence.hat_matrix_diag, std_resid, s=12, alpha=0.6)
axes[1, 1].set_title("Residuals vs leverage")
axes[1, 1].set_xlabel("leverage h_ii")
for ax in (axes[0, 0], axes[1, 0]):
    ax.set_xlabel("fitted value")
fig.tight_layout()
fig.savefig(FIGURES / "diagnostic_plots.png", dpi=110)
plt.close(fig)
print(f"   residual skew {stats.skew(good.resid):.3f}; Breusch-Pagan p = {het_breuschpagan(good.resid, good.model.exog)[1]:.3f}")
print(f"   max leverage {influence.hat_matrix_diag.max():.3f} (average is p/n = {4 / len(df):.3f}); "
      f"max Cook's distance {influence.cooks_distance[0].max():.3f}")

print("\n2. Non-linearity: fit a straight line to a curved relationship")
x = rng.uniform(0, 10, 300)
y = 5 + 0.8 * x ** 2 + rng.normal(0, 5, 300)
curved = pd.DataFrame({"x": x, "y": y})
lin = smf.ols("y ~ x", curved).fit()
quad = smf.ols("y ~ x + I(x**2)", curved).fit()
print(f"   linear R² {lin.rsquared:.3f}; quadratic R² {quad.rsquared:.3f}")
bins = pd.cut(curved["x"], 5)
print("   mean residual of the linear fit by x-bin (should be ~0 everywhere):",
      lin.resid.groupby(bins, observed=True).mean().round(1).tolist())

print("\n3. Heteroscedasticity: noise that grows with x")
income = rng.uniform(20, 200, 400)
spend = 5 + 0.3 * income + rng.normal(0, 0.0015 * income ** 2)     # noise sd grows with the square of income
het = pd.DataFrame({"income": income, "spend": spend})
ols_classic = smf.ols("spend ~ income", het).fit()
ols_robust = smf.ols("spend ~ income", het).fit(cov_type="HC3")
print(f"   slope {ols_classic.params['income']:.4f}; classic SE {ols_classic.bse['income']:.4f}; "
      f"robust (HC3) SE {ols_robust.bse['income']:.4f}")
print(f"   Breusch-Pagan p = {het_breuschpagan(ols_classic.resid, ols_classic.model.exog)[1]:.2e}")
cover_classic = cover_robust = 0
for _ in range(2000):
    inc = rng.uniform(20, 200, 100)
    sp = 5 + 0.3 * inc + rng.normal(0, 0.0015 * inc ** 2)
    d = pd.DataFrame({"inc": inc, "sp": sp})
    for fit, counter in [(smf.ols("sp ~ inc", d).fit(), "c"), (smf.ols("sp ~ inc", d).fit(cov_type="HC3"), "r")]:
        lo, hi = fit.conf_int().loc["inc"]
        if lo <= 0.3 <= hi:
            if counter == "c":
                cover_classic += 1
            else:
                cover_robust += 1
print(f"   95% CI coverage over 2,000 simulations: classic {cover_classic / 2000:.3f}, robust {cover_robust / 2000:.3f}")

print("\n4. Influence: one bad point")
clean = df[["study_hours", "exam_score"]].copy()
dirty = pd.concat([clean, pd.DataFrame({"study_hours": [60.0], "exam_score": [20.0]})], ignore_index=True)
m_clean = smf.ols("exam_score ~ study_hours", clean).fit()
m_dirty = smf.ols("exam_score ~ study_hours", dirty).fit()
cooks = m_dirty.get_influence().cooks_distance[0]
print(f"   slope without the point {m_clean.params['study_hours']:.3f}; with one data-entry error (60 h, 20 points) "
      f"{m_dirty.params['study_hours']:.3f}")
print(f"   leverage of that point {m_dirty.get_influence().hat_matrix_diag[-1]:.3f}; Cook's distance {cooks[-1]:.2f} "
      f"(next largest {np.sort(cooks)[-2]:.3f})")

print("\n5. Multicollinearity: two predictors that measure the same thing")
df["study_minutes_reported"] = df["study_hours"] * 60 + rng.normal(0, 30, len(df))   # a noisy duplicate
X = sm.add_constant(df[["study_hours", "study_minutes_reported", "sleep_hours"]])
vifs = {col: round(float(variance_inflation_factor(X.values, i)), 1) for i, col in enumerate(X.columns) if col != "const"}
print("   VIF:", vifs)
both = smf.ols("exam_score ~ study_hours + study_minutes_reported + sleep_hours", df).fit()
print(f"   study_hours coefficient {both.params['study_hours']:.3f} (SE {both.bse['study_hours']:.3f}) — "
      f"without the duplicate it was {good.params['study_hours']:.3f} (SE {good.bse['study_hours']:.3f})")
print(f"   but predictions barely change: R² {both.rsquared:.4f} vs {smf.ols('exam_score ~ study_hours + sleep_hours', df).fit().rsquared:.4f}")

print("\n6. Coefficient tests and the overall F-test")
print(f"   overall F = {good.fvalue:.1f}, p = {good.f_pvalue:.1e} (H0: all slopes are 0)")
small = smf.ols("exam_score ~ study_hours", df).fit()
f_test = good.compare_f_test(small)
print(f"   nested F-test: do sleep + tutored add to study alone? F = {f_test[0]:.2f}, p = {f_test[1]:.1e}")
print("\nFigure saved:", FIGURES / "diagnostic_plots.png")
