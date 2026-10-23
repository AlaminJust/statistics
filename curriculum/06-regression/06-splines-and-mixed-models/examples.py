"""06.06 Splines and mixed models — flexible curves and grouped data.

Run:  python curriculum/06-regression/06-splines-and-mixed-models/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(66)

print("Worked example 1: a non-linear relationship — performance vs hours of practice")
n = 250
x = np.sort(rng.uniform(0, 10, n))
truth = lambda t: 50 + 30 * (1 - np.exp(-0.5 * t)) - 2 * np.maximum(t - 7, 0) ** 2
y = truth(x) + rng.normal(0, 4, n)
X = x.reshape(-1, 1)
models = {
    "straight line": LinearRegression(),
    "polynomial degree 3": make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "polynomial degree 15": make_pipeline(PolynomialFeatures(15), LinearRegression()),
    "cubic spline, 5 knots": make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()),
    "cubic spline, 20 knots": make_pipeline(SplineTransformer(n_knots=20, degree=3), LinearRegression()),
}
for name, model in models.items():
    folds = KFold(n_splits=10, shuffle=True, random_state=0)       # shuffle! x is sorted
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=folds, scoring="neg_mean_squared_error").mean())
    print(f"  {name:<24} 10-fold CV RMSE {cv_rmse:6.3f}")

grid = np.linspace(-0.5, 10.5, 400).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x, y, s=8, alpha=0.4, color="grey")
for name in ["straight line", "polynomial degree 15", "cubic spline, 5 knots"]:
    axes[0].plot(grid, models[name].fit(X, y).predict(grid), label=name)
axes[0].plot(grid, truth(grid.ravel()), color="black", ls="--", label="truth")
axes[0].set_ylim(30, 95)
axes[0].set_title("Splines bend locally; high-degree polynomials wiggle at the edges")
axes[0].set_xlabel("hours of practice")
axes[0].set_ylabel("performance")
axes[0].legend(fontsize=8)
basis = SplineTransformer(n_knots=5, degree=3).fit(X).transform(grid)
axes[1].plot(grid, basis)
axes[1].set_title("The spline basis: each function is local")
axes[1].set_xlabel("x")
fig.tight_layout()
fig.savefig(FIGURES / "splines.png", dpi=120)
plt.close(fig)

print("\nWorked example 2: a spline inside a statsmodels formula (B-spline with 5 degrees of freedom)")
frame = pd.DataFrame({"x": x, "y": y})
spline_fit = smf.ols("y ~ bs(x, df=5)", frame).fit()
lin_fit = smf.ols("y ~ x", frame).fit()
print(f"  linear R² {lin_fit.rsquared:.3f}, spline R² {spline_fit.rsquared:.3f}; "
      f"F-test that the curve is needed: p = {spline_fit.compare_f_test(lin_fit)[1]:.1e}")

print("\nWorked example 3: grouped data — 30 schools, students nested within schools")
n_schools, per_school = 30, rng.integers(3, 30, 30)
school_effect = rng.normal(0, 6, n_schools)
rows = []
for s in range(n_schools):
    hours = rng.gamma(4, 2.5, per_school[s])
    score = 45 + school_effect[s] + 2.0 * hours + rng.normal(0, 8, per_school[s])
    rows += [{"school": s, "hours": h, "score": sc} for h, sc in zip(hours, score)]
schools = pd.DataFrame(rows)
print(f"  {len(schools)} students; school sizes from {per_school.min()} to {per_school.max()}")

pooled = smf.ols("score ~ hours", schools).fit()
fixed = smf.ols("score ~ hours + C(school)", schools).fit()
mixed = smf.mixedlm("score ~ hours", schools, groups=schools["school"]).fit()
print(f"  pooled OLS (ignores schools): hours {pooled.params['hours']:.3f}, SE {pooled.bse['hours']:.3f}")
print(f"  school fixed effects:         hours {fixed.params['hours']:.3f}, SE {fixed.bse['hours']:.3f}")
print(f"  random-intercept mixed model: hours {mixed.params['hours']:.3f}, SE {mixed.bse['hours']:.3f}")
var_school = float(mixed.cov_re.iloc[0, 0])
var_resid = mixed.scale
print(f"  school sd {np.sqrt(var_school):.2f} (true 6), residual sd {np.sqrt(var_resid):.2f} (true 8), "
      f"ICC = {var_school / (var_school + var_resid):.3f} (true {36 / (36 + 64):.2f})")
print(f"  (the 30 school effects actually drawn have sd {school_effect.std(ddof=1):.2f}: with only 30 schools,"
      " the between-school sd is estimated imprecisely)")

print("\nWorked example 4: partial pooling — 40 small schools, estimate each school's true mean")


def one_simulation(rng):
    J = 40
    sizes = rng.integers(2, 21, J)
    true_means = rng.normal(60, 3, J)                       # schools differ a little (sd 3)
    d = pd.DataFrame([{"s": j, "y": v} for j in range(J) for v in rng.normal(true_means[j], 10, sizes[j])])
    raw = d.groupby("s")["y"].mean().to_numpy()
    fit = smf.mixedlm("y ~ 1", d, groups=d["s"]).fit()
    shrunk = np.array([fit.params["Intercept"] + float(fit.random_effects[j].iloc[0]) for j in range(J)])
    return sizes, true_means, raw, shrunk, fit


sizes, true_means, raw, shrunk, fit = one_simulation(rng)
print(f"  grand mean {fit.params['Intercept']:.2f}; school sd {np.sqrt(float(fit.cov_re.iloc[0, 0])):.2f} (true 3); "
      f"residual sd {np.sqrt(fit.scale):.2f} (true 10)")
print(f"  this run: RMSE raw means {np.sqrt(np.mean((raw - true_means) ** 2)):.3f}, "
      f"shrunk estimates {np.sqrt(np.mean((shrunk - true_means) ** 2)):.3f}")
for j in np.argsort(sizes)[:3]:
    print(f"    school {j:>2} (n = {sizes[j]:>2}): raw {raw[j]:6.2f}  shrunk {shrunk[j]:6.2f}  truth {true_means[j]:6.2f}")
errors = np.array([[np.sqrt(np.mean((r - m) ** 2)), np.sqrt(np.mean((s - m) ** 2))]
                   for _, m, r, s, _ in (one_simulation(rng) for _ in range(100))])
print(f"  average over 100 simulations: raw {errors[:, 0].mean():.3f}, shrunk {errors[:, 1].mean():.3f} "
      f"({1 - errors[:, 1].mean() / errors[:, 0].mean():.0%} smaller error)")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(sizes, raw - true_means, label="raw school mean − truth")
ax.scatter(sizes, shrunk - true_means, label="mixed-model (shrunk) estimate − truth")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Partial pooling helps most for small schools")
ax.set_xlabel("students in the school")
ax.set_ylabel("error of the school's estimated mean")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "partial_pooling.png", dpi=120)
plt.close(fig)
print("\nFigures saved in", FIGURES)
