"""06.05 Regularisation — ridge, lasso and elastic net on the diabetes data and on p > n data.

Run:  python curriculum/06-regression/05-regularization/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(65)
X, y = load_diabetes(return_X_y=True, scaled=False)
names = load_diabetes().feature_names
Xs = StandardScaler().fit_transform(X)

print("Worked example 1: the ridge closed form on standardised data")
yc = y - y.mean()
for lam in [0, 10, 100, 1000]:
    beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
    sk = Ridge(alpha=lam, fit_intercept=True).fit(Xs, y).coef_ if lam > 0 else LinearRegression().fit(Xs, y).coef_
    print(f"  λ = {lam:>4}: ‖β‖ = {np.linalg.norm(beta):7.2f}; s1 coefficient {beta[4]:8.2f} (sklearn {sk[4]:8.2f})")

print("\nWorked example 2: coefficient paths")
alphas = np.logspace(-2, 4, 60)
ridge_path = np.array([Ridge(alpha=a).fit(Xs, y).coef_ for a in alphas])
lasso_alphas = np.logspace(-2, 1.6, 60)
lasso_path = np.array([Lasso(alpha=a, max_iter=50_000).fit(Xs, y).coef_ for a in lasso_alphas])
for a in [0.1, 1, 5, 20]:
    coef = Lasso(alpha=a, max_iter=50_000).fit(Xs, y).coef_
    kept = [n for n, c in zip(names, coef) if abs(c) > 1e-8]
    print(f"  lasso α = {a:>4}: {len(kept)} non-zero coefficients: {kept}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for j, name in enumerate(names):
    axes[0].plot(alphas, ridge_path[:, j], label=name)
    axes[1].plot(lasso_alphas, lasso_path[:, j], label=name)
axes[0].set_xscale("log")
axes[0].set_title("Ridge: coefficients shrink smoothly towards 0")
axes[0].set_xlabel("penalty λ (log scale)")
axes[1].set_xscale("log")
axes[1].set_title("Lasso: coefficients hit exactly 0 one by one")
axes[1].set_xlabel("penalty α (log scale)")
for ax in axes:
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("coefficient (standardised x)")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "coefficient_paths.png", dpi=120)
plt.close(fig)

print("\nWorked example 3: choosing λ by cross-validation (pipelines so scaling is learned inside each fold)")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "ridge (CV)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 100))),
    "lasso (CV)": make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=0, max_iter=50_000)),
    "elastic net (CV)": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=10, random_state=0, max_iter=50_000)),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    rmse = mean_squared_error(y_te, model.predict(X_te)) ** 0.5
    est = model[-1]
    extra = f", chosen penalty {est.alpha_:.3f}" if hasattr(est, "alpha_") else ""
    nonzero = f", {np.sum(np.abs(est.coef_) > 1e-8)} non-zero" if name.startswith(("lasso", "elastic")) else ""
    print(f"  {name:<17} test RMSE {rmse:6.2f}{extra}{nonzero}")

print("\nWorked example 4: more features than observations (p > n)")
n, p = 60, 200
Xw = rng.normal(size=(n, p))
true_beta = np.zeros(p)
true_beta[:5] = [3, -2, 1.5, 1, -1]
yw = Xw @ true_beta + rng.normal(0, 1, n)
Xw_test = rng.normal(size=(1000, p))
yw_test = Xw_test @ true_beta + rng.normal(0, 1, 1000)
ols_min_norm = np.linalg.lstsq(Xw, yw, rcond=None)[0]
print(f"  OLS (minimum-norm solution): training RMSE {np.sqrt(np.mean((Xw @ ols_min_norm - yw) ** 2)):.3f}, "
      f"test RMSE {np.sqrt(np.mean((Xw_test @ ols_min_norm - yw_test) ** 2)):.3f}")
for name, model in [("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50))), ("lasso", LassoCV(cv=5, random_state=0, max_iter=50_000))]:
    model.fit(Xw, yw)
    test_rmse = np.sqrt(np.mean((model.predict(Xw_test) - yw_test) ** 2))
    found = np.flatnonzero(np.abs(model.coef_) > 1e-6)
    print(f"  {name:<6} test RMSE {test_rmse:.3f}" +
          (f"; selected {len(found)} features, including true ones {sorted(set(found) & set(range(5)))}" if name == "lasso" else ""))

print("\nWorked example 5: why standardise first")
raw = Ridge(alpha=100).fit(X, y).coef_
std = Ridge(alpha=100).fit(Xs, y).coef_ / X.std(axis=0)
print("  ridge on raw units penalises features unevenly (sex is coded 1/2, s1 is in mg/dL):")
for name, a, b in zip(names[:4], raw[:4], std[:4]):
    print(f"    {name:<4} raw-fit coef {a:9.3f}   standardised-fit coef (back in raw units) {b:9.3f}")
print("\nFigure saved:", FIGURES / "coefficient_paths.png")
