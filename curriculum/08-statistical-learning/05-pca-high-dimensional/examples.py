"""08.05 PCA and high-dimensional data — the curse of dimensionality and principal components from scratch.

Run:  python curriculum/08-statistical-learning/05-pca-high-dimensional/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(85)

print("Worked example 1: the curse of dimensionality — distances concentrate")
for d in [2, 10, 100, 1000]:
    pts = rng.uniform(size=(500, d))
    dists = np.linalg.norm(pts[1:] - pts[0], axis=1)
    print(f"  d = {d:>4}: nearest {dists.min():6.3f}, farthest {dists.max():6.3f}, ratio farthest/nearest {dists.max() / dists.min():6.2f}")
for d in [2, 10, 50]:
    inside = np.mean(np.linalg.norm(rng.uniform(-1, 1, size=(200_000, d)), axis=1) <= 1)
    print(f"  share of a cube [-1, 1]^{d} inside the unit ball: {inside:.6f}")

print("\nWorked example 2: PCA by hand on two correlated variables")
x = rng.normal(0, 1, 500)
data = np.column_stack([x, 0.8 * x + rng.normal(0, 0.4, 500)])
centred = data - data.mean(axis=0)
cov = np.cov(centred, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
print("  covariance matrix:\n", cov.round(3))
print(f"  eigenvalues (variance along each component): {eigvals.round(3)}  -> explained {np.round(eigvals / eigvals.sum(), 3)}")
print(f"  first component direction: {eigvecs[:, 0].round(3)}")
scores = centred @ eigvecs
print(f"  variance of the scores: {scores.var(axis=0, ddof=1).round(3)} (= eigenvalues); correlation between scores "
      f"{np.corrcoef(scores.T)[0, 1]:.1e} (uncorrelated)")
sk = PCA().fit(data)
print(f"  sklearn PCA: explained variance ratio {sk.explained_variance_ratio_.round(3)}, component 1 {sk.components_[0].round(3)}")
U, S, Vt = np.linalg.svd(centred, full_matrices=False)
print(f"  via SVD: singular values² / (n - 1) = {(S ** 2 / (len(data) - 1)).round(3)}  (same eigenvalues)")

print("\nWorked example 3: digits (64 pixels) — how many components do we need?")
Xd, yd = load_digits(return_X_y=True)
pca = PCA().fit(Xd)
cum = np.cumsum(pca.explained_variance_ratio_)
for k in [2, 5, 10, 20, 30, 40]:
    print(f"  {k:>2} components explain {cum[k - 1]:.1%} of the variance")
print(f"  components for 90%: {np.searchsorted(cum, 0.90) + 1}; for 95%: {np.searchsorted(cum, 0.95) + 1}")
for k in [5, 10, 20, 64]:
    p = PCA(k).fit(Xd)
    recon = p.inverse_transform(p.transform(Xd))
    print(f"  k = {k:>2}: reconstruction RMSE {np.sqrt(np.mean((Xd - recon) ** 2)):.3f} (pixel range 0-16)")

print("\nWorked example 4: PCA before a classifier")
for k in [2, 5, 10, 20, 40, None]:
    steps = [StandardScaler()] + ([PCA(k)] if k else []) + [KNeighborsClassifier(5)]
    acc = cross_val_score(make_pipeline(*steps), Xd, yd, cv=5).mean()
    print(f"  kNN on {'all 64 pixels' if k is None else f'{k:>2} components':<15}: CV accuracy {acc:.3f}")

print("\nWorked example 5: standardise first when variables have different units")
Xb, yb = load_breast_cancer(return_X_y=True)
raw = PCA(2).fit(Xb)
std = PCA(2).fit(StandardScaler().fit_transform(Xb))
names = load_breast_cancer().feature_names
print(f"  raw data: PC1 explains {raw.explained_variance_ratio_[0]:.1%}, dominated by '{names[np.argmax(np.abs(raw.components_[0]))]}'"
      f" (sd {Xb[:, np.argmax(np.abs(raw.components_[0]))].std():.0f})")
print(f"  standardised: PC1 explains {std.explained_variance_ratio_[0]:.1%}; top loadings "
      f"{[str(names[i]) for i in np.argsort(np.abs(std.components_[0]))[::-1][:3]]}")
print(f"  logistic regression on 2 standardised PCs: CV accuracy "
      f"{cross_val_score(make_pipeline(StandardScaler(), PCA(2), LogisticRegression(max_iter=2000)), Xb, yb, cv=5).mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(data[:, 0], data[:, 1], s=6, alpha=0.4)
for val, vec in zip(eigvals, eigvecs.T):
    axes[0].annotate("", xy=data.mean(axis=0) + 2 * np.sqrt(val) * vec, xytext=data.mean(axis=0),
                     arrowprops={"arrowstyle": "->", "lw": 2})
axes[0].set_title("Principal components = directions of most variance")
axes[0].set_aspect("equal")
axes[1].plot(np.arange(1, 65), cum, marker=".")
axes[1].axhline(0.9, color="grey", ls="--")
axes[1].set_title("Digits: cumulative explained variance")
axes[1].set_xlabel("number of components")
proj = PCA(2).fit_transform(Xd)
sc = axes[2].scatter(proj[:, 0], proj[:, 1], c=yd, s=5, cmap="tab10")
axes[2].set_title("Digits in the first two components")
plt.colorbar(sc, ax=axes[2])
fig.tight_layout()
fig.savefig(FIGURES / "pca.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "pca.png")
