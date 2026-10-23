"""08.01 Generalisation and the bias-variance trade-off — risk, complexity, learning curves, double descent.

Run:  python curriculum/08-statistical-learning/01-generalization/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(81)


def truth(x):
    return np.sin(2 * x) + 0.5 * x


def sample(n):
    x = rng.uniform(0, 3, n)
    return x, truth(x) + rng.normal(0, 0.4, n)


print("Worked example 1: training vs test error as model complexity grows (polynomial degree)")
x_tr, y_tr = sample(30)
x_te, y_te = sample(5000)
degrees = range(0, 16)
train_mse, test_mse = [], []
for d in degrees:
    coefs = np.polyfit(x_tr, y_tr, d)
    train_mse.append(np.mean((np.polyval(coefs, x_tr) - y_tr) ** 2))
    test_mse.append(np.mean((np.polyval(coefs, x_te) - y_te) ** 2))
for d in [0, 1, 3, 5, 9, 15]:
    print(f"  degree {d:>2}: training MSE {train_mse[d]:.4f}   test MSE {test_mse[d]:9.4f}")
best = int(np.argmin(test_mse))
print(f"  best test error at degree {best}; noise variance (the floor) = 0.4² = {0.16:.2f}")

print("\nWorked example 2: the bias-variance decomposition at x0 = 2.5, by refitting on 2,000 training sets")
x0 = 2.5
for d in [1, 3, 9]:
    preds = []
    for _ in range(2000):
        xs, ys = sample(30)
        preds.append(np.polyval(np.polyfit(xs, ys, d), x0))
    preds = np.array(preds)
    bias2 = (preds.mean() - truth(x0)) ** 2
    var = preds.var()
    print(f"  degree {d}: bias² {bias2:.4f}  variance {var:.4f}  noise 0.1600  -> expected test MSE {bias2 + var + 0.16:.4f}")

print("\nWorked example 3: k-nearest neighbours on breast cancer — small k overfits, large k underfits")
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for k in [1, 3, 7, 15, 51, 151]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_tr, y_tr2)
    print(f"  k = {k:>3}: training accuracy {model.score(X_tr, y_tr2):.3f}   test accuracy {model.score(X_te, y_te2):.3f}")

print("\nWorked example 4: learning curves — how much would more data help?")
sizes, train_scores, val_scores = learning_curve(make_pipeline(StandardScaler(), KNeighborsClassifier(7)), X, y,
                                                 train_sizes=np.linspace(0.1, 1.0, 6), cv=5, random_state=0, shuffle=True)
for n_, tr, va in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"  n = {n_:>3}: training {tr:.3f}   validation {va:.3f}   gap {tr - va:.3f}")

print("\nWorked example 5: double descent — ReLU random-feature regression with minimum-norm least squares")
d, n_train = 10, 100
beta_true = rng.normal(size=d) / np.sqrt(d)


def make_data(m):
    Xd = rng.normal(size=(m, d))
    return Xd, 2 * np.tanh(Xd @ beta_true) + rng.normal(0, 0.3, m)


X_dd_tr, y_dd_tr = make_data(n_train)
X_dd_te, y_dd_te = make_data(4000)
W = rng.normal(size=(d, 2000)) / np.sqrt(d)
features = lambda Xd, p: np.maximum(Xd @ W[:, :p], 0)        # p random ReLU features
dd = []
for p in [10, 25, 50, 80, 95, 100, 105, 120, 200, 400, 1000, 2000]:
    coef = np.linalg.pinv(features(X_dd_tr, p)) @ y_dd_tr       # least squares; minimum-norm solution when p > n
    err = np.mean((features(X_dd_te, p) @ coef - y_dd_te) ** 2)
    dd.append((p, err))
    print(f"  {p:>4} features: test MSE {err:8.4f}{'   <- interpolation threshold (p = n)' if p == n_train else ''}")
print("\nWorked example 6: a generalisation bound for a finite set of models (Hoeffding + union bound)")
for M in [1, 10, 1000, 10 ** 6]:
    n_needed = math.ceil(math.log(2 * M / 0.05) / (2 * 0.02 ** 2))
    print(f"  choosing among {M:>9,} models: n >= ln(2M/δ)/(2ε²) = {n_needed:,} test examples for ±0.02 at 95%")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].plot(list(degrees), train_mse, marker="o", label="training")
axes[0].plot(list(degrees), test_mse, marker="o", label="test")
axes[0].set_yscale("log")
axes[0].axhline(0.16, color="grey", ls="--", lw=0.8, label="noise floor")
axes[0].set_title("Training error falls; test error is U-shaped")
axes[0].set_xlabel("polynomial degree")
axes[0].set_ylabel("MSE (log scale)")
axes[0].legend()
axes[1].plot(sizes, train_scores.mean(axis=1), marker="o", label="training")
axes[1].plot(sizes, val_scores.mean(axis=1), marker="o", label="validation")
axes[1].set_title("Learning curve (kNN, k = 7)")
axes[1].set_xlabel("training examples")
axes[1].set_ylabel("accuracy")
axes[1].legend()
ps, errs = zip(*dd)
axes[2].plot(ps, errs, marker="o")
axes[2].axvline(n_train, color="grey", ls="--", lw=0.8)
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_title("Double descent: test error peaks at p = n")
axes[2].set_xlabel("number of random features p")
axes[2].set_ylabel("test MSE (log)")
fig.tight_layout()
fig.savefig(FIGURES / "generalisation.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "generalisation.png")
