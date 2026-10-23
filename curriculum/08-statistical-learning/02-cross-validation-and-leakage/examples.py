"""08.02 Cross-validation and data leakage — honest performance estimates and the ways they go wrong.

Run:  python curriculum/08-statistical-learning/02-cross-validation-and-leakage/examples.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(82)
X, y = load_breast_cancer(return_X_y=True)

print("Worked example 1: a single train/test split is noisy")
scores = []
for seed in range(200):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scores.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr).score(X_te, y_te))
print(f"  200 random 80/20 splits: test accuracy from {min(scores):.3f} to {max(scores):.3f} (sd {np.std(scores):.3f})")
cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y,
                     cv=StratifiedKFold(10, shuffle=True, random_state=0))
print(f"  10-fold CV: mean {cv.mean():.4f}, sd across folds {cv.std():.4f}")

print("\nWorked example 2: leakage through feature selection — 2,000 pure-noise features, labels are coin flips")
n, p = 100, 2000
Xn = rng.normal(size=(n, p))
yn = rng.integers(0, 2, n)
selected = SelectKBest(f_classif, k=20).fit(Xn, yn).get_support()          # WRONG: selected using ALL the labels
wrong = cross_val_score(LogisticRegression(max_iter=2000), Xn[:, selected], yn, cv=5).mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000)), Xn, yn, cv=5).mean()
print(f"  select 20 features on all data, then cross-validate: accuracy {wrong:.3f}   <- looks great, is pure leakage")
print(f"  selection inside each fold (pipeline):               accuracy {right:.3f}   <- the truth: chance level")

print("\nWorked example 3: grouped data — several rows per patient")
n_patients, per = 60, 8
patient_effect = rng.normal(0, 2, (n_patients, 5))      # each patient has a personal 'fingerprint'
rows = []
for pid in range(n_patients):
    label = int(rng.random() < 0.5)
    for _ in range(per):
        signal = 0.3 * label
        features = patient_effect[pid] + signal + rng.normal(0, 0.3, 5)
        rows.append((pid, label, *features))
grouped = pd.DataFrame(rows, columns=["patient", "label", *[f"f{i}" for i in range(5)]])
Xg, yg, groups = grouped[[f"f{i}" for i in range(5)]].to_numpy(), grouped["label"].to_numpy(), grouped["patient"].to_numpy()
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

knn = KNeighborsClassifier(3)
kf = cross_val_score(knn, Xg, yg, cv=KFold(5, shuffle=True, random_state=0)).mean()
gkf = cross_val_score(knn, Xg, yg, cv=GroupKFold(5), groups=groups).mean()
print(f"  KFold (same patient in train and test): accuracy {kf:.3f}   <- the model recognises patients, not disease")
print(f"  GroupKFold (patients kept together):    accuracy {gkf:.3f}   <- what you'd get on new patients")

print("\nWorked example 4: time series — never train on the future")
t = np.arange(600)
series = np.cumsum(rng.normal(0, 1, 600)) + 0.02 * t
lags = np.column_stack([series[i:i + 590] for i in range(10)])
target = series[10:]
model = Ridge(1.0)
shuffled = cross_val_score(model, lags, target, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_absolute_error").mean()
forward = cross_val_score(model, lags, target, cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error").mean()
print(f"  shuffled KFold MAE {-shuffled:.3f};  TimeSeriesSplit (train on past, test on future) MAE {-forward:.3f}")

print("\nWorked example 5: nested cross-validation — tuning AND evaluating honestly")
pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": [0.01, 0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=1))
search.fit(X, y)
print(f"  best inner-CV score (optimistic: it's the best of {len(search.cv_results_['params'])} tries): {search.best_score_:.4f}")
nested = cross_val_score(GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=1)), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=2))
print(f"  nested CV estimate of the whole tuning procedure: {nested.mean():.4f} (sd {nested.std():.4f})")

print("\nWorked example 6: target leakage — a feature that is a consequence of the label")
leaky = np.column_stack([X, y + rng.normal(0, 0.1, len(y))])     # e.g. 'treatment given', recorded after diagnosis
print(f"  with the leaky feature: CV accuracy "
      f"{cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), leaky, y, cv=5).mean():.4f}"
      f"  — perfect, and useless at prediction time")
