"""08.04 Uncertainty and conformal prediction — intervals and sets with guaranteed coverage.

Run:  python curriculum/08-statistical-learning/04-uncertainty-and-conformal/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(84)
alpha = 0.1                           # target: 90% coverage

print("Worked example 1: split conformal prediction intervals for regression (diabetes data)")
X, y = load_diabetes(return_X_y=True)
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_cal, y_train, y_cal = train_test_split(X_rest, y_rest, test_size=0.4, random_state=0)
model = RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X_train, y_train)
scores = np.abs(y_cal - model.predict(X_cal))                  # nonconformity scores on held-out calibration data
n_cal = len(scores)
q = np.quantile(scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")
pred = model.predict(X_test)
covered = np.abs(y_test - pred) <= q
print(f"  train {len(y_train)}, calibration {n_cal}, test {len(y_test)}")
print(f"  conformal quantile q = {q:.1f}; every interval is prediction ± {q:.1f}")
print(f"  test coverage {covered.mean():.3f} (target {1 - alpha:.2f}); average width {2 * q:.1f}")

print("\nWorked example 2: the guarantee holds for ANY model, even a bad one")
bad = RandomForestRegressor(20, max_depth=2, random_state=0).fit(X_train[:, :2], y_train)
scores_bad = np.abs(y_cal - bad.predict(X_cal[:, :2]))
q_bad = np.quantile(scores_bad, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")
cov_bad = np.mean(np.abs(y_test - bad.predict(X_test[:, :2])) <= q_bad)
print(f"  weak model (2 features, depth 2): coverage {cov_bad:.3f}, width {2 * q_bad:.1f}  — still valid, just wider")

print("\nWorked example 3: coverage over many random splits")
covs = []
for seed in range(200):
    Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.3, random_state=seed)
    Xa, Xc, ya, yc = train_test_split(Xr, yr, test_size=0.4, random_state=seed)
    m = RandomForestRegressor(100, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(Xa, ya)
    s = np.abs(yc - m.predict(Xc))
    qq = np.quantile(s, np.ceil((len(s) + 1) * (1 - alpha)) / len(s), method="higher")
    covs.append(np.mean(np.abs(yt - m.predict(Xt)) <= qq))
print(f"  200 splits: mean coverage {np.mean(covs):.3f}, 5th–95th percentile [{np.percentile(covs, 5):.3f}, {np.percentile(covs, 95):.3f}]")

print("\nWorked example 4: conformalised quantile regression — intervals that adapt to difficulty")
n = 3000
x1 = rng.uniform(0, 10, n)
y1 = np.sin(x1) * 3 + rng.normal(0, 0.2 + 0.3 * x1, n)        # noise grows with x
Xh = x1.reshape(-1, 1)
idx = rng.permutation(n)
tr, ca, te = idx[:1500], idx[1500:2200], idx[2200:]
lo_m = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=0).fit(Xh[tr], y1[tr])
hi_m = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=0).fit(Xh[tr], y1[tr])
mean_m = GradientBoostingRegressor(random_state=0).fit(Xh[tr], y1[tr])
s_cqr = np.maximum(lo_m.predict(Xh[ca]) - y1[ca], y1[ca] - hi_m.predict(Xh[ca]))
q_cqr = np.quantile(s_cqr, np.ceil((len(ca) + 1) * (1 - alpha)) / len(ca), method="higher")
s_abs = np.abs(y1[ca] - mean_m.predict(Xh[ca]))
q_abs = np.quantile(s_abs, np.ceil((len(ca) + 1) * (1 - alpha)) / len(ca), method="higher")
lo_te, hi_te = lo_m.predict(Xh[te]) - q_cqr, hi_m.predict(Xh[te]) + q_cqr
for name, cover, width_small, width_big in [
    ("absolute-residual conformal", np.abs(y1[te] - mean_m.predict(Xh[te])) <= q_abs, 2 * q_abs, 2 * q_abs),
    ("CQR", (y1[te] >= lo_te) & (y1[te] <= hi_te), np.mean((hi_te - lo_te)[x1[te] < 2]), np.mean((hi_te - lo_te)[x1[te] > 8])),
]:
    small = x1[te] < 2
    big = x1[te] > 8
    print(f"  {name:<28} overall {np.mean(cover):.3f} | coverage x<2: {np.mean(cover[small]):.3f}, x>8: {np.mean(cover[big]):.3f}"
          f" | width x<2: {width_small:.2f}, x>8: {width_big:.2f}")

print("\nWorked example 5: conformal prediction SETS for classification (digits, 10 classes)")
Xd, yd = load_digits(return_X_y=True)
Xd_r, Xd_t, yd_r, yd_t = train_test_split(Xd, yd, test_size=0.4, random_state=0, stratify=yd)
Xd_a, Xd_c, yd_a, yd_c = train_test_split(Xd_r, yd_r, test_size=0.4, random_state=0, stratify=yd_r)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05)).fit(Xd_a, yd_a)
for a in [0.1, 0.02]:
    s_cal = 1 - clf.predict_proba(Xd_c)[np.arange(len(yd_c)), yd_c]        # 1 - probability of the true class
    qd = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - a)) / len(s_cal), method="higher")
    sets = clf.predict_proba(Xd_t) >= 1 - qd
    sizes = sets.sum(axis=1)
    cover = sets[np.arange(len(yd_t)), yd_t]
    print(f"  target {1 - a:.0%}: coverage {cover.mean():.3f}; average set size {sizes.mean():.2f}; "
          f"sets of size 1: {np.mean(sizes == 1):.1%}; size ≥ 3: {np.mean(sizes >= 3):.1%}")
print(f"  plain accuracy of the top prediction: {clf.score(Xd_t, yd_t):.3f}")

print("\nWorked example 6: distribution shift breaks the guarantee")
shift = np.abs(y1[te] - mean_m.predict(Xh[te]))
x_shift = rng.uniform(12, 18, 800)                                 # new data entirely beyond the training range (0–10)
y_shift = np.sin(x_shift) * 3 + rng.normal(0, 0.2 + 0.3 * x_shift, 800)
cov_shift = np.mean((y_shift >= lo_m.predict(x_shift.reshape(-1, 1)) - q_cqr) & (y_shift <= hi_m.predict(x_shift.reshape(-1, 1)) + q_cqr))
print(f"  CQR coverage on shifted inputs (x in [12, 18]): {cov_shift:.3f}  (was {np.mean((y1[te] >= lo_te) & (y1[te] <= hi_te)):.3f})")

order = np.argsort(x1[te])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x1[te], y1[te], s=4, alpha=0.4, color="grey")
axes[0].fill_between(x1[te][order], lo_te[order], hi_te[order], alpha=0.3, label="CQR 90% interval")
m_te = mean_m.predict(Xh[te])
axes[0].fill_between(x1[te][order], (m_te - q_abs)[order], (m_te + q_abs)[order], alpha=0.15, label="constant-width conformal")
axes[0].set_title("Conformalised quantile regression adapts to the noise")
axes[0].set_xlabel("x")
axes[0].legend(fontsize=8)
axes[1].hist(covs, bins=25)
axes[1].axvline(0.9, color="black", ls="--")
axes[1].set_title("Coverage over 200 random splits (target 0.9)")
axes[1].set_xlabel("test coverage")
fig.tight_layout()
fig.savefig(FIGURES / "conformal.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "conformal.png")
