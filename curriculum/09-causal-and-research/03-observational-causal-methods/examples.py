"""09.03 Observational causal methods — adjustment, propensity scores, IPW, matching, doubly robust, DiD, IV, RD.

Every dataset is simulated with a KNOWN true effect, so you can see which methods recover it.
Run:  python curriculum/09-causal-and-research/03-observational-causal-methods/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(93)

print("A simulated job-training programme (in the spirit of LaLonde, 1986). True effect: +$1,800 for everyone.")
n = 5000
age = rng.uniform(18, 55, n)
educ = rng.integers(8, 17, n)
prior = np.maximum(0, rng.normal(12_000, 6_000, n) + 400 * (educ - 12))      # earnings before the programme
logit = 2.0 - 0.00025 * prior - 0.03 * (age - 30) - 0.15 * (educ - 12)       # poorer, younger, less educated -> more likely to join
treated = rng.random(n) < 1 / (1 + np.exp(-logit))
earnings = 3_000 + 0.8 * prior + 150 * (educ - 12) + 40 * (age - 30) + 1_800 * treated + rng.normal(0, 3_000, n)
df = pd.DataFrame({"age": age, "educ": educ, "prior": prior, "t": treated.astype(int), "y": earnings})
X = df[["age", "educ", "prior"]].to_numpy()
t = df["t"].to_numpy()
y = df["y"].to_numpy()
print(f"  {t.sum()} treated, {len(t) - t.sum()} controls")


def smd(x, t, w=None):
    w = np.ones_like(x) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1])
    m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1])
    v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)


print("\n1. Naive comparison")
print(f"  mean earnings treated - control = {y[t == 1].mean() - y[t == 0].mean():+,.0f}   <- negative! participants were poorer to begin with")

print("\n2. Regression adjustment")
reg = smf.ols("y ~ t + age + educ + prior", df).fit()
print(f"  coefficient on t = {reg.params['t']:+,.0f} (95% CI {reg.conf_int().loc['t', 0]:+,.0f} to {reg.conf_int().loc['t', 1]:+,.0f})")

print("\n3. Propensity scores e(x) = P(T = 1 | x)")
ps_model = LogisticRegression(max_iter=5000).fit((X - X.mean(0)) / X.std(0), t)
e = ps_model.predict_proba((X - X.mean(0)) / X.std(0))[:, 1]
print(f"  propensity range: treated {e[t == 1].min():.3f}–{e[t == 1].max():.3f}, controls {e[t == 0].min():.3f}–{e[t == 0].max():.3f}")

print("\n4. Inverse probability weighting (ATE)")
w = np.where(t == 1, 1 / e, 1 / (1 - e))
ipw = np.average(y[t == 1], weights=w[t == 1]) - np.average(y[t == 0], weights=w[t == 0])
print(f"  IPW ATE (normalised weights) = {ipw:+,.0f}; largest weight {w.max():.1f}")
print("  covariate balance, standardised mean differences (|SMD| < 0.1 is the usual target):")
for j, name in enumerate(["age", "educ", "prior"]):
    print(f"    {name:<6} before weighting {smd(X[:, j], t):+.3f}   after {smd(X[:, j], t, w):+.3f}")

e_trim = np.clip(e, 0.05, 0.95)
w_trim = np.where(t == 1, 1 / e_trim, 1 / (1 - e_trim))
ipw_trim = np.average(y[t == 1], weights=w_trim[t == 1]) - np.average(y[t == 0], weights=w_trim[t == 0])
print(f"  with propensities clipped to [0.05, 0.95]: IPW ATE = {ipw_trim:+,.0f}; largest weight {w_trim.max():.1f}")

print("\n5. Propensity-score matching (ATT: each treated person matched to the control with the closest score)")
nn = NearestNeighbors(n_neighbors=1).fit(e[t == 0].reshape(-1, 1))
dist, idx = nn.kneighbors(e[t == 1].reshape(-1, 1))
att = np.mean(y[t == 1] - y[t == 0][idx.ravel()])
print(f"  matched ATT = {att:+,.0f}; median distance between matched scores {np.median(dist):.4f}")

print("\n6. Doubly robust (augmented IPW)")
mu1 = LinearRegression().fit(X[t == 1], y[t == 1]).predict(X)
mu0 = LinearRegression().fit(X[t == 0], y[t == 0]).predict(X)
aipw = np.mean(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e))
print(f"  AIPW ATE = {aipw:+,.0f}")
bad_ps = np.full(n, t.mean())                      # a useless propensity model
aipw_bad_ps = np.mean(mu1 - mu0 + t * (y - mu1) / bad_ps - (1 - t) * (y - mu0) / (1 - bad_ps))
mu1_bad = np.full(n, y[t == 1].mean())             # a useless outcome model
mu0_bad = np.full(n, y[t == 0].mean())
aipw_bad_out = np.mean(mu1_bad - mu0_bad + t * (y - mu1_bad) / e - (1 - t) * (y - mu0_bad) / (1 - e))
print(f"  with a wrong propensity model but a right outcome model: {aipw_bad_ps:+,.0f}")
print(f"  with a right propensity model but a wrong outcome model: {aipw_bad_out:+,.0f}   <- 'doubly robust'")


print("\n   How noisy is each estimator? 200 bootstrap resamples of the 5,000 people:")


def estimators(idx):
    Xb, tb, yb = X[idx], t[idx], y[idx]
    Zb = (Xb - Xb.mean(0)) / Xb.std(0)
    eb = LogisticRegression(max_iter=5000).fit(Zb, tb).predict_proba(Zb)[:, 1]
    wb = np.where(tb == 1, 1 / eb, 1 / (1 - eb))
    ipw_b = np.average(yb[tb == 1], weights=wb[tb == 1]) - np.average(yb[tb == 0], weights=wb[tb == 0])
    m1 = LinearRegression().fit(Xb[tb == 1], yb[tb == 1]).predict(Xb)
    m0 = LinearRegression().fit(Xb[tb == 0], yb[tb == 0]).predict(Xb)
    aipw_b = np.mean(m1 - m0 + tb * (yb - m1) / eb - (1 - tb) * (yb - m0) / (1 - eb))
    reg_b = LinearRegression().fit(np.column_stack([tb, Xb]), yb).coef_[0]
    return reg_b, ipw_b, aipw_b


boots = np.array([estimators(rng.integers(0, n, n)) for _ in range(200)])
for name, col in [("regression", 0), ("IPW", 1), ("AIPW", 2)]:
    print(f"     {name:<10} bootstrap mean {boots[:, col].mean():+8,.0f}   SE {boots[:, col].std(ddof=1):6,.0f}")

print("\n7. Difference-in-differences: a minimum-wage-style policy in 20 of 50 regions")
regions = np.arange(50)
policy = regions < 20
region_level = rng.normal(100, 15, 50) + 12 * policy             # policy regions start higher (no problem for DiD)
rows = []
for period in [0, 1]:
    common_trend = 5 * period                                      # every region grows by 5
    effect = 4 * period * policy                                   # true policy effect: +4
    outcome = region_level + common_trend + effect + rng.normal(0, 2, 50)
    rows += [{"region": r, "post": period, "policy": int(p), "y": o} for r, p, o in zip(regions, policy, outcome)]
panel = pd.DataFrame(rows)
means = panel.groupby(["policy", "post"])["y"].mean().unstack()
did = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(means.round(2))
print(f"  naive post-period comparison: {means.loc[1, 1] - means.loc[0, 1]:+.2f}  (includes the pre-existing gap)")
print(f"  DiD = (change in policy regions) - (change in others) = {did:+.2f}  (true +4)")
did_reg = smf.ols("y ~ policy * post", panel).fit(cov_type="cluster", cov_kwds={"groups": panel["region"]})
print(f"  regression: coefficient on policy:post = {did_reg.params['policy:post']:+.2f} (SE {did_reg.bse['policy:post']:.2f})")

print("\n8. Instrumental variables: the return to education with unobserved ability")
m = 20_000
ability = rng.normal(size=m)
distance = rng.uniform(0, 50, m)                                   # km to the nearest college: the instrument
education = 14 - 0.05 * distance + 1.0 * ability + rng.normal(0, 1, m)
log_wage = 1.0 + 0.08 * education + 0.20 * ability + rng.normal(0, 0.3, m)   # true return 8% per year
iv = pd.DataFrame({"educ": education, "dist": distance, "lw": log_wage})
ols_b = smf.ols("lw ~ educ", iv).fit().params["educ"]
first = smf.ols("educ ~ dist", iv).fit()
iv["educ_hat"] = first.fittedvalues
second = smf.ols("lw ~ educ_hat", iv).fit().params["educ_hat"]
print(f"  OLS: {ols_b:.4f}  (biased upward: able people get more education AND earn more)")
print(f"  first stage: educ on distance, slope {first.params['dist']:.4f}, F = {first.fvalue:.0f} (strong instrument)")
print(f"  2SLS: {second:.4f}  (true 0.08); Wald ratio cov(lw, dist)/cov(educ, dist) = "
      f"{np.cov(iv.lw, iv.dist)[0, 1] / np.cov(iv.educ, iv.dist)[0, 1]:.4f}")

print("\n9. Regression discontinuity: a scholarship for entrance scores >= 70")
k = 4000
entrance = rng.uniform(40, 100, k)
scholar = entrance >= 70
later_gpa = 1.5 + 0.02 * entrance + 0.25 * scholar + rng.normal(0, 0.3, k)    # true effect +0.25
rd = pd.DataFrame({"score": entrance - 70, "s": scholar.astype(int), "gpa": later_gpa})
print(f"  naive: scholars vs others {later_gpa[scholar].mean() - later_gpa[~scholar].mean():+.3f}  (confounded by ability)")
for bw in [5, 10, 20]:
    local = rd[np.abs(rd.score) <= bw]
    fit = smf.ols("gpa ~ s * score", local).fit()
    print(f"  local linear RD, bandwidth ±{bw:>2}: jump at the cutoff {fit.params['s']:+.3f} (SE {fit.bse['s']:.3f}, n = {len(local)})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(e[t == 1], bins=40, alpha=0.5, density=True, label="treated")
axes[0].hist(e[t == 0], bins=40, alpha=0.5, density=True, label="control")
axes[0].set_title("Propensity scores: check overlap")
axes[0].set_xlabel("estimated P(treated | x)")
axes[0].legend()
for p, label in [(1, "policy regions"), (0, "other regions")]:
    axes[1].plot([0, 1], means.loc[p], marker="o", label=label)
axes[1].plot([0, 1], [means.loc[1, 0], means.loc[1, 0] + means.loc[0, 1] - means.loc[0, 0]], ls="--", color="grey",
             label="counterfactual (parallel trend)")
axes[1].set_xticks([0, 1], ["before", "after"])
axes[1].set_title(f"Difference-in-differences: effect {did:+.1f}")
axes[1].legend(fontsize=8)
binned = rd.assign(bin=(rd.score // 2) * 2).groupby("bin")["gpa"].mean()
axes[2].scatter(binned.index + 70, binned.values, s=12)
axes[2].axvline(70, color="grey", ls="--")
axes[2].set_title("Regression discontinuity at score 70")
axes[2].set_xlabel("entrance score")
axes[2].set_ylabel("later GPA (bin means)")
fig.tight_layout()
fig.savefig(FIGURES / "observational_methods.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "observational_methods.png")
