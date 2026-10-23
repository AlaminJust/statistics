"""06.04 Logistic regression and GLMs — logit link, odds ratios, IRLS, Poisson regression, overdispersion.

Run:  python curriculum/06-regression/04-logistic-regression-and-glms/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(64)
df = pd.read_csv(ROOT / "data" / "students.csv")
df["high"] = (df["exam_score"] >= 70).astype(int)
df["tutored"] = (df["tutoring"] == "yes").astype(int)

print("Worked example 1: why not a straight line for a 0/1 outcome?")
lpm = smf.ols("high ~ study_hours", df).fit()
print(f"  linear probability model predicts P(high) at 0 h = {lpm.predict(pd.DataFrame({'study_hours': [0]})).iloc[0]:.3f}"
      f" and at 30 h = {lpm.predict(pd.DataFrame({'study_hours': [30]})).iloc[0]:.3f}  <- outside [0, 1]")

print("\nWorked example 2: logistic regression of P(score >= 70) on study hours and tutoring")
logit = smf.logit("high ~ study_hours + tutored", df).fit(disp=0)
print(logit.summary().tables[1])
odds_ratios = np.exp(logit.params)
ci = np.exp(logit.conf_int())
for name in ["study_hours", "tutored"]:
    print(f"  odds ratio {name}: {odds_ratios[name]:.3f} (95% CI {ci.loc[name, 0]:.3f} to {ci.loc[name, 1]:.3f})")
for hours in [5, 10, 15, 20]:
    p0 = logit.predict(pd.DataFrame({"study_hours": [hours], "tutored": [0]})).iloc[0]
    p1 = logit.predict(pd.DataFrame({"study_hours": [hours], "tutored": [1]})).iloc[0]
    print(f"  {hours:>2} h: P(high) = {p0:.3f} without tutoring, {p1:.3f} with  (difference {p1 - p0:+.3f})")

print("\nWorked example 3: fitting by IRLS (Newton's method) from scratch")
X = np.column_stack([np.ones(len(df)), df["study_hours"], df["tutored"]])
y = df["high"].to_numpy()
beta = np.zeros(3)
for it in range(25):
    p = 1 / (1 + np.exp(-X @ beta))
    W = p * (1 - p)
    gradient = X.T @ (y - p)
    hessian = X.T @ (X * W[:, None])
    step = np.linalg.solve(hessian, gradient)
    beta = beta + step
    if np.max(np.abs(step)) < 1e-10:
        break
se = np.sqrt(np.diag(np.linalg.inv(hessian)))
print(f"  converged in {it + 1} iterations: beta = {np.round(beta, 4)}, SE = {np.round(se, 4)}")
print(f"  statsmodels:                     beta = {np.round(logit.params.to_numpy(), 4)}")

print("\nWorked example 4: deviance and the likelihood-ratio test")
null = smf.logit("high ~ 1", df).fit(disp=0)
lr = 2 * (logit.llf - null.llf)
print(f"  log-likelihood: null {null.llf:.2f}, model {logit.llf:.2f}; LR statistic {lr:.2f} on 2 df, p = {logit.llr_pvalue:.1e}")
print(f"  McFadden pseudo-R² = 1 - llf/llnull = {1 - logit.llf / null.llf:.3f}")
pred = (logit.predict(df) >= 0.5).astype(int)
print(f"  accuracy at threshold 0.5: {(pred == y).mean():.3f} (always predicting 'not high' would give {1 - y.mean():.3f})")

print("\nWorked example 5: Poisson regression with an exposure offset — website errors per day")
n_days = 300
traffic = rng.uniform(1, 20, n_days)                 # thousands of visits per day
new_release = rng.integers(0, 2, n_days)             # 1 if the day used the new release
rate = np.exp(-1.0 + 0.4 * new_release)              # errors per thousand visits
errors = rng.poisson(rate * traffic)
logs = pd.DataFrame({"errors": errors, "traffic": traffic, "new_release": new_release})
pois = smf.glm("errors ~ new_release", logs, family=sm.families.Poisson(), offset=np.log(logs["traffic"])).fit()
print(pois.summary().tables[1])
print(f"  rate ratio for the new release: exp({pois.params['new_release']:.3f}) = {np.exp(pois.params['new_release']):.3f}"
      f"  (true exp(0.4) = {np.exp(0.4):.3f})")
print(f"  Pearson chi2 / df = {pois.pearson_chi2 / pois.df_resid:.3f}  (about 1 means no overdispersion)")

print("\nWorked example 6: overdispersion — when counts vary more than Poisson allows")
mu_i = np.exp(-1.0 + 0.4 * new_release) * traffic
over = rng.negative_binomial(n=2, p=2 / (2 + mu_i))   # same mean, extra variance
logs["errors_over"] = over
pois2 = smf.glm("errors_over ~ new_release", logs, family=sm.families.Poisson(), offset=np.log(logs["traffic"])).fit()
nb = smf.glm("errors_over ~ new_release", logs, family=sm.families.NegativeBinomial(alpha=0.5),
             offset=np.log(logs["traffic"])).fit()
print(f"  Poisson fit:           coef {pois2.params['new_release']:.3f}, SE {pois2.bse['new_release']:.3f}, "
      f"dispersion {pois2.pearson_chi2 / pois2.df_resid:.2f}")
print(f"  negative binomial fit: coef {nb.params['new_release']:.3f}, SE {nb.bse['new_release']:.3f}")
print("  -> similar estimate, but the Poisson SE is far too small")

fig, ax = plt.subplots(figsize=(6.5, 4))
grid = np.linspace(0, 28, 200)
for t, label in [(0, "not tutored"), (1, "tutored")]:
    ax.plot(grid, logit.predict(pd.DataFrame({"study_hours": grid, "tutored": t})), label=label)
jitter = rng.uniform(-0.03, 0.03, len(df))
ax.scatter(df["study_hours"], df["high"] + jitter, s=8, alpha=0.4, color="grey")
ax.set_title("Logistic regression: P(score ≥ 70) by study hours")
ax.set_xlabel("study hours per week")
ax.set_ylabel("probability (dots: observed 0/1, jittered)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "logistic_curves.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "logistic_curves.png")
