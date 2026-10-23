"""06.02 Multiple regression in matrix form — fitting, interpreting, dummies, interactions, omitted variables.

Run:  python curriculum/06-regression/02-multiple-regression-matrix-form/examples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
rng = np.random.default_rng(62)
df = pd.read_csv(ROOT / "data" / "students.csv")
df["tutored"] = (df["tutoring"] == "yes").astype(int)

print("Worked example 1: the normal equations by hand, three predictors")
X = np.column_stack([np.ones(len(df)), df["study_hours"], df["sleep_hours"], df["tutored"]])
y = df["exam_score"].to_numpy(dtype=float)
beta = np.linalg.solve(X.T @ X, X.T @ y)
names = ["intercept", "study_hours", "sleep_hours", "tutored"]
truth = [40 - 2 * 7, 2.2, 2.0, 5.0]
for name, b, t in zip(names, beta, truth):
    print(f"  {name:<12} estimate {b:8.3f}   true value used to generate the data {t:6.1f}")
resid = y - X @ beta
sigma2 = resid @ resid / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
print("  standard errors sqrt(diag(σ̂² (XᵀX)⁻¹)):", np.round(se, 3))

print("\nWorked example 2: the same with statsmodels' formula interface")
full = smf.ols("exam_score ~ study_hours + sleep_hours + tutored", data=df).fit()
print(full.summary().tables[1])
print(f"  R² = {full.rsquared:.4f}, adjusted R² = {full.rsquared_adj:.4f}, residual SE = {np.sqrt(full.scale):.3f}")

print("\nWorked example 3: omitted-variable bias — what happens to the tutoring coefficient")
alone = smf.ols("exam_score ~ tutored", data=df).fit()
print(f"  score ~ tutored only:          tutoring coefficient {alone.params['tutored']:.2f} (SE {alone.bse['tutored']:.2f})")
print(f"  score ~ tutored + study + sleep: tutoring coefficient {full.params['tutored']:.2f} (SE {full.bse['tutored']:.2f})")
# A confounded simulation where the bias is large:
n = 2000
motivation = rng.normal(0, 1, n)
hours = 10 + 3 * motivation + rng.normal(0, 2, n)
tutor = (motivation + rng.normal(0, 1, n) > 0.5).astype(int)
score = 40 + 2.0 * hours + 3.0 * tutor + rng.normal(0, 8, n)
sim = pd.DataFrame({"score": score, "hours": hours, "tutor": tutor})
b_naive = smf.ols("score ~ tutor", sim).fit().params["tutor"]
b_adj = smf.ols("score ~ tutor + hours", sim).fit().params["tutor"]
print(f"  simulated confounding (true tutoring effect 3): naive {b_naive:.2f}, adjusting for hours {b_adj:.2f}")

print("\nWorked example 4: a categorical predictor — dummy variables")
prog = smf.ols("exam_score ~ C(program)", data=df).fit()
print(prog.params.round(3))
print("  group means:", df.groupby("program")["exam_score"].mean().round(3).to_dict())
print("  -> intercept = baseline group (Arts); each coefficient = that group's mean minus Arts' mean")
prog2 = smf.ols("exam_score ~ C(program, Treatment(reference='Science'))", data=df).fit()
print("  with Science as the baseline:", prog2.params.round(3).to_dict())
print(f"  overall F-test for program: F = {prog.fvalue:.3f}, p = {prog.f_pvalue:.3f} (same as one-way ANOVA in 05.06)")

print("\nWorked example 5: an interaction — does the study-hours slope differ with tutoring?")
inter = smf.ols("exam_score ~ study_hours * tutored", data=df).fit()
print(inter.params.round(3))
print(f"  slope without tutoring {inter.params['study_hours']:.3f}; with tutoring "
      f"{inter.params['study_hours'] + inter.params['study_hours:tutored']:.3f}; "
      f"interaction p = {inter.pvalues['study_hours:tutored']:.3f}")

print("\nWorked example 6: standardised coefficients — which predictor matters most?")
z = df[["exam_score", "study_hours", "sleep_hours", "commute_minutes"]].apply(lambda s: (s - s.mean()) / s.std())
std_model = smf.ols("exam_score ~ study_hours + sleep_hours + commute_minutes", data=z).fit()
print(std_model.params.round(3).drop("Intercept").to_dict())
print("  (change in score SDs per one-SD change in each predictor, holding the others fixed)")

print("\nWorked example 7: adjusted R² penalises useless predictors")
df["noise1"], df["noise2"], df["noise3"] = rng.normal(size=(3, len(df)))
bigger = smf.ols("exam_score ~ study_hours + sleep_hours + tutored + noise1 + noise2 + noise3", data=df).fit()
print(f"  3 real predictors:      R² {full.rsquared:.4f}  adjusted {full.rsquared_adj:.4f}")
print(f"  + 3 pure-noise columns: R² {bigger.rsquared:.4f}  adjusted {bigger.rsquared_adj:.4f}")
print(f"  AIC {full.aic:.1f} vs {bigger.aic:.1f};  BIC {full.bic:.1f} vs {bigger.bic:.1f}  (lower is better)")
