"""09.02 Potential outcomes and DAGs — confounders, mediators, colliders and Simpson's paradox.

Run:  python curriculum/09-causal-and-research/02-potential-outcomes-and-dags/examples.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(92)

print("Worked example 1: potential outcomes for six students (we only ever see one of Y(1), Y(0))")
table = pd.DataFrame({
    "student": list("ABCDEF"),
    "Y0": [60, 70, 55, 80, 65, 75],        # score without tutoring
    "Y1": [68, 72, 63, 81, 70, 80],        # score with tutoring
    "tutored": [1, 0, 1, 0, 1, 0],
})
table["effect"] = table["Y1"] - table["Y0"]
table["observed"] = np.where(table["tutored"] == 1, table["Y1"], table["Y0"])
print(table.to_string(index=False))
print(f"  true average treatment effect (ATE) = mean(Y1 - Y0) = {table['effect'].mean():.2f}")
naive = table.loc[table.tutored == 1, "observed"].mean() - table.loc[table.tutored == 0, "observed"].mean()
print(f"  naive difference in observed means = {naive:.2f}  <- the weaker students chose tutoring")

print("\nWorked example 2: randomisation recovers the ATE on average")
estimates = []
for _ in range(20_000):
    t = rng.permutation([1, 1, 1, 0, 0, 0])
    obs = np.where(t == 1, table["Y1"], table["Y0"])
    estimates.append(obs[t == 1].mean() - obs[t == 0].mean())
print(f"  average over 20,000 random assignments: {np.mean(estimates):.3f} (true ATE {table['effect'].mean():.3f})")

print("\nThree causal structures, simulated (n = 100,000). True effect of X on Y is 2 in every case.")
n = 100_000


def fit(formula, data):
    return smf.ols(formula, data).fit().params["x"]


# 1. Confounder: Z -> X and Z -> Y
z = rng.normal(size=n)
x = 1.5 * z + rng.normal(size=n)
y = 2 * x + 3 * z + rng.normal(size=n)
d = pd.DataFrame({"x": x, "y": y, "z": z})
print(f"  CONFOUNDER  Z→X, Z→Y : y ~ x {fit('y ~ x', d):.3f}   y ~ x + z {fit('y ~ x + z', d):.3f}   -> adjust for Z")

# 2. Mediator: X -> M -> Y (and X -> Y directly with effect 0.5; total effect 0.5 + 1.5 * 1 = 2)
x = rng.normal(size=n)
m = 1.5 * x + rng.normal(size=n)
y = 0.5 * x + 1.0 * m + rng.normal(size=n)
d = pd.DataFrame({"x": x, "y": y, "m": m})
print(f"  MEDIATOR    X→M→Y    : y ~ x {fit('y ~ x', d):.3f}   y ~ x + m {fit('y ~ x + m', d):.3f}   -> DON'T adjust for M (it removes part of the effect)")

# 3. Collider: X -> C <- Y
x = rng.normal(size=n)
y = 2 * x + rng.normal(size=n)
c = x + y + rng.normal(size=n)
d = pd.DataFrame({"x": x, "y": y, "c": c})
print(f"  COLLIDER    X→C←Y    : y ~ x {fit('y ~ x', d):.3f}   y ~ x + c {fit('y ~ x + c', d):.3f}   -> DON'T adjust for C")

print("\nWorked example 3: collider bias by selection — talent and looks among famous actors")
talent = rng.normal(size=n)
looks = rng.normal(size=n)
famous = (talent + looks) > 2.0
print(f"  correlation in the whole population: {np.corrcoef(talent, looks)[0, 1]:+.3f}")
print(f"  correlation among the famous ({famous.mean():.1%}): {np.corrcoef(talent[famous], looks[famous])[0, 1]:+.3f}")

print("\nWorked example 4: Simpson's paradox — UC Berkeley graduate admissions, fall 1973 (six largest departments)")
berkeley = pd.DataFrame({
    "dept": list("ABCDEF") * 2,
    "gender": ["men"] * 6 + ["women"] * 6,
    "applied": [825, 560, 325, 417, 191, 373, 108, 25, 593, 375, 393, 341],
    "admitted": [512, 353, 120, 138, 53, 22, 89, 17, 202, 131, 94, 24],
})
totals = berkeley.groupby("gender")[["applied", "admitted"]].sum()
totals["rate"] = totals["admitted"] / totals["applied"]
print("  overall:", {g: f"{r:.1%}" for g, r in totals["rate"].items()})
wide = berkeley.assign(rate=berkeley.admitted / berkeley.applied).pivot(index="dept", columns="gender", values=["rate", "applied"])
print("  by department (admission rate, number of applicants):")
for dept in "ABCDEF":
    print(f"    {dept}: men {wide.loc[dept, ('rate', 'men')]:.0%} of {int(wide.loc[dept, ('applied', 'men')]):>3}, "
          f"women {wide.loc[dept, ('rate', 'women')]:.0%} of {int(wide.loc[dept, ('applied', 'women')]):>3}")
women_share = (berkeley[berkeley.gender == "women"].set_index("dept")["applied"] / totals.loc["women", "applied"])
dept_rate = berkeley.groupby("dept")[["admitted", "applied"]].sum().pipe(lambda t: t.admitted / t.applied)
print(f"  women applied mostly to departments with low admission rates: "
      f"{(women_share * dept_rate).sum():.1%} expected rate for women's applications vs "
      f"{((berkeley[berkeley.gender == 'men'].set_index('dept')['applied'] / totals.loc['men', 'applied']) * dept_rate).sum():.1%} for men's")
