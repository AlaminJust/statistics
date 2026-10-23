"""05.06 Common tests — t-tests, chi-square, ANOVA and non-parametric tests on real-looking data.

Run:  python curriculum/05-inference/06-common-tests/examples.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
rng = np.random.default_rng(56)
df = pd.read_csv(ROOT / "data" / "students.csv")
yes = df.loc[df["tutoring"] == "yes", "exam_score"]
no = df.loc[df["tutoring"] == "no", "exam_score"]

print("1. One-sample t-test: is the average sleep different from 7 hours?")
sleep = df["sleep_hours"]
t = (sleep.mean() - 7) / (sleep.std() / math.sqrt(len(sleep)))
res = stats.ttest_1samp(sleep, 7)
print(f"   mean {sleep.mean():.3f}, t = {t:.3f}, df = {len(sleep) - 1}, p = {res.pvalue:.3f}")

print("\n2. Two-sample t-tests: tutoring vs not")
welch = stats.ttest_ind(yes, no, equal_var=False)
student = stats.ttest_ind(yes, no, equal_var=True)
print(f"   Welch (unequal variances):  t = {welch.statistic:.3f}, df = {welch.df:.1f}, p = {welch.pvalue:.2e}")
print(f"   Student (pooled variance):  t = {student.statistic:.3f}, df = {student.df:.0f}, p = {student.pvalue:.2e}")
print(f"   group sds: tutoring {yes.std():.2f}, no tutoring {no.std():.2f}")

print("\n3. Paired t-test: the same 30 students before and after a revision course (simulated)")
before = rng.normal(60, 12, 30)
after = before + rng.normal(3, 5, 30)          # each student improves by about 3 points, with noise
paired = stats.ttest_rel(after, before)
unpaired = stats.ttest_ind(after, before)
d = after - before
print(f"   mean improvement {d.mean():.2f}, sd of differences {d.std(ddof=1):.2f}")
print(f"   paired t = {paired.statistic:.2f}, p = {paired.pvalue:.4f}")
print(f"   WRONG unpaired t = {unpaired.statistic:.2f}, p = {unpaired.pvalue:.4f}  (ignores that each pair shares a student)")
print(f"   the paired test is a one-sample t-test on the differences: p = {stats.ttest_1samp(d, 0).pvalue:.4f}")

print("\n4. Chi-square test of independence: program x tutoring")
table = pd.crosstab(df["program"], df["tutoring"])
chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
print("   observed:")
print(table)
print("   expected if independent:")
print(pd.DataFrame(np.round(expected, 1), index=table.index, columns=table.columns))
print(f"   chi2 = Σ (O - E)²/E = {chi2:.3f}, df = (3-1)(2-1) = {dof}, p = {p:.3f}")

print("\n5. Chi-square goodness of fit: is a die fair? (600 simulated rolls of a loaded die)")
counts = np.bincount(rng.choice(6, size=600, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25]), minlength=6)
gof = stats.chisquare(counts)
print(f"   counts {counts.tolist()}, expected 100 each; chi2 = {gof.statistic:.2f}, df = 5, p = {gof.pvalue:.2e}")

print("\n6. One-way ANOVA: does exam score differ across the three programs?")
groups = [g["exam_score"].to_numpy() for _, g in df.groupby("program")]
anova = stats.f_oneway(*groups)
grand = df["exam_score"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
F = (ss_between / 2) / (ss_within / (len(df) - 3))
print(f"   SS_between {ss_between:.1f} (df 2), SS_within {ss_within:.1f} (df {len(df) - 3})")
print(f"   F = {F:.3f}, p = {anova.pvalue:.3f}   (scipy F = {anova.statistic:.3f})")
groups_year = [g["study_hours"].to_numpy() for _, g in df.groupby("year")]
print(f"   study_hours by year: F = {stats.f_oneway(*groups_year).statistic:.3f}, p = {stats.f_oneway(*groups_year).pvalue:.3f}")

print("\n7. Non-parametric alternatives")
mw = stats.mannwhitneyu(yes, no, alternative="two-sided")
print(f"   Mann-Whitney U (tutoring): U = {mw.statistic:.0f}, p = {mw.pvalue:.2e}")
print(f"   P(random tutored student beats random non-tutored student) = U/(n1 n2) = {mw.statistic / (len(yes) * len(no)):.3f}")
wil = stats.wilcoxon(after, before)
print(f"   Wilcoxon signed-rank (paired before/after): p = {wil.pvalue:.4f}")
kw = stats.kruskal(*groups)
print(f"   Kruskal-Wallis (score by program): p = {kw.pvalue:.3f}")

print("\n8. Checking assumptions before a t-test")
for name, g in [("tutoring", yes), ("no tutoring", no)]:
    print(f"   {name:<12} Shapiro-Wilk p = {stats.shapiro(g).pvalue:.3f}, skew {stats.skew(g):+.2f}")
print(f"   Levene test for equal variances: p = {stats.levene(yes, no).pvalue:.3f}")
print("   (with n this large, the CLT makes the t-test fine even if Shapiro-Wilk rejected normality)")
