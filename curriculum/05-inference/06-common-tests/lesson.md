# 05.06 — Common tests: t, χ², ANOVA and non-parametric

**Time:** about 5 hours · **Prerequisites:** 05.05, 04.05

## Why this matters

A small set of tests covers most practical questions: comparing a mean with a value, comparing two groups, comparing
before and after, comparing several groups, and checking whether two categorical variables are related. Knowing
**which test fits which design**, and what each one assumes, is the difference between a correct analysis and a
misleading one. The most common error in published research is not a wrong formula but a test that doesn't match how
the data were collected, such as an unpaired test on paired data.

## Learning objectives

- Run and interpret one-sample, two-sample (Welch) and paired t-tests.
- Run χ² tests of independence and of goodness of fit.
- Run a one-way ANOVA and explain its F statistic.
- Use Mann–Whitney, Wilcoxon and Kruskal–Wallis when normality is doubtful.
- Choose the right test from the design.

## Before you start

30 students take a test, attend a revision course, and take a similar test again. You want to know if they improved.
Should you compare the "before" and "after" groups with a two-sample t-test?

---

## 1. Choosing a test

| Question | Data | Parametric test | Non-parametric alternative |
| --- | --- | --- | --- |
| Is a mean equal to μ₀? | one numeric sample | one-sample t | Wilcoxon signed-rank |
| Do two independent groups differ? | numeric, 2 groups | **Welch two-sample t** | Mann–Whitney U |
| Did the same units change? | numeric, paired | **paired t** | Wilcoxon signed-rank |
| Do 3+ groups differ? | numeric, k groups | one-way ANOVA (F) | Kruskal–Wallis |
| Are two categorical variables related? | contingency table | **χ² test of independence** | Fisher's exact (small counts) |
| Do counts match expected proportions? | one categorical | **χ² goodness of fit** | exact multinomial |
| Is a proportion equal to p₀? | successes/n | binomial test | — |
| Are two numeric variables related? | pairs | Pearson correlation test | Spearman correlation test |

## 2. t-tests

All t-tests have the form **t = (estimate − null value)/SE**, compared with a t distribution (Lesson 04.05).

**One-sample.** Is average sleep different from 7 hours? x̄ = 6.955, t = (6.955 − 7)/(1.00/√200) = **−0.63**, df 199,
**p = 0.53**. No evidence of a difference.

**Two-sample (Welch).** Tutoring vs not: t = **3.995**, df = 111.9, **p = 1.2 × 10⁻⁴**.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}.$$

The pooled ("Student") version assumes equal variances and gives a nearly identical result here (t = 3.991, since the
sds are 13.33 and 13.35). **Use Welch by default**: it costs almost nothing when the variances are equal and protects you
when they aren't. (`stats.ttest_ind(a, b, equal_var=False)`.)

### Paired data: the most important design distinction

**Worked example (Before you start).** 30 simulated students improve by about 3 points each, with noise; their
starting scores vary a lot (sd 12).

| Analysis | t | p |
| --- | --- | --- |
| **paired t-test** on the 30 differences | 3.15 | **0.004** |
| WRONG unpaired two-sample test | 0.91 | 0.366 |

The unpaired test compares the spread *between* students (sd ≈ 12) with a 3-point change and finds nothing. The paired
test removes each student's baseline by working with differences (sd 5.3), and it detects the improvement. **A paired
t-test is just a one-sample t-test on the differences.** Pairs arise in before/after studies, matched patients, twins,
and two ML models evaluated on the *same* test examples (Lesson 05.08).

## 3. χ² tests for counts

### Independence in a contingency table

Under independence, the expected count in cell (i, j) is (row total × column total)/grand total. The statistic

$$\chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E}$$

follows approximately a χ² distribution with (rows − 1)(columns − 1) degrees of freedom.

**Program × tutoring:**

| | no (observed / expected) | yes (observed / expected) |
| --- | --- | --- |
| Arts | 52 / 50.4 | 20 / 21.6 |
| Commerce | 37 / 34.3 | 12 / 14.7 |
| Science | 51 / 55.3 | 28 / 23.7 |

χ² = **1.99**, df = 2, **p = 0.37**. The 35% vs 25–28% tutoring gap from Lesson 02.06 is consistent with chance. (It
was: tutoring was assigned independently of program in `make_students.py`.)

Rule: all expected counts should be at least about 5. Otherwise use Fisher's exact test (`stats.fisher_exact`, for 2 × 2
tables).

### Goodness of fit

Do observed counts match a hypothesised distribution? 600 rolls of a (secretly loaded) die give counts
(74, 102, 105, 89, 93, 137) against 100 expected each: χ² = **22.4**, df = 5, **p = 0.0004**. The die is not fair. The
same test checks whether data fit a Poisson (Lesson 04.01) or whether an A/B test split traffic 50/50 as intended (the
"sample ratio mismatch" check).

## 4. One-way ANOVA: comparing several means

Running all pairwise t-tests inflates false positives (Lesson 05.08). ANOVA asks one question: **are all k group means
equal?** It compares variation *between* groups with variation *within* them:

$$F = \frac{\text{SS}_{\text{between}}/(k - 1)}{\text{SS}_{\text{within}}/(n - k)}, \qquad \text{SS}_{\text{between}} = \sum_g n_g(\bar{x}_g - \bar{x})^2.$$

If the groups have equal means, F is around 1 and follows F(k − 1, n − k).

**Exam score by program:** SS_between = 112.6 (df 2), SS_within = 37,992.5 (df 197). F = (112.6/2)/(37,992.5/197) =
**0.29**, **p = 0.75**. No evidence that programs differ. (Study hours by year: F = 1.48, p = 0.22.)

A significant F says *some* means differ, not which. Follow up with corrected pairwise comparisons (Tukey's HSD:
`stats.tukey_hsd`). ANOVA is a special case of regression with a categorical predictor (Module 06).

## 5. Non-parametric (rank-based) tests

These replace values with **ranks** (Lesson 02.06). That makes them robust to outliers and skew, and they don't assume
normality.

| Test | Replaces | Example result |
| --- | --- | --- |
| Mann–Whitney U | two-sample t | tutoring: p = 1.6 × 10⁻⁴ |
| Wilcoxon signed-rank | paired / one-sample t | before/after: p = 0.002 |
| Kruskal–Wallis | one-way ANOVA | program: p = 0.89 |

Mann–Whitney has a lovely interpretation: U/(n₁n₂) estimates **P(a random member of group 1 beats a random member of
group 2)**. For tutoring it's **0.669**. (That probability is also exactly what the ROC AUC measures for a classifier,
Module 08.)

When the data are roughly normal, these tests lose a little power. With outliers or strong skew in small samples, they
can be far more powerful and more valid than t-tests.

## 6. Checking assumptions, sensibly

- **Independence** is the critical assumption for every test above. No test can fix dependence: use paired or
  clustered methods.
- **Normality** matters mainly for small samples. Look at Q–Q plots or skewness. Formal tests such as Shapiro–Wilk
  reject tiny, harmless departures when n is large and miss real ones when n is small, so don't rely on them alone.
  Here both tutoring groups look normal (Shapiro p = 0.91 and 0.96; skew +0.08 and +0.02).
- **Equal variances:** use Welch and stop worrying. (Levene p = 0.97 here.)
- **Expected counts ≥ 5** for χ² tests.

Avoid "test the assumptions, then pick the test" pipelines: they distort error rates. Choose the test from the design
and the data's known properties, and report a robustness check.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Unpaired test on paired data | Paired t or Wilcoxon on the differences |
| Many pairwise t-tests instead of ANOVA | ANOVA first, then corrected post-hoc tests |
| χ² with tiny expected counts | Fisher's exact test |
| Pooled-variance t by habit | Welch |
| "Shapiro p < 0.05, so the t-test is invalid" with n = 500 | The CLT protects the t-test; check skew and Q–Q plots |

## Summary

- Pick the test from the design: one sample / two independent / paired / k groups / categorical.
- t = (estimate − null)/SE. Welch for two groups; paired = one-sample on differences.
- χ² = Σ(O − E)²/E for independence ((r − 1)(c − 1) df) and goodness of fit.
- ANOVA F = between-group / within-group variance.
- Rank tests (Mann–Whitney, Wilcoxon, Kruskal–Wallis) for skew, outliers and ordinal data.

## Self-test

1. Which test: (a) the same users' task times on the old and new app versions; (b) click rates on three page designs
   (clicked or not); (c) salaries of two independent groups, with a few huge outliers?
2. Why does pairing increase power?
3. Compute the expected count for Science × yes in the table above.
4. What does F ≈ 1 suggest in an ANOVA?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
