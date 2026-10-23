# 02.06 — Two variables and correlation

**Time:** about 4 hours · **Prerequisites:** 01.05, 02.03, 02.05

## Why this matters

Most questions are about relationships: does studying more go with higher scores? Do tutored students do better? Are
two features in my model telling me the same thing? This lesson covers the tools for describing two variables
together: contingency tables for categories, and covariance and correlation for numbers. It also covers the most
important warning in statistics, **correlation is not causation**. In ML, correlation matrices are used for feature
screening and for spotting redundant features (multicollinearity, Module 06), and the correlation coefficient is a
cosine similarity, the same quantity used to compare embeddings.

## Learning objectives

- Build and read contingency tables with joint, marginal and conditional proportions.
- Compute covariance and Pearson's correlation by hand and in code, and interpret them.
- Explain why correlation only measures *linear* association, using Anscombe's quartet.
- Compute Spearman's rank correlation and say when to prefer it.
- Explain why correlation does not imply causation, and name three alternative explanations.

## Before you start

Across many cities, the number of ice-cream sales and the number of drownings per day are strongly correlated. Should
cities ban ice cream? Write down what else could explain the pattern.

---

## 1. Two categorical variables: contingency tables

A **contingency table** (cross-tabulation) counts every combination of two categorical variables.

**Worked example 1: program × tutoring** in `students.csv`.

| program | no | yes | total |
| --- | --- | --- | --- |
| Arts | 52 | 20 | 72 |
| Commerce | 37 | 12 | 49 |
| Science | 51 | 28 | 79 |
| **total** | **140** | **60** | **200** |

Three kinds of proportion answer three different questions:

| Proportion | Divide by | Example | Answer |
| --- | --- | --- | --- |
| **Joint** | grand total | P(Science and tutoring) | 28/200 = 0.14 |
| **Marginal** | grand total, using a row or column total | P(tutoring) | 60/200 = 0.30 |
| **Conditional** | the row (or column) total | P(tutoring \| Science) | 28/79 = 0.354 |

Conditional proportions by row (`pd.crosstab(..., normalize="index")`):

| program | P(no \| program) | P(yes \| program) |
| --- | --- | --- |
| Arts | 0.722 | 0.278 |
| Commerce | 0.755 | 0.245 |
| Science | 0.646 | **0.354** |

Science students attend tutoring more often (35% vs about 25–28%). **If the row-conditional proportions were the same
in every row, the two variables would be unrelated** (independent). Module 05 tests whether gaps like these are bigger
than chance (the χ² test). Module 03 turns these proportions into conditional probability and Bayes' theorem.

**Always condition in the direction of the question.** P(tutoring | Science) = 0.354 and P(Science | tutoring) =
28/60 = 0.467 are different numbers answering different questions.

## 2. One numeric and one categorical variable

Compare the groups' distributions: side-by-side box plots, overlaid ECDFs (02.05), and a table of group summaries:

```python
df.groupby("tutoring")["exam_score"].agg(["count", "mean", "median", "std"])
```

Tutoring students: mean 69.7; others: 61.5. Report the difference **together with** the spreads (sd about 14).
An 8-point gap means more when spreads are 3 points than when they are 30.

## 3. Two numeric variables: covariance

Covariance measures whether two variables move **together**:

$$\text{cov}(x, y) = s_{xy} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}).$$

For each point, multiply its x-deviation by its y-deviation:

| Point is… | x − x̄ | y − ȳ | product |
| --- | --- | --- | --- |
| above average on both | + | + | **+** |
| below average on both | − | − | **+** |
| above on one, below on the other | ± | ∓ | **−** |

If most products are positive, the covariance is positive (the variables rise together).

**Worked example 2.** Five students: hours x = (2, 4, 6, 8, 10), scores y = (50, 60, 65, 80, 85). x̄ = 6, ȳ = 68.

| xᵢ | yᵢ | xᵢ − x̄ | yᵢ − ȳ | product |
| --- | --- | --- | --- | --- |
| 2 | 50 | −4 | −18 | 72 |
| 4 | 60 | −2 | −8 | 16 |
| 6 | 65 | 0 | −3 | 0 |
| 8 | 80 | 2 | 12 | 24 |
| 10 | 85 | 4 | 17 | 68 |
| | | | **sum** | **180** |

cov = 180 / 4 = **45 (hour·points)**.

The problem with covariance: **its size depends on the units.** Measure study time in minutes and the covariance becomes
45 × 60 = 2,700, for exactly the same relationship. You can't tell from 45 or 2,700 whether the relationship is strong.

## 4. Pearson's correlation coefficient r

Divide by both standard deviations to remove the units:

$$r = \frac{s_{xy}}{s_x\,s_y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\,\sqrt{\sum (y_i - \bar{y})^2}}.$$

**Worked example 2, continued.** s_x = √(40/4) = √10 ≈ 3.162 and s_y = √(830/4) = √207.5 ≈ 14.405.
r = 45 / (3.162 × 14.405) = 45 / 45.55 ≈ **0.988**.

The same in minutes: r = 2,700 / (189.7 × 14.405) = 0.988. **Correlation doesn't depend on units.**

**Correlation is a cosine similarity.** The second formula is exactly the cosine of the angle between the *centred*
vectors (x − x̄) and (y − ȳ) (Lesson 01.05). `examples.py` checks that both give 0.988.

### Properties

| Property | Meaning |
| --- | --- |
| −1 ≤ r ≤ 1 | ±1 means the points lie exactly on a straight line |
| sign = direction | positive: rise together; negative: one rises as the other falls |
| unitless; unchanged by shifting or by rescaling with a positive factor | minutes vs hours gives the same r |
| symmetric | r(x, y) = r(y, x) |
| r² = share of variance "explained" by a straight line | study vs score: r = 0.728, so r² = 0.53, i.e. 53% (Module 06) |
| measures **linear** association only | a perfect curve can have r = 0 |

Rough verbal guide, for |r| (context matters a lot): < 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, 0.5–0.7
strong, > 0.7 very strong.

**Correlation matrix of `students.csv`:**

| | study | sleep | commute | score |
| --- | --- | --- | --- | --- |
| study_hours | 1 | −0.08 | −0.01 | **0.73** |
| sleep_hours | −0.08 | 1 | −0.04 | 0.13 |
| commute_minutes | −0.01 | −0.04 | 1 | −0.02 |
| exam_score | **0.73** | 0.13 | −0.02 | 1 |

Only study hours is strongly related to the score. That matches how the data were generated (`make_students.py`),
where commute plays no role at all.

## 5. Always plot: Anscombe's quartet

Four small datasets, built by the statistician Frank Anscombe in 1973, have **nearly identical** summaries:

| Statistic | Set I | Set II | Set III | Set IV |
| --- | --- | --- | --- | --- |
| mean of x | 9.0 | 9.0 | 9.0 | 9.0 |
| mean of y | 7.50 | 7.50 | 7.50 | 7.50 |
| variance of x | 11.0 | 11.0 | 11.0 | 11.0 |
| variance of y | 4.13 | 4.13 | 4.12 | 4.12 |
| correlation r | 0.816 | 0.816 | 0.816 | 0.817 |
| least-squares line | y = 3.00 + 0.50x | same | same | same |

Yet `examples.py` plots them as four completely different pictures: (I) a noisy straight line, (II) a smooth **curve**,
(III) a perfect line with **one outlier**, and (IV) a vertical stack plus **one influential point** that creates the whole
correlation. **r = 0.82 describes only set I well.** Summary statistics never replace a scatter plot.

Two more cases where r misleads:

- **Curved relationships:** for x = −3, …, 3 and y = x², y is completely determined by x, yet r = **0**.
- **Restricted range:** among students who study more than 12 hours (n = 66), r(study, score) drops from 0.73 to
  **0.65**. Selecting a narrow slice of x hides part of the relationship. (Elite universities see a weaker link between
  entrance scores and grades than exists in the whole population.)

## 6. Spearman's rank correlation

Replace each variable by its **ranks** (1 = smallest; tied values get the average of their ranks), then compute
Pearson's r on the ranks.

**Worked example 3.** Ranks of (10, 20, 20, 40) are (1, 2.5, 2.5, 4): the two 20s share ranks 2 and 3.

**Worked example 4.** x = (1, 2, 3, 4, 5), y = (2, 4, 6, 8, 100). Pearson r = **0.74**, pulled down by the 100.
Every increase in x comes with an increase in y, so the ranks agree perfectly: Spearman ρ = **1.0**.

| Use Pearson when | Use Spearman when |
| --- | --- |
| the relationship looks linear | it is monotonic (always rising or always falling) but curved |
| no extreme outliers | there are outliers or heavy skew |
| both variables numeric | one or both variables are ordinal (ratings, ranks) |

## 7. Correlation is not causation

A correlation between X and Y can arise in several ways:

| Explanation | Diagram | Example |
| --- | --- | --- |
| X causes Y | X → Y | study hours → exam score |
| Y causes X (reverse causation) | Y → X | doing well makes students enjoy the subject and study more |
| A **confounder** Z causes both | X ← Z → Y | hot weather → more ice cream **and** more swimming → more drownings |
| Coincidence (with many variables, some correlate by chance) | | divorce rate vs margarine consumption |
| Selection (only certain units are in the data) | | hospital patients show odd correlations between unrelated diseases |

Answer to "Before you start": temperature is a **confounder**. Banning ice cream wouldn't prevent drownings.

To show causation you need a **randomised experiment** or careful causal methods with clearly stated assumptions
(Module 09). Until then, write "is associated with", not "causes" or "leads to".

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "r = 0, so no relationship" | Only no *linear* relationship; plot it |
| Comparing covariances across datasets | Use correlation (unitless) |
| "r = 0.5 means 50% explained" | r² = 0.25 is the explained share |
| Reporting r without a scatter plot | Anscombe: always plot |
| Mixing up P(A \| B) and P(B \| A) | Condition on what the question gives |
| "Correlated, so X causes Y" | Consider reverse causation, confounding, chance, selection |

## Summary

- Contingency tables give joint, marginal and conditional proportions. Equal conditional rows mean no association.
- cov = Σ(xᵢ − x̄)(yᵢ − ȳ)/(n − 1): sign = direction, size depends on units.
- r = cov/(s_x s_y): unitless, between −1 and 1, linear only. It is the cosine of the centred vectors.
- Spearman = Pearson on ranks: robust, and it captures monotonic curves.
- Plot first. Correlation ≠ causation.

## Self-test

1. From the table in section 1, compute P(Arts | no tutoring).
2. If r(x, y) = 0.6, what is r(2x + 5, y)? What about r(−x, y)?
3. Give an example of two variables with r ≈ 0 that are strongly related.
4. When would you report Spearman rather than Pearson?
5. Shoe size and reading ability are correlated in children. Name the confounder.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
