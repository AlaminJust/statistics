# 06.06 — Splines and mixed models

**Time:** about 5 hours · **Prerequisites:** 06.02–06.05, 05.01 (shrinkage)

## Why this matters

Two assumptions of basic regression fail constantly in real data: relationships are often **curved**, and observations
often come in **groups** (students in schools, measurements on patients, users in countries, runs on datasets).
**Splines** model curves flexibly without the wild behaviour of high-degree polynomials. **Mixed (multilevel) models**
handle grouped data by estimating how much the groups differ and "borrowing strength" across them. That **partial
pooling** is the same idea as shrinkage (Lesson 05.01) and the heart of Bayesian hierarchical models (Module 07).

## Learning objectives

- Model non-linear relationships with polynomial and spline bases, and choose their flexibility by cross-validation.
- Explain why splines behave better than high-degree polynomials.
- Distinguish pooled, fixed-effects and random-effects (mixed) models for grouped data.
- Interpret variance components and the intraclass correlation.
- Explain partial pooling, and show when it improves estimates.

## Before you start

A school with only 3 students tested has the highest average score of 40 schools. Would you bet that it's really the
best school?

---

## Part 1: curves

### 1. Basis expansions

A linear model is linear in the *coefficients*, not necessarily in x. Replace x by several functions of x (a
**basis**) and fit by ordinary least squares:

- polynomial: 1, x, x², x³, …;
- splines: piecewise polynomials joined smoothly at **knots**.

**Worked example 1:** performance vs hours of practice. The true curve rises quickly, levels off, then falls after
7 hours (fatigue). 10-fold cross-validated RMSE:

| Model | CV RMSE |
| --- | --- |
| straight line | 7.38 |
| polynomial, degree 3 | 4.28 |
| polynomial, degree 15 | 7.26: wild wiggles at the edges |
| **cubic spline, 5 knots** | **4.05** |
| cubic spline, 20 knots | 4.17 |

(Shuffle the folds when the data are sorted by x, as here. Unshuffled folds would each be an extrapolation, and every
flexible model would look terrible. Module 08 covers this.)

### 2. Why splines beat high-degree polynomials

A polynomial's coefficients are **global**: changing the data in one place moves the curve everywhere, and high
degrees explode near the edges of the data (Runge's phenomenon). A spline's basis functions are **local**
(`splines.png`, right panel): each one is non-zero only over a few neighbouring intervals, so the curve bends where the
data say and stays calm elsewhere. **Cubic** splines are smooth (continuous first and second derivatives) at the knots.
*Natural* splines are also linear beyond the outer knots, which makes extrapolation safer.

In practice:

```python
make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression())   # scikit-learn
smf.ols("y ~ bs(x, df=5)", data)                                            # statsmodels formula (B-spline)
```

**Worked example 2.** The spline raises R² from 0.14 (straight line) to **0.75**, and the nested F-test for the extra
curvature gives p ≈ 10⁻⁶³. You can mix spline terms with ordinary linear terms and dummies.

### 3. Choosing the flexibility

More knots means less bias and more variance: the bias–variance trade-off again (Module 08). Choose the number of knots
by cross-validation, or use **penalised splines** (many knots plus a smoothness penalty, as in generalised additive
models, GAMs). A GAM, y = β₀ + f₁(x₁) + f₂(x₂) + …, gives a smooth curve per predictor while staying interpretable.
That's a strong middle ground between linear models and black-box ML.

## Part 2: grouped data

### 4. Three ways to handle groups

Data: 527 students in 30 schools (5 to 29 per school); score = 45 + school effect + 2 × hours + noise, with school
effects ~ N(0, 6²) and noise sd 8.

| Approach | Model | Hours coefficient (SE) | What it assumes |
| --- | --- | --- | --- |
| **complete pooling** | `score ~ hours` | 2.003 (0.078) | schools don't differ |
| **no pooling (fixed effects)** | `score ~ hours + C(school)` | 2.017 (0.071) | each school is unrelated to the others |
| **partial pooling (mixed model)** | `mixedlm("score ~ hours", groups=school)` | 2.014 (0.071) | school effects come from a common distribution |

A **random-intercept model**:

$$\text{score}_{ij} = \beta_0 + u_j + \beta_1\,\text{hours}_{ij} + \varepsilon_{ij}, \qquad u_j \sim N(0, \tau^2), \quad \varepsilon_{ij} \sim N(0, \sigma^2).$$

Here j indexes schools and i students. It estimates two **variance components**: τ (how much schools differ) =
**4.05** and σ (the student-level noise) = **7.89**. The **intraclass correlation**, ICC = τ²/(τ² + σ²) = **0.21**, is
the share of the variation that lies between schools, and the correlation between two students from the same school.

The truth was τ = 6, but the 30 school effects actually drawn had sd 4.33. With only 30 groups, τ is estimated
imprecisely; you need many groups to pin down a between-group variance.

**When does the choice matter?** Here the hours coefficient is similar in all three, because hours vary *within*
schools. Ignoring clustering matters most for predictors that vary *between* groups (a school-level policy). Then
pooled OLS gives far too small SEs: in Lesson 06.03, exercise 10, a null treatment assigned to whole classrooms was
"significant" 38% of the time.

### 5. Partial pooling: borrowing strength

A mixed model's estimate of each group's mean is a **weighted average** of that group's own mean and the overall mean:

$$\hat\mu_j = w_j\,\bar{y}_j + (1 - w_j)\,\hat\mu, \qquad w_j = \frac{\tau^2}{\tau^2 + \sigma^2/n_j}.$$

- A large school (big nⱼ) has wⱼ ≈ 1 and keeps its own mean.
- A small school (small nⱼ) is pulled strongly towards the overall mean, because its own mean is mostly noise.

**Worked example 4 (Before you start).** 40 schools of 2–20 students, true school means ~ N(60, 3²), student noise
sd 10:

| School | n | raw mean | shrunk estimate | truth |
| --- | --- | --- | --- | --- |
| 37 | 2 | 65.17 | **61.57** | 59.73 |
| 3 | 2 | 60.97 | 60.69 | 59.88 |
| 26 | 3 | 65.39 | **61.98** | 61.50 |

Across 100 simulations, the error of the school estimates is **3.63** for raw means and **2.29** for partial pooling,
**37% smaller** (`partial_pooling.png`). The tiny school at the top of the raw ranking is usually there because of luck:
with 2–3 students, its average is mostly noise. Don't bet on it.

This is the same phenomenon as regression to the mean (Lesson 06.01) and the James–Stein result (Lesson 05.01). It's
used for ranking hospitals, estimating batting averages, small-area statistics, recommender systems (shrink a new
product's average rating towards the category average) and A/B tests across many segments.

### 6. Random slopes and beyond

Groups can differ in their slopes too: `mixedlm("score ~ hours", data, groups=school, re_formula="~hours")` gives each
school its own hours effect, drawn from a common distribution. Generalised linear mixed models extend this to binary and
count outcomes, and Module 07 fits fully Bayesian hierarchical versions, which handle small numbers of groups better.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| High-degree polynomials for curves | Splines or GAMs |
| Cross-validating on sorted data without shuffling | Shuffle (or use a design-appropriate split) |
| Ignoring grouping when the predictor varies between groups | Mixed model or cluster-robust SEs |
| Ranking groups by raw means when sizes differ a lot | Partial pooling (shrunk estimates) |
| Trusting τ from a handful of groups | You need many groups; consider Bayesian priors |

## Summary

- Basis expansions keep regression linear in the coefficients; splines are local and smooth, polynomials are global and wiggly.
- Choose flexibility by CV or with a smoothness penalty (GAMs).
- Grouped data: complete pooling (ignore groups), no pooling (fixed effects), partial pooling (mixed models).
- A random-intercept model estimates τ² (between groups) and σ² (within); ICC = τ²/(τ² + σ²).
- Partial pooling shrinks small groups towards the overall mean and gives more accurate group estimates.

## Self-test

1. Why does a degree-15 polynomial behave badly near the edges of the data?
2. In a random-intercept model, what does an ICC of 0.5 mean?
3. Write the partial-pooling weight wⱼ and explain what happens as nⱼ → ∞.
4. Give a real example where you'd use a mixed model rather than OLS.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
