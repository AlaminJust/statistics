# 09.02 Exercises — Potential outcomes and DAGs

`make lesson L=09-causal-and-research/02-potential-outcomes-and-dags`

## ★ Warm-up

1. **(M)** In the lesson's six-student table, compute the average effect on the treated (ATT, students A, C, E) and on the
   untreated (ATC). Why do they differ from the ATE?
2. **(C)** For each, draw the DAG and say whether Z is a confounder, mediator or collider: (a) X = exercise, Y = weight,
   Z = age; (b) X = education, Y = income, Z = job type; (c) X = smoking, Y = lung disease, Z = being in a hospital study.
3. **(C)** Why is "adjust for everything" a bad rule? Give one example of each way it goes wrong.

## ★★ Core

4. **(M)** Prove the decomposition: E[Y | T=1] − E[Y | T=0] = ATT + {E[Y(0) | T=1] − E[Y(0) | T=0]}.
5. **(P)** Simulate the "M-bias" structure: X ← U₁ → Z ← U₂ → Y with no effect of X on Y (U₁ and U₂ unobserved). Show that
   y ~ x gives about 0, but y ~ x + z gives a non-zero coefficient. Explain with the backdoor criterion.
6. **(P)** Simulate a hospital dataset: diseases A and B are independent in the population, each raises the chance of
   hospitalisation. Show they're negatively correlated among hospitalised patients (Berkson's paradox).
7. **(P)** Write `ate_difference_in_means(y, t)` (returning (ATE, SE) using the Welch SE) in `statkit/causal.py`.
8. **(P)** Using the Berkeley data, fit a logistic regression of admission on gender, with and without department dummies
   (expand the counts to individual rows, or use a binomial GLM with counts). Interpret both gender coefficients.

## ★★★ Challenge

9. **(C)** Draw a DAG for this question: "Does using our app's study planner raise students' grades?" Include at least a
   confounder, a mediator, a collider and an unmeasured variable. Write down a valid adjustment set, or explain why none
   exists.
10. **(P)** **Front-door adjustment.** Simulate U → X, U → Y (U unobserved) and X → M → Y with no other path. Show that you
    can recover the effect of X on Y as (effect of X on M) × (effect of M on Y adjusting for X), even though U is
    unmeasured.
11. **(C)** A fairness audit compares a hiring model's acceptance rates by gender, then "controls for" job level. Using
    the mediator/collider ideas, explain how controlling for job level could hide or create disparities.

Solutions: [solutions/solutions.md](solutions/solutions.md)
