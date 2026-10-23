# 09.02 — Potential outcomes and causal diagrams

**Time:** about 5 hours · **Prerequisites:** 02.06, 03.03 (conditional independence), 06.02 (omitted-variable bias), 09.01

## Why this matters

"Does tutoring raise scores?" "Did the new feature increase retention?" "Does this drug reduce mortality?" These are
**causal** questions, and correlation can't answer them on its own (Lesson 02.06). Causal inference gives the
vocabulary and the tools: **potential outcomes** define what an effect *is*, and **causal diagrams** (DAGs) show which
variables you must adjust for, and which you must *not*. Getting this wrong (adjusting for a mediator or a collider) can
create or hide effects. It's increasingly central to ML research too: fairness, robustness to distribution shift,
recommendation and policy learning are causal problems.

## Learning objectives

- Define potential outcomes, individual and average treatment effects, and the fundamental problem of causal inference.
- Explain why randomisation identifies the ATE, and why naive comparisons don't.
- Draw DAGs, identify confounders, mediators and colliders, and apply the backdoor criterion.
- Demonstrate confounding, over-adjustment and collider bias by simulation.
- Explain Simpson's paradox with the UC Berkeley admissions data.

## Before you start

Among famous actors, the most talented tend to be the least good-looking, and vice versa. Does that mean talent and good
looks are negatively related in general?

---

## 1. Potential outcomes

For each unit i and a binary treatment T:

- Yᵢ(1): the outcome if treated; Yᵢ(0): the outcome if not treated;
- **individual effect** τᵢ = Yᵢ(1) − Yᵢ(0); **average treatment effect** ATE = E[Y(1) − Y(0)].

**The fundamental problem of causal inference:** we only ever observe one of the two for each unit, Yᵢ = Yᵢ(Tᵢ). The other
is a *counterfactual*. Causal inference is a missing-data problem.

**Worked example 1.** Six students, with both potential outcomes shown (in reality we'd see only one):

| Student | Y(0) | Y(1) | Effect | Tutored | Observed |
| --- | --- | --- | --- | --- | --- |
| A | 60 | 68 | 8 | yes | 68 |
| B | 70 | 72 | 2 | no | 70 |
| C | 55 | 63 | 8 | yes | 63 |
| D | 80 | 81 | 1 | no | 80 |
| E | 65 | 70 | 5 | yes | 70 |
| F | 75 | 80 | 5 | no | 75 |

True ATE = **4.83**: tutoring helps every student. But the naive comparison of observed means is 67.0 − 75.0 = **−8.0**:
tutoring *looks harmful*, because the weaker students chose it.

$$\underbrace{E[Y \mid T{=}1] - E[Y \mid T{=}0]}_{\text{naive}} = \text{ATE on the treated} + \underbrace{E[Y(0) \mid T{=}1] - E[Y(0) \mid T{=}0]}_{\text{selection bias}}.$$

Here the selection bias is hugely negative: students who chose tutoring would have scored much lower without it.

## 2. Why randomisation works

If T is assigned at random, T is independent of (Y(0), Y(1)). Then E[Y | T = 1] = E[Y(1)] and E[Y | T = 0] = E[Y(0)], so the
difference in means **is** the ATE. **Worked example 2:** over 20,000 random assignments of the six students (3 treated),
the average estimate is **4.86**, against the true 4.83.

Without randomisation we need the **conditional exchangeability** assumption ("no unmeasured confounding"): within levels
of measured covariates X, treatment is as good as random. Then we can compare like with like (Lesson 09.03). We also
need **positivity** (every type of unit has some chance of each treatment) and **consistency / SUTVA** (one version of
the treatment, and no interference between units, Lesson 09.01, exercise 11).

## 3. Causal diagrams (DAGs)

A **directed acyclic graph** has an arrow A → B when A directly causes B. Three building blocks:

| Structure | Picture | Example | Adjust for the middle variable? |
| --- | --- | --- | --- |
| **confounder** (fork) | X ← Z → Y | motivation → tutoring, motivation → score | **yes**: it opens a non-causal path |
| **mediator** (chain) | X → M → Y | tutoring → study hours → score | **no** if you want the *total* effect |
| **collider** | X → C ← Y | talent → fame ← looks | **no**: conditioning on C *creates* a spurious association |

**Simulations** (true effect of X on Y = 2 in each case, n = 100,000):

| Structure | y ~ x | y ~ x + (middle variable) | Correct choice |
| --- | --- | --- | --- |
| confounder Z | **3.38** (biased) | **2.00** | adjust for Z |
| mediator M | **2.01** (the total effect) | 0.50 (only the direct part) | don't adjust for M |
| collider C | **2.00** | 0.49 (biased) | don't adjust for C |

"Adjust for everything" is **wrong**. Adjusting for a mediator removes the part of the effect that flows through it.
Adjusting for a collider manufactures bias.

## 4. The backdoor criterion

To estimate the effect of X on Y, find a set of variables S such that:

1. no variable in S is a descendant of X (so no mediators, and no colliders downstream of X); and
2. S blocks every **backdoor path** from X to Y, i.e. every path that starts with an arrow *into* X.

Then adjusting for S (stratification, regression, matching, weighting) identifies the causal effect. A path is blocked
by a non-collider in S, or by a collider that is *not* in S (and has no descendant in S). Conditioning on a collider
**opens** a path.

**Example DAG for tutoring:**

```text
motivation ──► tutoring ──► study_hours ──► score
     │                                       ▲
     └───────────────────────────────────────┘
tutoring ──► score            score ──► scholarship ◄── tutoring
```

The backdoor path tutoring ← motivation → score must be blocked, so adjust for motivation (if measured). study_hours is a
mediator: don't adjust for it if you want the total effect. scholarship is a collider: don't adjust, and don't select
your sample on it.

## 5. Collider bias in disguise: selection

Selecting the sample on a collider is the same as conditioning on it. **Worked example 3 (Before you start):** in the
simulated population, talent and looks are independent (r = −0.003). Among the "famous", the 7.8% whose
talent + looks > 2, the correlation is **−0.73**. Fame requires enough of one or the other, so among the famous, those low
on one must be high on the other. Real examples: hospital patients (two diseases that each cause hospitalisation look
negatively related among inpatients); "survivors" in survivorship bias (Lesson 02.01); ML datasets filtered by a model's
own predictions.

## 6. Simpson's paradox: UC Berkeley, 1973

Graduate admissions in the six largest departments (Bickel, Hammel and O'Connell, *Science*, 1975):

| | Men | Women |
| --- | --- | --- |
| overall admission rate | **44.5%** | **30.4%** |

By department:

| Dept | Men: rate (applicants) | Women: rate (applicants) |
| --- | --- | --- |
| A | 62% (825) | **82%** (108) |
| B | 63% (560) | 68% (25) |
| C | 37% (325) | 34% (593) |
| D | 33% (417) | 35% (375) |
| E | 28% (191) | 24% (393) |
| F | 6% (373) | 7% (341) |

Within most departments women were admitted at similar or *higher* rates. The overall gap appears because women applied
mostly to departments with **low** admission rates (C–F), while men applied mostly to A and B. Weighting each group's
applications by department admission rates gives an expected rate of 29.5% for women's applications against 45.1% for
men's, almost the whole gap.

**Which comparison is "right" depends on the causal question**, which is why you need a DAG. If gender → department
choice → admission, then department is a *mediator*: the overall comparison answers "what is the total effect of
gender?", including through choices, and the within-department comparison answers "do departments treat equal applicants
differently?". The data alone can't tell you which question to ask.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "Adjust for every available variable" | Adjust for confounders; not mediators, not colliders |
| Selecting the sample on an outcome-related variable | That's conditioning on a collider |
| Reading a regression coefficient as causal by default | Only under the DAG assumptions that justify it |
| Thinking the aggregated or the stratified answer is always right | It depends on the causal question |

## Summary

- Potential outcomes Y(1), Y(0); ATE = E[Y(1) − Y(0)]; we observe only one per unit.
- Naive difference = effect + selection bias; randomisation removes the bias.
- Observational studies need no unmeasured confounding, positivity and consistency.
- DAGs: adjust for confounders (forks); not for mediators (chains, for total effects); never for colliders.
- The backdoor criterion selects a valid adjustment set; Simpson's paradox needs a causal model to resolve.

## Self-test

1. Why is causal inference called a missing-data problem?
2. In X → M → Y, what does adjusting for M estimate?
3. Give a real example of collider bias from selection.
4. In the Berkeley data, is department a confounder or a mediator for the effect of gender? Why does it matter?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
