# 09.02 Solutions

1. ATT = (8 + 8 + 5)/3 = **7.0**; ATC = (2 + 1 + 5)/3 ≈ **2.67**; ATE = 4.83. The weaker students (who chose tutoring) gain
   more from it, so the effect differs between the treated and untreated groups. Different questions have different
   answers when effects are heterogeneous.
2. (a) Age is a **confounder**: age → exercise and age → weight. Adjust for it. (b) Job type is a **mediator**:
   education → job type → income. Don't adjust for it if you want the total effect. (c) Being in a hospital study is a
   **collider**: smoking → hospital and lung disease → hospital. Selecting on it induces bias.
3. Adjusting for a mediator removes part of the effect (tutoring → study hours → score). Adjusting for a collider creates
   a spurious association (talent → fame ← looks). Adjusting for a variable affected by the outcome (a "scholarship") also
   biases the estimate.
4. E[Y | T=1] − E[Y | T=0] = E[Y(1) | T=1] − E[Y(0) | T=0]. Add and subtract E[Y(0) | T=1]:
   = {E[Y(1) | T=1] − E[Y(0) | T=1]} + {E[Y(0) | T=1] − E[Y(0) | T=0]} = ATT + selection bias. ∎
5. y ~ x gives ≈ **0.00** (correct: no effect). y ~ x + z gives ≈ **−0.20**. Z is a collider on the path
   X ← U₁ → Z ← U₂ → Y, which is blocked *until* you condition on Z. Adjusting for a pre-treatment variable can create
   bias: "pre-treatment" isn't enough to make a variable safe.
6. Population correlation ≈ 0.00; among hospitalised patients ≈ **−0.33**. Patients without disease A are more likely to
   be there because of disease B.
7. ```python
   def ate_difference_in_means(y, t):
       y, t = np.asarray(y, float), np.asarray(t).astype(bool)
       treated, control = y[t], y[~t]
       ate = treated.mean() - control.mean()
       se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
       return ate, se
   ```
8. Without department: female coefficient **−0.61**, OR 0.54 (women's odds of admission about half of men's). With
   department dummies: **+0.10**, OR 1.11, p = 0.22: no evidence of disadvantage within departments. The first answers
   "how different are women's overall outcomes (including through department choice)?"; the second, "are equal
   applicants to the same department treated differently?".
9. One possible DAG: prior GPA (confounder) → uses planner, → grades; motivation (unmeasured) → uses planner, → grades;
   planner → study hours (mediator) → grades; grades → stays enrolled (collider, if the sample is current students only).
   Valid adjustment needs prior GPA *and* motivation. Motivation is unmeasured, so no valid adjustment set exists from
   the observed data alone. You'd need an experiment, an instrument, or a sensitivity analysis.
10. Front-door estimate: (X → M) × (M → Y | X) = 0.80 × 1.51 ≈ **1.21**; the truth is 0.8 × 1.5 = 1.2. The naive
    regression gives 2.20, confounded by U. Front-door works because M carries the whole effect of X, U doesn't affect M,
    and adjusting for X blocks M ← X ← U → Y.
11. If gender → job level → acceptance, then job level is a mediator. Controlling for it hides discrimination that
    operates by steering women to lower levels. If job level is affected by both gender and an unmeasured ability that
    also drives acceptance, it's also a collider, so conditioning on it can create spurious disparities in either
    direction. A causal model of the hiring process is needed to decide which comparison answers the fairness question.
