# 05.07 — Power and effect size

**Time:** about 4 hours · **Prerequisites:** 05.03, 05.05, 05.06

## Why this matters

A p-value answers "is there an effect?", and only partly. Research and product decisions need two more answers:
**how big is the effect?** (effect size) and **could this study have found it at all?** (power). Underpowered studies
are the main cause of the replication crisis: they miss real effects, and the "significant" results they do produce
*exaggerate* the truth. Every serious A/B test, clinical trial and ML benchmark comparison should be planned with a power
analysis **before** collecting data.

## Learning objectives

- Compute and interpret effect sizes: mean differences, Cohen's d, Cohen's h and odds ratios.
- Define power and the four quantities it depends on.
- Compute the sample size needed for a target power, by formula, with statsmodels and by simulation.
- Compute a minimum detectable effect for a fixed budget.
- Explain practical vs statistical significance, and the winner's curse.

## Before you start

Your team will run an A/B test of a new checkout page. Conversion is 10% today, and you hope for 11%. Guess how many
users each version needs for an 80% chance of detecting that lift: 500? 5,000? 15,000?

---

## 1. Effect sizes: report the magnitude

| Effect size | Formula | Use | Rough benchmarks |
| --- | --- | --- | --- |
| raw difference | x̄₁ − x̄₂ (in units) | always report first; it's the most interpretable | — |
| **Cohen's d** | (x̄₁ − x̄₂)/s_pooled | standardised mean difference | 0.2 small, 0.5 medium, 0.8 large |
| Cohen's h | 2 arcsin√p₁ − 2 arcsin√p₂ | difference of proportions | same benchmarks |
| odds ratio | [p₁/(1 − p₁)]/[p₂/(1 − p₂)] | binary outcomes, logistic regression | 1 = no effect |
| r, r² | correlation | association strength | 0.1, 0.3, 0.5 |

**Worked example 1: tutoring.** Difference 8.22 points, pooled sd 13.35, so **d = 0.62**: a medium-to-large effect. For
comparison, a one-sd increase in study hours (4.92 h) goes with about 10.1 points, 0.73 sd. Cohen's benchmarks are
generic; in your field, compare effects with ones people care about.

Always pair an effect size with its **confidence interval**: "tutored students scored 8.2 points higher (95% CI 4.1 to
12.3)".

## 2. Power and what drives it

**Power** = P(reject H₀ | a specific alternative is true) = 1 − β. Four quantities are linked; fix three and the fourth
is determined:

| Quantity | Effect on power |
| --- | --- |
| effect size (d) | bigger effect → more power |
| sample size (n) | more data → more power |
| significance level (α) | larger α → more power (but more false positives) |
| noise (σ) | less noise (better measurement, pairing, covariates) → more power |

**Sample size per group** for a two-sided, two-sample comparison (normal approximation):

$$n \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}.$$

For α = 0.05 and 80% power, (1.96 + 0.84)² ≈ 7.85, so **n ≈ 15.7/d²**.

**Worked example 2.**

| d | formula | exact t-test (statsmodels) |
| --- | --- | --- |
| 0.2 | 392.4 | **394** per group |
| 0.5 | 62.8 | **64** per group |
| 0.8 | 24.5 | **26** per group |

**Halving the effect size quadruples the sample size.** Small effects need large studies.

**Worked example 3: check by simulation.** 10,000 simulated studies with d = 0.5 and 64 per group reject H₀ **79.95%**
of the time (statsmodels: 80.15%). Simulation is the universal power calculator: write code that generates data the
way you expect it to look, analyse it the way you plan to, repeat, and count the rejections. It handles any design.

```python
TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.80)   # -> 63.8
```

**Worked example 4: power curves** (`power_curves.png`). At **n = 20 per group**, typical of small lab studies, power is
0.10 for d = 0.2, 0.34 for d = 0.5 and 0.69 for d = 0.8. A study that small usually misses even medium effects.

## 3. A/B tests: proportions need big samples

**Worked example 5 (Before you start).** 10% → 11% conversion, α = 0.05, power 0.8: Cohen's h = 0.033, so
**n ≈ 14,745 per arm**, about 29,500 users in total. For a 10% → 10.5% lift, **57,756 per arm**. Small lifts on low base
rates are expensive to detect, which is why big tech companies run experiments on millions of users.

**Minimum detectable effect (MDE).** If you only have 1,000 users per arm, compute the smallest effect you could detect
with 80% power. **Worked example 6:** conversion would have to rise from 10% to about **14.1%**. If the realistic lift is
1 point, the experiment is pointless at that size: don't run it, or find more traffic, or reduce noise (Module 09:
CUPED).

## 4. Statistical vs practical significance

**Worked example 7.** With 500,000 users per group and a true difference of 0.3 points on a scale with sd 15:
**p = 3 × 10⁻¹⁷**, d = **0.017**. "Highly significant" and practically meaningless. Before a study, decide on the
**smallest effect size of interest** (a 1% conversion lift, a 2-point accuracy gain), and judge results against it, not
against zero.

| Result | Interval relative to the smallest effect that matters | Conclusion |
| --- | --- | --- |
| significant, large | entirely above it | real and important |
| significant, tiny | above 0 but below it | real but unimportant |
| not significant, narrow interval around 0 | inside ± that effect | evidence of no meaningful effect |
| not significant, wide interval | includes both 0 and important effects | **inconclusive: underpowered** |

## 5. The winner's curse: low power exaggerates effects

**Worked example 8.** True d = 0.2 with n = 20 per group: power is only 9.3%. Among the studies that *do* reach
significance, the average estimated effect is **0.76, 3.8 times the truth**. Only the studies that were lucky enough to
overestimate could cross the significance bar. Publication filters for significance, so **published small-sample
effects are systematically inflated**, and replications find smaller effects. The same happens in ML when "the best of
20 configurations" is reported from a small validation set.

**Post-hoc power is meaningless.** Computing "observed power" from the effect you just estimated is a one-to-one
transformation of the p-value and adds nothing. Do power analysis *before* the study, using an effect size justified
by previous work or by what would matter.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Reporting only p-values | Report the effect size and its interval |
| Planning n by convenience | Power analysis before the study |
| "Not significant, so no effect" | Check the interval: maybe underpowered |
| Calculating observed power after the fact | Plan power in advance |
| Trusting a big effect from a tiny study | The winner's curse: expect shrinkage on replication |

## Summary

- Effect sizes: raw difference, Cohen's d = Δ/s, h for proportions, odds ratios. Always with intervals.
- Power = 1 − β; it increases with effect size, n and α, and decreases with noise.
- n per group ≈ 2(z_{α/2} + z_β)²/d² ≈ 15.7/d² for 80% power at α = 0.05.
- Proportion A/B tests on small lifts need tens of thousands of users per arm. Know your MDE.
- Statistical ≠ practical significance; low power inflates significant effects.

## Self-test

1. How many per group for 90% power to detect d = 0.4 at α = 0.05?
2. A study of 15 per group finds p = 0.3. What would you tell the authors?
3. Compute Cohen's d for groups with means 72 and 66 and a pooled sd of 12.
4. Why does a small study's significant result tend to overestimate the effect?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
