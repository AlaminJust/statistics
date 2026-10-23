# 09.01 Solutions

1. 5% → 6%: **8,158 per arm**. 30% → 33%: **3,763 per arm**. Higher baselines need fewer users for the same relative lift.
2. χ² p ≈ 10⁻³⁶: a severe sample ratio mismatch. Don't analyse the outcome. Investigate the assignment and logging
   pipeline (redirects, bot filtering, a crash in one arm, different eligibility rules), fix it, and rerun.
3. X is measured before randomisation, so it's independent of the treatment: E[θ(X − X̄)] is the same in both arms and
   cancels in the difference. If X were measured *after* treatment began, it could be affected by the treatment, and
   subtracting it would remove part of the real effect (bias), the same problem as adjusting for a mediator (Lesson 09.02).
4. Your plan should name the hypothesis, one primary metric, guardrails, the unit, the minimum effect that matters, α,
   power, the computed n, the duration in whole weeks, a stopping rule, and the analysis method.
5. See `../solutions/statkit_causal_reference.py`.
6. The A/A false-positive rate is about **5.0%**. With a true effect giving 80% fixed-horizon power, the O'Brien–Fleming
   design also has about **79%** power. Rejections happen on average at look 3.1 of 4, and the expected sample size is
   about **82%** of the fixed design. So you get the same error rates, can stop early when the effect is real, and give
   up very little power.
7. Unadjusted: effect +0.0120, SE 0.00429. Adjusted for device and pre_spend: +0.0125, SE 0.00428. Almost no change,
   because these covariates predict conversion only weakly. Adjustment helps in proportion to how much outcome variance
   the covariates explain.
8. | Method | False-positive rate (effect 0) | Power (effect 3) |
   | --- | --- | --- |
   | student-level t-test | **0.44** | 0.88 (illusory) |
   | t-test on 40 school means | 0.049 | 0.41 |
   | OLS with cluster-robust SEs | 0.060 | 0.45 |

   The student-level test's "power" is meaningless, because its false-positive rate is 44%. With only 40 clusters the
   honest power is about 0.4: you need more schools, not more students per school.
9. A 1-week test would estimate roughly +2.4% (mostly novelty); a 4-week test, about +1.5% on average, still above the
   long-run +0.5%. To detect novelty: plot the daily treatment effect over time; compare new users (who have no prior
   habit) with existing users; or run long-term holdouts.
10. Var(Y − θX) = Var(Y) − 2θCov(X, Y) + θ²Var(X). The derivative −2Cov + 2θVar = 0 gives θ = Cov/Var. Substituting:
    Var(Y) − Cov²/Var(X) = Var(Y)(1 − ρ²). ∎
11. Treated and control riders compete for the same drivers. If treated riders get better prices and grab more drivers,
    control riders get worse service, so the measured difference is exaggerated compared with launching to everyone
    (interference: the "stable unit" assumption fails). A **switchback** design randomises whole cities, or time slots
    within a city, to one pricing algorithm, so everyone in a market at a given time gets the same treatment. Then
    compare outcomes across slots, with cluster-robust analysis.
