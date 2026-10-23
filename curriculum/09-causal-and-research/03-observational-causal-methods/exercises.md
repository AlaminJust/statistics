# 09.03 Exercises — Observational causal methods

`make lesson L=09-causal-and-research/03-observational-causal-methods`

## ★ Warm-up

1. **(C)** State in one sentence the key assumption of IPW, DiD, IV and RD.
2. **(C)** A propensity score is 0.95. What does that mean, and why are such units a problem for estimating the ATE?
3. **(M)** Before: treated regions 50, control regions 40. After: treated 58, control 45. Compute the DiD estimate, and say
   what you'd assume to interpret it causally.

## ★★ Core

4. **(P)** Implement `ipw_ate(y, t, propensity)` (normalised weights), `standardized_mean_difference(x, t, weights=None)`
   and `difference_in_differences(pre_t, post_t, pre_c, post_c)` in `statkit/causal.py`.
5. **(P)** Change the lesson's job-training simulation so that earnings depend on prior earnings **non-linearly**
   (add 0.00003 × prior²). Compare linear regression adjustment, IPW and AIPW, with the outcome models still linear. Which
   methods remain close to +1,800?
6. **(P)** Add an **unmeasured** confounder (motivation, which raises both participation and earnings) to the simulation.
   Show that every Part 1 method is now biased. How large must motivation's effect be to make the estimated effect zero?
7. **(P)** **Violated parallel trends.** In the DiD simulation, give policy regions an extra growth of +2 per period that
   has nothing to do with the policy. What does DiD estimate now? Add a second pre-period, and show how an event-study
   plot reveals the problem.
8. **(P)** **Weak instrument.** In the IV simulation, make the instrument weak (education = 14 − 0.002 × distance + …).
   Report the first-stage F and the 2SLS estimate over 500 simulations. What happens to its bias and spread?

## ★★★ Challenge

9. **(P)** **Manipulation in RD.** Simulate students who scored 67–69 retaking the exam until they pass 70. Plot a histogram
   of scores around 70, and show how the naive RD estimate becomes biased. (This is what a McCrary density test detects.)
10. **(M)** Show that E[T Y/e(X)] = E[Y(1)] under no unmeasured confounding and positivity. (Condition on X and use
    E[T | X] = e(X).)
11. **(C)** Pick a published observational study in your field. Identify its estimand, its method, its key assumption, and
    one way that assumption could fail. Suggest a robustness check.

Solutions: [solutions/solutions.md](solutions/solutions.md)
