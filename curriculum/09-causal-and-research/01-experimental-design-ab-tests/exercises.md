# 09.01 Exercises — Experimental design and A/B tests

`make lesson L=09-causal-and-research/01-experimental-design-ab-tests`

## ★ Warm-up

1. **(M)** Users per arm for 5% → 6% conversion at α = 0.05 and 80% power? For 30% → 33%?
2. **(M)** You expected 50/50 but got 52,000 vs 48,000. Run the SRM test. What do you do next?
3. **(C)** Why doesn't CUPED bias the treatment-effect estimate? What would happen if X were measured *after* the
   treatment started?

## ★★ Core

4. **(C)** Write a one-page experiment plan (in your workspace) for a real or imagined change, such as a new onboarding
   email. Include every item in the lesson's table, with numbers.
5. **(P)** Implement `sample_size_two_proportions(p1, p2, alpha=0.05, power=0.8)`, `srm_test(n_a, n_b, expected_ratio=0.5)`
   (χ² p-value) and `cuped_adjust(y, x)` in `statkit/causal.py`.
6. **(P)** **O'Brien–Fleming boundaries.** For 4 equally spaced looks, the boundaries are z = 2.024·√(4/k) at look k
   (4.05, 2.86, 2.34, 2.02). Simulate A/A tests to show the overall false-positive rate is about 5%. Then simulate a true
   lift and compare the average stopping time with a fixed-horizon test of the same power.
7. **(P)** On `data/ab_test.csv`, estimate the treatment effect on conversion with and without adjusting for device and
   pre_spend (a linear probability model with robust SEs). How much does the SE change?
8. **(P)** **Clustered randomisation.** Simulate 40 schools (20 treated) of 30 students, with a school effect sd 5,
   student noise sd 10 and a true effect of 3. Compare (a) the student-level t-test, (b) a t-test on school means and
   (c) OLS with cluster-robust SEs: false-positive rate when the effect is 0, and power when it is 3.

## ★★★ Challenge

9. **(P)** **Novelty effects.** Simulate a treatment whose effect starts at +3% and decays to +0.5% over 4 weeks. What
   would a 1-week test conclude? A 4-week test? How would you detect a novelty effect in real data?
10. **(M)** Show that Var(Y − θX) is minimised at θ = Cov(X, Y)/Var(X), and that the minimum is Var(Y)(1 − ρ²).
11. **(C)** In a two-sided marketplace (riders and drivers), why can a user-level A/B test of a new pricing algorithm give
    a biased answer? Describe a switchback or cluster design that fixes it.

Solutions: [solutions/solutions.md](solutions/solutions.md)
