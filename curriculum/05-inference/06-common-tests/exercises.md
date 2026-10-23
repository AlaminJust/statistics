# 05.06 Exercises — Common tests

`make lesson L=05-inference/06-common-tests`

## ★ Warm-up

1. **(C)** Which test? (a) The same users' task times on the old and new versions of an app. (b) Click / no click on
   three page designs. (c) Salaries of two independent groups with a few huge outliers. (d) Whether a die is fair.
2. **(M)** Compute the expected count for (Science, yes) in the program × tutoring table, and its contribution
   (O − E)²/E to χ².
3. **(C)** Why does pairing increase power? Explain using the sds in the lesson's paired example.

## ★★ Core

4. **(P)** Test whether `study_hours` differs between Commerce and Science with a Welch t-test. Then run a one-way ANOVA
   across all three programs, followed by Tukey's HSD (`stats.tukey_hsd`). Explain why the pairwise conclusion changes.
5. **(P)** Is scoring ≥ 70 related to tutoring? Build the 2 × 2 table and run a χ² test of independence. Compare with the
   t-test on raw scores: which uses the information better?
6. **(P)** Test whether tutoring attendance differs by `year` (χ² test). Report the statistic, df and p.
7. **(P)** Implement `welch_t_test(x, y)` in `statkit/inference.py`, returning (t, df, two-sided p). Compute t and the
   Welch–Satterthwaite df yourself, and use `scipy.stats.t.sf` only for the p-value. Check against `stats.ttest_ind`.
8. **(P)** Simulate a paired design where 25 people try two diets. Each person has their own baseline weight
   (sd 15 kg), and diet B gives 1 kg more loss on average (each diet's result has noise sd 1 kg). Show that the
   paired test usually detects
   the effect while the unpaired test usually doesn't. (Run 1,000 simulations and report how often each rejects at
   α = 0.05.)
9. **(M)** Show that for a 2 × 2 table, χ² (without continuity correction) equals z², where z is the two-proportion
   z-statistic. Check numerically on the table from exercise 5.

## ★★★ Challenge

10. **(P)** **Robustness.** Simulate two groups of 15 from heavy-tailed t₂ distributions with a true shift of 1.
    Compare the power of the Welch t-test and Mann–Whitney over 5,000 simulations. Then repeat with normal data.
11. **(P)** **Fisher's exact test.** A trial has 4 treated and 4 control patients; 3 treated and 1 control recover.
    Run Fisher's exact test, and explain why a χ² test is inappropriate here.
12. **(C)** A paper compares 5 machine-learning models on the same 10 datasets and reports 10 pairwise t-tests, all at
    α = 0.05, treating the datasets as independent samples of each model's accuracy. List three problems with this
    analysis. (Lesson 05.08 gives better methods.)

Solutions: [solutions/solutions.md](solutions/solutions.md)
