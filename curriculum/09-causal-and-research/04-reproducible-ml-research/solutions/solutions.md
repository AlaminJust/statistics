# 09.04 Solutions

1. d = 0.3/0.4 = 0.75. n per method ≈ 2(1.96 + 0.84)²/0.75² ≈ 28; the exact t-test answer is **29 seeds per method**. With
   paired runs (shared seeds and splits), far fewer may be needed if the methods' results are correlated (exercise 10).
2. Pairing removes variation that both methods share: an unusually easy test split, or a lucky data shuffle. Comparing on
   the same seeds and splits measures the difference with much less noise.
3. It could be (a) the sd across runs (how much one run varies), (b) the standard error of the mean (how precisely the
   average is known), or (c) the half-width of a 95% CI, perhaps from a single run's test-set bootstrap, ignoring training
   randomness. The reader needs to know which, and over how many runs.
4. ```python
   def seed_summary(scores, level=0.95):
       s = np.asarray(scores, float)
       m, sd = s.mean(), s.std(ddof=1)
       half = stats.t.ppf(0.5 + level / 2, len(s) - 1) * sd / np.sqrt(len(s))
       return m, sd, m - half, m + half

   def interquartile_mean(scores):
       s = np.sort(np.asarray(scores, float).ravel())
       k = len(s) // 4
       return s[k:len(s) - k].mean()
   ```
5. Expected best of k (from the 20 recorded runs): k = 1 → 0.9750; 3 → 0.9786; 5 → 0.9801; 10 → **0.9824**; 20 → 0.9852.
   The best of 10 "improves" by +0.7 points with no change at all to the method, larger than many published gains.
6. Regression on 32 runs: scaling **+0.0067** (p = 0.006); weight decay −0.0012 (p = 0.61); interaction +0.0019 (p = 0.56).
   Scaling matters; weight decay doesn't, with or without scaling. A factorial design estimates main effects and the
   interaction with every run used twice: more efficient than one-at-a-time ablations.
7. The Wilcoxon test on 30 paired differences treats them as independent, but the 10 splits within each dataset share
   that dataset's characteristics, so they're correlated. A safer approach: average the differences within each dataset,
   then compare the 3 dataset-level means (low power), or fit a mixed model with a random effect for dataset (Lesson 06.06).
8. Key items to check: error bars and their definition, the number of runs, how splits were made, the compute used,
   whether code and data are available, and whether hyperparameters were tuned fairly.
9. A typical audit finding: the original reports a single run; 10 seeds give a mean 0.5–1 point lower, with an sd similar
   to the claimed improvement.
10. Var(Ā − B̄) = Var(Ā) + Var(B̄) − 2Cov(Ā, B̄) = (σ_A² + σ_B² − 2ρσ_Aσ_B)/k. Pairing helps most when ρ is large: shared
    splits and seeds affect both methods similarly. With ρ = 0.8 and equal σ, the variance falls to 40% of the unpaired
    value.
11. See the template in `curriculum/10-capstone/README.md`. Tag it with `git tag -a capstone-plan` before any analysis.
