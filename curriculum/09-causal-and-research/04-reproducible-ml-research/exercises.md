# 09.04 Exercises — Reproducible ML research

`make lesson L=09-causal-and-research/04-reproducible-ml-research`

## ★ Warm-up

1. **(M)** Your method beats a baseline by 0.3 points on average, and the seed sd is 0.4 points for both. With independent
   (unpaired) runs, how many seeds per method give 80% power at α = 0.05?
2. **(C)** Why should both methods use the same seeds and data splits? What does pairing remove?
3. **(C)** A table says "accuracy 91.2 ± 0.4". List three different things "± 0.4" could mean, and say which the reader
   needs to know.

## ★★ Core

4. **(P)** Implement `seed_summary(scores, level=0.95)`, returning (mean, sd, CI low, CI high) using a t-interval, and
   `interquartile_mean(scores)` in `statkit/causal.py`.
5. **(P)** **The best-of-k effect.** For the lesson's baseline network, simulate "best of k seeds" for k = 1, 3, 5, 10 and 20
   by resampling from the 20 recorded accuracies. Plot the expected best-of-k score against k. How much "improvement"
   does choosing the best of 10 produce, with no change in method?
6. **(P)** Run a 2 × 2 factorial ablation (weight decay on/off × input scaling on/off) with 8 seeds per cell. Estimate both
   main effects and their interaction with a regression on the 32 runs. Is there an interaction?
7. **(P)** Compare two methods on the three lesson datasets with **paired** differences over 10 splits per dataset.
   Combine the evidence with a Wilcoxon signed-rank test on all 30 paired differences. Explain why this test is not quite
   valid either (hint: are the 30 differences independent?).
8. **(C)** Go through the [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) for one of your own
   mini-projects (m05–m08). Write the answers in your journal.

## ★★★ Challenge

9. **(P)** **A reproducibility audit.** Pick any small published result you can re-run (a scikit-learn example, a blog
   post, a Kaggle notebook). Reproduce it with 10 seeds, report mean ± CI, and compare with the original claim. Write a
   one-page audit.
10. **(M)** Show that pairing reduces the variance of an estimated difference: Var(Ā − B̄) = (σ_A² + σ_B² − 2ρσ_Aσ_B)/k,
    where ρ is the correlation between the methods' results on the same seed or split. When does pairing help most?
11. **(C)** Write a short pre-registration (one page) for your capstone project, following the template in
    `curriculum/10-capstone/README.md`. Commit and tag it before you start the analysis.

Solutions: [solutions/solutions.md](solutions/solutions.md)
