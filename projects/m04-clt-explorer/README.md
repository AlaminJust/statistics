# Mini-project m04 — Distribution explorer and CLT simulator

**Module 04 · week 11 · about 5 hours** · Branch: `git switch -c project/m04-clt-explorer`

## Goal

Build a small command-line tool that shows how sample means behave for any distribution, then use it to answer the
question every analyst faces: **"is my sample big enough for a normal-based interval?"**

## Part A — the tool (2.5 hours)

`projects/m04-clt-explorer/clt.py` should run as:

```bash
python projects/m04-clt-explorer/clt.py --dist exponential --n 5 30 100 --reps 20000 --seed 1
```

For each n it must:

1. draw `reps` samples of size n from the chosen distribution (support at least: `normal`, `uniform`, `exponential`,
   `poisson`, `lognormal`, `cauchy`, `bernoulli`);
2. compute the sample means and standardise them with the *true* mean and sd: z = (x̄ − μ)/(σ/√n);
3. print: the sd of the means vs σ/√n, the skewness of the means, P(z < −1.96) and P(z > 1.96);
4. save one figure with a histogram of the means for every n, overlaid with the CLT normal density;
5. for `cauchy`, print a clear warning that the mean and variance don't exist, then plot the running mean instead.

Use `scipy.stats` for the true means and variances (`stats.expon(scale=...).mean()`, etc.).

## Part B — the analysis (2.5 hours)

Write `report.md` answering, with figures from your tool:

1. For each distribution, the smallest n (from 5, 10, 20, 30, 50, 100, 300, 1000) at which **both** tail probabilities
   lie between 0.020 and 0.030. Present this as one table.
2. How does that n relate to the skewness of the original distribution? Plot "required n" against skewness.
3. For a Bernoulli(p) with p = 0.5, 0.1 and 0.01, how does the required n depend on p? Compare with the rule of thumb
   np ≥ 10.
4. A colleague says "n = 30 is always enough". Write a short, polite reply with your evidence.

## What "done" looks like

- [ ] `clt.py --help` explains every option
- [ ] Results are reproducible with `--seed`
- [ ] The table in question 1 covers all seven distributions (Cauchy: "never")
- [ ] Merged into `main`; tag `m04-done` after the checkpoint
