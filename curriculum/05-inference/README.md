# Module 05 — Statistical inference

**Weeks 12–15 · about 32 hours** · Status: **outline**

Inference answers: *what can a sample tell us about the population, and how sure can we be?* Training a model is
estimation. Claiming that model A beats model B is a hypothesis test. Reporting results with error bars is interval
estimation. This is the module that separates research-grade work from guesswork.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 05.01 | Estimators, bias and variance | Judge an estimator | bias, variance, MSE = bias² + variance, consistency, standard error, shrinkage |
| 05.02 | Maximum likelihood | Derive and compute MLEs | likelihood and log-likelihood, MLE for Bernoulli, Poisson and normal, numerical MLE with `scipy.optimize`, Fisher information, NLL as an ML loss |
| 05.03 | Confidence intervals | Build and correctly interpret intervals | z and t intervals for means, intervals for proportions (Wald vs Wilson), what "95% confidence" means, coverage simulation |
| 05.04 | The bootstrap | Get intervals for any statistic | resampling with replacement, percentile and BCa intervals, intervals for medians, correlations, F1 and AUC |
| 05.05 | Hypothesis testing | Run and interpret a test correctly | null and alternative, test statistics, p-values (and what they are not), α, Type I and II errors, permutation tests |
| 05.06 | Common tests | Pick the right test | one- and two-sample t-tests, paired tests, χ² tests of independence, one-way ANOVA, Mann–Whitney, Wilcoxon, checking assumptions |
| 05.07 | Power and effect size | Plan sample sizes and report magnitudes | power, minimum detectable effect, sample-size calculation, Cohen's d, odds ratios, practical vs statistical significance |
| 05.08 | Multiple testing and comparing ML models | Avoid false discoveries | family-wise error, Bonferroni, false discovery rate, Benjamini–Hochberg, p-hacking, McNemar's test, comparing models across folds |

## statkit functions (`statkit/inference.py`)

`standard_error`, `mle_bernoulli`, `mle_normal`, `t_interval`, `wilson_interval`, `bootstrap_ci(statistic, data,
n_boot, rng)`, `permutation_test(x, y, statistic, n_perm, rng)`, `welch_t_test`, `cohens_d`, `bonferroni`,
`benjamini_hochberg`.

## Mini-project: analyse an A/B test

Given a simulated A/B test of two website versions (conversion and revenue per user), produce a decision memo:
conversion rates with Wilson intervals, a test of the difference, a bootstrap interval for the revenue difference, the
power the test had, and a recommendation that separates statistical from practical significance.

## Checkpoint (without notes)

1. Derive the MLE of a Poisson rate λ.
2. Correct this sentence: "There is a 95% probability that μ lies in [2.1, 3.4]."
3. Explain why p = 0.04 does not mean a 96% chance the effect is real.
4. You test 20 features at α = 0.05 and none truly matter. How many "significant" results do you expect? What do you do?
5. `make test M=inference` passes.

## Resources

[MIT 18.650 Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/) ·
[OpenIntro Statistics](https://www.openintro.org/book/os/) chapters 5–7 ·
[Think Stats](https://allendowney.github.io/ThinkStats/) chapters on estimation and testing
