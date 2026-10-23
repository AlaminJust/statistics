# Module 05 — Statistical inference

**Weeks 12–15 · about 34 hours**

Inference answers: *what can a sample tell us about the population, and how sure can we be?* Training a model is
estimation; claiming that model A beats model B is a hypothesis test; reporting results with error bars is interval
estimation. This module separates research-grade work from guesswork.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 05.01 | [Estimators, bias and variance](01-estimators-bias-variance/lesson.md) | MSE = bias² + variance; shrinkage can win | `standard_error` |
| 05.02 | [Maximum likelihood](02-maximum-likelihood/lesson.md) | Every ML loss is a negative log-likelihood; SE from curvature | `mle_bernoulli`, `mle_poisson`, `mle_normal` |
| 05.03 | [Confidence intervals](03-confidence-intervals/lesson.md) | "95%" is a property of the procedure; use Wilson for proportions | `t_interval`, `wilson_interval` |
| 05.04 | [The bootstrap](04-bootstrap/lesson.md) | Resample your data to get SEs for anything, including ML metrics | `bootstrap_ci` |
| 05.05 | [Hypothesis testing](05-hypothesis-testing/lesson.md) | p = P(data this extreme \| H₀), not P(H₀ \| data); permutation tests | `permutation_test` |
| 05.06 | [Common tests](06-common-tests/lesson.md) | Match the test to the design: paired vs independent | `welch_t_test` |
| 05.07 | [Power and effect size](07-power-and-effect-size/lesson.md) | Plan n before collecting; low power inflates effects | `cohens_d`, `sample_size_two_means` |
| 05.08 | [Multiple testing and comparing models](08-multiple-testing-and-model-comparison/lesson.md) | Correct for many tests; compare models with paired methods | `bonferroni`, `benjamini_hochberg` |

Run `make test M=inference` as you implement each function.

## Mini-project: analyse an A/B test

Brief in [projects/m05-ab-test/README.md](../../projects/m05-ab-test/README.md), using `data/ab_test.csv`.

## Checkpoint (without notes)

1. Derive the MLE of a Poisson rate λ and its standard error.
2. Correct this sentence: "There is a 95% probability that μ lies in [2.1, 3.4]."
3. Explain why p = 0.04 does not mean a 96% chance the effect is real.
4. When must you use a paired test? Why is it more powerful?
5. How many users per arm does an A/B test need to detect 10% → 11% conversion with 80% power?
6. You test 20 features at α = 0.05 and none truly matters. How many "significant" results do you expect? What do you do?
7. `make test M=inference` passes.

Pass all seven → `git tag -a m05-done -m "Module 05 complete"`.

## Resources

[MIT 18.650 Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/) ·
[OpenIntro Statistics](https://www.openintro.org/book/os/) chapters 5–7 ·
[Think Stats](https://allendowney.github.io/ThinkStats/) ·
Wasserstein and Lazar (2016), [The ASA statement on p-values](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
