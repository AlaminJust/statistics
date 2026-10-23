# Module 09 — Experiments, causal inference and research methods

**Weeks 28–31 · about 32 hours** · Status: **outline** · Adds `dowhy` and `econml` to `requirements.txt`

Research asks "what causes what?" and "would this result replicate?". Prediction answers neither. This module covers
how to design experiments, how to reason about causation when you can't experiment, and how to report ML results so
that they survive peer review.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 09.01 | Experimental design and A/B tests | Design an experiment before running it | randomisation, control groups, blocking and stratification, units of randomisation, power analysis, the peeking problem, sequential testing, CUPED, factorial designs |
| 09.02 | Potential outcomes and DAGs | Reason formally about causation | potential outcomes, the average treatment effect, confounding, causal DAGs, d-separation, the backdoor criterion, colliders, Simpson's paradox (UC Berkeley 1973) |
| 09.03 | Observational causal methods | Estimate effects without randomising | regression adjustment, propensity scores, matching, inverse probability weighting, doubly robust estimators, difference-in-differences, instrumental variables, regression discontinuity, double machine learning |
| 09.04 | Reproducible ML research | Produce results that hold up | seed variance, intervals over runs, ablation studies, fair baselines and tuning budgets, pre-registration, reproducibility checklists, auditing a paper's statistics |

## statkit functions (`statkit/causal.py`)

`sample_size_two_proportions(p1, p2, alpha, power)`, `ate_difference_in_means`, `propensity_ipw_ate`,
`difference_in_differences`, `seed_variance_summary(scores)`.

## Mini-project: a causal analysis report

Using the LaLonde job-training data (downloaded into `data/raw/`), compare the naive difference in means, regression
adjustment, matching and IPW estimates of the training programme's effect on earnings, check covariate balance, and
write a report with a sensitivity-analysis section.

## Checkpoint (without notes)

1. Why must the randomisation unit match the analysis unit?
2. Draw a DAG for "does tutoring raise exam scores?" with one confounder and one collider, and say what to adjust for.
3. State the key assumption behind IPW, difference-in-differences and instrumental variables, in one sentence each.
4. Why is "best of 5 runs" misleading, and what should you report instead?
5. `make test M=causal` passes.

## Resources

[Causal Inference: What If — Hernán and Robins](https://miguelhernan.org/whatifbook) ·
[Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) ·
[Introduction to Causal Inference — Brady Neal](https://www.bradyneal.com/causal-inference-course) ·
[NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) ·
[rliable](https://github.com/google-research/rliable)
