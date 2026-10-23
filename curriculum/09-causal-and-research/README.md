# Module 09 — Experiments, causal inference and research methods

**Weeks 28–31 · about 30 hours**

Research asks "what causes what?" and "would this replicate?", and prediction answers neither. This module covers how
to design experiments, how to reason about causation when you can't experiment, and how to report ML results so they
survive peer review. Every causal method is implemented directly with NumPy, statsmodels and scikit-learn, so you can see
exactly what it does. DoWhy and EconML are good next steps, listed in the resources.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 09.01 | [Experimental design and A/B tests](01-experimental-design-ab-tests/lesson.md) | Plan first; SRM checks; peeking inflates errors; CUPED and blocking cut noise | `sample_size_two_proportions`, `srm_test`, `cuped_adjust` |
| 09.02 | [Potential outcomes and DAGs](02-potential-outcomes-and-dags/lesson.md) | Adjust for confounders, never colliders; Simpson's paradox | `ate_difference_in_means` |
| 09.03 | [Observational causal methods](03-observational-causal-methods/lesson.md) | IPW, matching, AIPW; DiD, IV, RD, each with one key assumption | `ipw_ate`, `standardized_mean_difference`, `difference_in_differences` |
| 09.04 | [Reproducible ML research](04-reproducible-ml-research/lesson.md) | Seeds, paired comparisons, ablations, IQM, pre-registration | `seed_summary`, `interquartile_mean` |

Run `make test M=causal` as you implement each function.

## Mini-project: a causal analysis report

Brief in [projects/m09-causal-analysis/README.md](../../projects/m09-causal-analysis/README.md), using
`data/job_training.csv`.

## Checkpoint (without notes)

1. Why must the randomisation unit match the analysis unit? What goes wrong otherwise?
2. Explain the peeking problem, and one valid way to monitor an experiment.
3. Draw a DAG for "does tutoring raise exam scores?" with a confounder, a mediator and a collider, and say what to adjust for.
4. State the key assumption behind IPW, difference-in-differences, instrumental variables and regression discontinuity.
5. Why is "best of 5 runs" misleading, and what should you report instead?
6. `make test M=causal` passes.

Pass all six → `git tag -a m09-done -m "Module 09 complete"`.

## Resources

[Causal Inference: What If — Hernán and Robins](https://miguelhernan.org/whatifbook) ·
[Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) ·
[Introduction to Causal Inference — Brady Neal](https://www.bradyneal.com/causal-inference-course) ·
[NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) ·
[rliable](https://github.com/google-research/rliable) ·
Optional libraries: [DoWhy](https://www.pywhy.org/dowhy/), [EconML](https://econml.azurewebsites.net/)
