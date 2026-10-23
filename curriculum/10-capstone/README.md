# Module 10 — Capstone

**Weeks 32–34 · about 24 hours**

One project, written up as a 4–6 page paper with a public notebook or script. It is the piece of work you show to a
supervisor, a hiring manager or an admissions committee. It must draw on at least four modules.

## Choose one

| Project | What you do | Modules used |
| --- | --- | --- |
| A. Replicate and stress-test an ML result | Re-run a published benchmark result with 10+ seeds, add intervals, ablations and a calibration check | 05, 08, 09 |
| B. Bayesian model of a real problem | Build a hierarchical model (sports ratings, clinic outcomes, sales by region), check it, and make a decision from the posterior | 03, 06, 07 |
| C. Causal question from observational data | Draw a DAG, estimate an effect three ways, run a sensitivity analysis | 05, 06, 09 |
| D. Uncertainty-aware ML system | Train a classifier, calibrate it, wrap it in conformal prediction, and test it under distribution shift | 04, 08 |

## Process (and what to commit when)

| Week | Step | Commit |
| --- | --- | --- |
| 32 | Write `projects/capstone/PLAN.md`: question, hypotheses, data, methods, analysis plan, **before looking at results** | `project(capstone): pre-register analysis plan` + tag `capstone-plan` |
| 33 | Analysis, following the plan; any change to the plan is written down with its reason | small commits per step |
| 34 | Write `report.md` (from `templates/project-report.md`), self-review with the rubric, then revise | tag `v1.0-course-complete` |

Tagging the plan before the analysis gives you a time-stamped pre-registration in your own git history.

## Self-review rubric

- [ ] The question and hypotheses are stated before any results
- [ ] Every number comes with an interval or a measure of spread
- [ ] Assumptions are listed and at least one is tested
- [ ] Baselines are tuned with the same budget as your method
- [ ] Multiple comparisons are corrected or explicitly discussed
- [ ] Causal language is used only where the design supports it
- [ ] Code and data run from a clean environment with one command
- [ ] Figures pass the checklist in Lesson 02.05
