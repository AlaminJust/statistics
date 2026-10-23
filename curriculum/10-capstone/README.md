# Module 10 — Capstone

**Weeks 32–34 · about 24 hours**

One project, written up as a 4–6 page paper with code that reproduces every number. It's the piece of work you show a
supervisor, a hiring manager or an admissions committee, and it should draw on at least four modules.

## Choose one

| Project | What you do | Modules used | Data ideas |
| --- | --- | --- | --- |
| **A. Replicate and stress-test an ML result** | Re-run a published benchmark result with 10+ seeds; add intervals, ablations and a calibration check | 05, 08, 09 | a scikit-learn or PyTorch tutorial result; a small paper with public code |
| **B. A Bayesian model of a real problem** | Build a hierarchical model, run the full workflow (07.04), make a decision from the posterior | 03, 06, 07 | sports results, hospital outcomes, sales by store, election polls |
| **C. A causal question from observational data** | Draw a DAG, estimate an effect at least three ways, run a sensitivity analysis | 05, 06, 09 | public policy panels, education data, health surveys |
| **D. An uncertainty-aware ML system** | Train a classifier or regressor, calibrate it, wrap it in conformal prediction, test it under distribution shift | 04, 08 | any tabular dataset with a natural shift (time, location) |

## Process, and what to commit when

| Week | Step | Commit |
| --- | --- | --- |
| 32 | Write `projects/capstone/PLAN.md` (template below) **before looking at results** | `project(capstone): pre-register analysis plan` + `git tag -a capstone-plan` |
| 33 | Analysis, following the plan; record every deviation and its reason in `DEVIATIONS.md` | small commits for each step |
| 34 | Write `report.md` (from `templates/project-report.md`), self-review with the rubric, revise | `git tag -a v1.0-course-complete` |

The `capstone-plan` tag gives you a time-stamped pre-registration in your own git history.

## Pre-registration template (`PLAN.md`)

```markdown
# Capstone plan — <title>
Date: <YYYY-MM-DD>

## 1. Question
One sentence. Is it descriptive, predictive or causal?

## 2. Data
Source, unit of observation, sample size, inclusion/exclusion rules, known problems. What will be held out?

## 3. Hypotheses
H1 … (with the direction and the smallest effect that would matter).

## 4. Primary analysis
Model or estimator, outcome, predictors/adjustment set (with a DAG if causal), the metric, how uncertainty is
quantified (CI, bootstrap, posterior), α or decision rule.

## 5. Secondary analyses
Robustness checks, sensitivity analysis, subgroups (with a multiplicity plan).

## 6. Sample size / power, or seeds / runs
Why this amount of data or this number of runs is enough.

## 7. What would change my conclusion
Results that would count as evidence against H1.
```

## Self-review rubric

- [ ] The question and hypotheses were committed before any results
- [ ] Every number comes with an interval or a measure of spread
- [ ] Assumptions are listed, and at least one is tested
- [ ] Baselines are tuned with the same budget as your method
- [ ] Multiple comparisons are corrected or explicitly discussed
- [ ] Causal language is used only where the design supports it
- [ ] Figures pass the checklist in Lesson 02.05
- [ ] `python projects/capstone/analysis.py` reproduces everything from a clean environment
- [ ] A "limitations" section a sceptical reviewer would accept

## After the capstone

Make the repository (or the capstone folder) public on GitHub, add a short summary to the top-level README, and link to it
from your CV. Your git history now documents 34 weeks of learning, every exercise, correction and project.
