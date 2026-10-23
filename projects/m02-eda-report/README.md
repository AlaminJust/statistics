# Mini-project m02 — Exploratory data analysis report

**Module 02 · week 5 · about 6 hours** · Branch: `git switch -c project/m02-eda-report`

## Goal

Write the kind of exploratory data analysis (EDA) a research team does before any modelling, on a real medical
dataset. Use your own `statkit.descriptive` functions for the numbers, and matplotlib or seaborn for the figures.

## Data

The diabetes dataset that ships with scikit-learn (no download needed):

```python
from sklearn.datasets import load_diabetes
df = load_diabetes(scaled=False, as_frame=True).frame     # 442 patients, 11 columns
```

| Column | Meaning |
| --- | --- |
| age | age in years |
| sex | coded 1 or 2 (the source doesn't say which is which, so treat it as a nominal label) |
| bmi | body-mass index |
| bp | average blood pressure |
| s1–s6 | six blood serum measurements (s1 total cholesterol, s2 LDL, s3 HDL, s4 total/HDL ratio, s5 log of triglycerides, s6 blood sugar) |
| target | a measure of disease progression one year after the baseline measurements |

Source: Efron, Hastie, Johnstone and Tibshirani (2004), "Least Angle Regression", *Annals of Statistics*. The
[scikit-learn documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) describes it.

## Questions your report must answer

1. **Data overview.** Rows, columns, variable types (02.01). Any missing values or impossible values?
2. **Each variable on its own.** For every column: centre, spread and shape, using the right measures for its shape
   (02.02–02.04). Present this as one summary table plus a grid of histograms. Point out any skewed variables and any
   outliers by the 1.5 × IQR rule, and say what you'd do about them.
3. **The outcome.** Describe `target` in detail: histogram, box plot, ECDF, five-number summary.
4. **Relationships with the outcome.** Correlation of every variable with `target` (Pearson and Spearman), shown as a
   sorted bar chart. Scatter plots of the three strongest. Does any relationship look curved?
5. **Relationships among predictors.** A correlation heat map. Which pairs are strongly correlated? Why might that matter
   for a regression model later (Module 06)?
6. **Groups.** Compare `target` between the two `sex` groups with box plots and a summary table.
7. **Limitations.** What can't this analysis tell you? (Think about correlation vs causation, and who is in the sample.)

## Deliverables

```text
projects/m02-eda-report/
├── analysis.py     ← produces every number and figure in the report
├── report.md       ← from templates/project-report.md, 2–4 pages with figures
└── figures/        ← the PNGs the report cites
```

## What "done" looks like

- [ ] `python projects/m02-eda-report/analysis.py` runs from the project root and recreates every figure
- [ ] Every number in the report comes from code, not typed by hand
- [ ] Every figure passes the checklist in Lesson 02.05, section 9
- [ ] The report separates *description* ("BMI is correlated with progression, r = …") from *causal claims*, and makes
      no causal claims
- [ ] Merged into `main` with `git merge --no-ff project/m02-eda-report` and tagged `m02-done` once the checkpoint passes
