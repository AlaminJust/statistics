# Projects

One mini-project per module plus the capstone. Each project lives in its own folder and is built on its own branch
(see [docs/git-workflow.md](../docs/git-workflow.md#branches-only-for-projects)).

| Folder | Module | Brief |
| --- | --- | --- |
| [m01-gradient-descent](m01-gradient-descent/README.md) | 01 Math toolkit | Fit a line with gradient descent written from scratch |
| [m02-eda-report](m02-eda-report/README.md) | 02 Descriptive statistics | Exploratory data analysis report on a real dataset |
| [m03-spam-filter](m03-spam-filter/README.md) | 03 Probability | Famous problems by simulation; Naive Bayes spam filter from scratch |
| [m04-clt-explorer](m04-clt-explorer/README.md) | 04 Distributions | A CLT simulator and "how big must n be?" study |
| [m05-ab-test](m05-ab-test/README.md) | 05 Inference | Analyse an A/B test and write a decision memo |
| [m06-regression-report](m06-regression-report/README.md) | 06 Regression | Recover a known truth; explain and predict disease progression |
| [m07-bayesian-report](m07-bayesian-report/README.md) | 07 Bayesian | Bayesian A/B test; hierarchical hospital league table |
| [m08-trustworthy-classifier](m08-trustworthy-classifier/README.md) | 08 Statistical learning | Nested CV, calibration, cost threshold, conformal sets, model card |
| m09 | | Brief added with Module 09 |
| capstone | 10 | See `curriculum/10-capstone/README.md` |

Every project folder should end up with:

```text
mNN-name/
├── README.md        ← the brief (given) + your notes on how to run it
├── analysis.py      ← your code (or analysis.ipynb)
├── report.md        ← your write-up, from templates/project-report.md
└── figures/         ← figures your code saves (commit the final ones you cite in the report)
```
