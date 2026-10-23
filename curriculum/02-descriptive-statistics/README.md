# Module 02 — Descriptive statistics

**Weeks 3–5 · about 24 hours**

Before any model, test or neural network, you must *look at* and *summarise* your data. This module teaches how:
what kinds of variables exist, how to describe a centre, a spread and a shape, how to spot outliers, how to draw the
right chart, and how to measure the relationship between two variables. In ML this is exploratory data analysis
(EDA), the step that catches most bugs and data problems before they reach a model.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 02.01 | [Data and variables](01-data-and-variables/lesson.md) | What kind of variable decides which summary and which chart | `frequency_table`, `relative_frequencies` |
| 02.02 | [Measures of centre](02-measures-of-center/lesson.md) | Mean vs median: sensitivity to outliers | `mean`, `median`, `mode`, `weighted_mean`, `trimmed_mean`, `geometric_mean` |
| 02.03 | [Measures of spread](03-measures-of-spread/lesson.md) | Variance, standard deviation and why we divide by n − 1 | `data_range`, `variance`, `std`, `median_absolute_deviation`, `coefficient_of_variation` |
| 02.04 | [Position, shape and outliers](04-position-shape-outliers/lesson.md) | Percentiles, z-scores, skewness, the 1.5 × IQR rule | `percentile`, `quartiles`, `iqr`, `five_number_summary`, `zscores`, `skewness`, `iqr_outliers` |
| 02.05 | [Visualising data](05-visualizing-data/lesson.md) | The right chart for each variable type; honest charts | `ecdf`, `histogram_counts` |
| 02.06 | [Two variables and correlation](06-two-variables-correlation/lesson.md) | Covariance, Pearson and Spearman correlation; correlation ≠ causation | `covariance`, `correlation`, `ranks`, `spearman` |

Run `make test M=descriptive` as you implement each function.

## Mini-project: EDA report

Brief in [projects/m02-eda-report/README.md](../../projects/m02-eda-report/README.md): a complete exploratory analysis
of the diabetes dataset that ships with scikit-learn, written up as a short report.

## Checkpoint (without notes)

1. For the data 3, 5, 5, 6, 8, 9, 30, compute the mean, median, sample standard deviation, Q1, Q3 and IQR, and say
   whether 30 is an outlier by the 1.5 × IQR rule.
2. Explain why sample variance divides by n − 1.
3. Name the right chart for: one categorical variable; one numeric variable; two numeric variables; a numeric
   variable across groups.
4. Explain what r = 0.73 between study hours and exam score does and does not tell you.
5. `make test M=descriptive` passes.

Pass all five → `git tag -a m02-done -m "Module 02 complete"`.
