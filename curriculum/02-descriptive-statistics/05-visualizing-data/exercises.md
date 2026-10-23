# 02.05 Exercises — Visualising data

`make lesson L=02-descriptive-statistics/05-visualizing-data`. Save every figure into your workspace folder, and
make sure each one passes the 7-point checklist in section 9 of the lesson.

## ★ Warm-up

1. **(C)** Which chart for: (a) the distribution of incomes; (b) students per year; (c) height vs weight; (d) daily
   website visits over a year; (e) exam scores of 12 students in each of 3 classes?
2. **(M)** Draw by hand (on paper) the ECDF of 2, 5, 5, 7, 10. Mark the median on it.
3. **(P)** Make a histogram of `sleep_hours` with 8, 15 and 40 bins side by side. Which bin count would you publish?

## ★★ Core

4. **(P)** Plot `commute_minutes` as (a) a histogram, (b) a box plot and (c) an ECDF in one figure with three panels.
   Which panel shows the skew most clearly? Which lets you read "70% of students commute less than ___ minutes"?
5. **(P)** Compare `study_hours` across the four `year` groups with a box plot and a strip plot on top of each other
   (`sns.boxplot` then `sns.stripplot` on the same axes). Describe what you see in two sentences.
6. **(P)** Make a correct bar chart of the share of students attending tutoring in each program (proportions, not
   counts). Then make a deliberately misleading version of the same chart, and list exactly what makes it misleading.
7. **(P)** Scatter `sleep_hours` (x) against `exam_score` (y), coloured by `tutoring`, with a trend line for each
   group. Is the sleep–score relationship strong or weak?
8. **(P)** Implement in `statkit/descriptive.py`: `ecdf(values)`, returning (sorted values, list of i/n for i = 1…n),
   and `histogram_counts(values, bins, low, high)`, returning the count in each of `bins` equal-width bins from `low` to
   `high`, with the last bin including its right edge (same as `np.histogram`). Test with `make test M=descriptive`.
9. **(P)** Simulate a bimodal variable: 150 students who study about 6 hours (sd 1.5) and 50 who study about 16 hours
   (sd 2). Plot its histogram with the mean and median marked. Explain why "the average student studies X hours" is a
   poor summary here.

## ★★★ Challenge

10. **(P)** Recreate `06_scatter.png` from scratch without looking at the code, and improve it: a title stating the
    finding, a colour-blind-safe palette, and the fitted equation in the legend.
11. **(P)** Find a chart in the news or on social media that misleads. Recreate an honest version with matplotlib
    using approximate numbers, and write three sentences on what was wrong. (Put the original screenshot in your
    workspace folder too.)
12. **(P)** Write a function `quick_eda(df, column)` that saves a 2 × 2 figure for any numeric column: histogram,
    box plot, ECDF, and a table of summary statistics drawn with `ax.table`. Use it on every numeric column of
    `students.csv`. You'll reuse it in the m02 project.

Solutions: [solutions/solutions.py](solutions/solutions.py)
