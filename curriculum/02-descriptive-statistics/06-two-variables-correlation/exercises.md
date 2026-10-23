# 02.06 Exercises — Two variables and correlation

`make lesson L=02-descriptive-statistics/06-two-variables-correlation`

## ★ Warm-up

1. **(M)** From the program × tutoring table in the lesson, compute P(Commerce and no tutoring),
   P(no tutoring | Commerce) and P(Commerce | no tutoring). Why are the last two different?
2. **(M)** For x = (1, 2, 3, 4, 5) and y = (2, 1, 4, 3, 5), compute the covariance and r by hand with a deviation table.
3. **(M)** If r(x, y) = 0.6, what are r(2x + 5, y), r(−x, y) and r(x, x)?

## ★★ Core

4. **(C)** Give an example of two variables with r close to 0 that are nonetheless strongly related. Sketch the scatter
   plot.
5. **(P)** Cross-tabulate `year` × `tutoring` with conditional proportions by year. Which year attends tutoring least?
   Draw a grouped bar chart of the proportions.
6. **(P)** Implement in `statkit/descriptive.py`: `covariance(x, y, ddof=1)`, `correlation(x, y)`, `ranks(values)`
   (ranks start at 1; ties get their average rank) and `spearman(x, y)` (Pearson on ranks). Run
   `make test M=descriptive`. All descriptive tests should now pass.
7. **(P)** Reproduce Anscombe's quartet plot from the data in `examples.py`, and add the Spearman correlation of each set
   to its title. Which sets does Spearman describe better than Pearson?
8. **(P)** Compute r(study_hours, exam_score) separately for each program. Are they similar? Plot them as three
   scatter panels with shared axes.
9. **(C)** "Students who use the library more have higher grades." Give three different explanations (one causal, one
   reverse-causal, one confounded) and describe a study that could tell them apart.

## ★★★ Challenge

10. **(M)** Prove that −1 ≤ r ≤ 1. (Hint: r is the cosine of the angle between the centred vectors, and
    |**u**·**v**| ≤ ‖**u**‖‖**v**‖ by the Cauchy–Schwarz inequality.)
11. **(P)** **Correlation by chance.** Generate a 20 × 100 matrix of independent standard normal numbers (20
    observations of 100 unrelated variables). Compute all 4,950 pairwise correlations. How many have |r| > 0.5? What's
    the largest? What does this mean for "data mining" studies that test thousands of correlations? (Module 05.08
    returns to this.)
12. **(P)** **Simpson's paradox.** Build a dataset with two groups where x and y are *negatively* correlated within each
    group but *positively* correlated when the groups are pooled. (Hint: give group B higher x and higher y overall.)
    Plot it with one colour per group, and explain in two sentences how this can happen.

Solutions: [solutions/solutions.md](solutions/solutions.md)
