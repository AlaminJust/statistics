# 02.04 Exercises — Position, shape and outliers

`make lesson L=02-descriptive-statistics/04-position-shape-outliers`

## ★ Warm-up

1. **(M)** Give the five-number summary and IQR of 3, 5, 5, 6, 8, 9, 30 (linear method). Is 30 an outlier by the
   1.5 × IQR rule?
2. **(M)** For 11, 13, 14, 16, 18, 21, 22, 25, 27, 40, compute the 10th, 25th, 50th and 75th percentiles by hand. Check
   with `np.percentile`.
3. **(M)** Test 1 has mean 500 and sd 100; test 2 has mean 20 and sd 5. Ana scored 650 on test 1 and Ben 27 on
   test 2. Who did relatively better?

## ★★ Core

4. **(P)** For each numeric column of `students.csv` (study_hours, sleep_hours, commute_minutes, exam_score), print the
   five-number summary, the IQR and the number of 1.5 × IQR outliers. Which variable has the most outliers, and why?
5. **(P)** Implement in `statkit/descriptive.py`: `percentile(values, p)` (p from 0 to 100, linear interpolation),
   `quartiles` (returns (Q1, Q2, Q3)), `iqr`, `five_number_summary`, `zscores` (sample sd), `skewness` (g₁, as in the
   lesson) and `iqr_outliers(values, k=1.5)` (returns the flagged values, sorted). Run `make test M=descriptive`.
6. **(P)** Standardise study_hours, sleep_hours and commute_minutes. Which student is the most unusual, i.e. has the
   largest |z| on any one variable? On which variable?
7. **(M)** What share of data does Chebyshev guarantee within 1.5 standard deviations of the mean? Check the actual share
   for `commute_minutes`.
8. **(P)** For `commute_minutes`, count the values flagged by (a) the 1.5 × IQR rule, (b) |z| > 3 and (c) robust
   |z| > 3. Why do they disagree so much on this variable?
9. **(C)** A dataset of house sizes includes one value of 25,000 m². List three possible explanations and what you would
   do in each case.

## ★★★ Challenge

10. **(M)** Prove that standardised values zᵢ = (xᵢ − x̄)/s always have mean 0 and standard deviation 1. (Use the
    transformation rules from 02.03.)
11. **(P)** Apply `np.log1p` to `commute_minutes`. How do the skewness and the number of 1.5 × IQR outliers change? Plot
    box plots before and after.
12. **(P)** Write `percentile_rank(values, x)` returning the percentage of values strictly below x. What is the percentile
    rank of a score of 90 in `exam_score`? Check with `scipy.stats.percentileofscore(..., kind="strict")`.

Solutions: [solutions/solutions.md](solutions/solutions.md)
