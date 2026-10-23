# 02.02 Exercises — Measures of centre

`make lesson L=02-descriptive-statistics/02-measures-of-center`

## ★ Warm-up

1. **(M)** Mean, median and mode of 4, 8, 6, 5, 3, 8, 9.
2. **(M)** Mean and median of 12, 7, 3, 14, 9, 10 (n is even).
3. **(C)** Which measure of centre would you report, and why? (a) house prices in a city; (b) shoe sizes, for a shop
   deciding which size to stock most; (c) exam scores in a class with a symmetric distribution; (d) favourite colour.

## ★★ Core

4. **(M)** Three groups have sizes 10, 25 and 15 and means 72, 65 and 80. Compute the overall mean. Compare it with the
   unweighted average of the three means.
5. **(M)** A price rises 10%, then rises 20%, then falls 30%. What is the average yearly change? Check your answer by
   following 100 through the three years.
6. **(M)** For 12, 15, 15, 16, 18, 19, 20, 21, 22, 95, compute the mean, median and 10% trimmed mean. Which two agree,
   and why?
7. **(P)** For each `program` in `students.csv`, compute the mean and median `study_hours`. Which program's study hours
   are most right-skewed, judging by mean − median?
8. **(P)** Implement in `statkit/descriptive.py`: `mean`, `median`, `mode` (return a **sorted list** of all values tied
   for most frequent), `weighted_mean`, `trimmed_mean(values, proportion)` (cut `int(proportion * n)` values from each
   end, like SciPy) and `geometric_mean` (raise `ValueError` if any value ≤ 0). Plain Python only. Run
   `make test M=descriptive`.
9. **(P)** Show numerically, then explain: if you add 10 to every value, what happens to the mean and median? If you
   multiply every value by 3?

## ★★★ Challenge

10. **(M)** Prove that the weighted mean of group means, with weights equal to group sizes, equals the overall mean of
    all observations. (Hint: write each group mean as a group sum divided by its size.)
11. **(P)** **Which centre is more stable?** Draw 10,000 samples of size 25 from (a) a normal distribution
    `rng.normal(50, 10, ...)`, (b) an exponential `rng.exponential(25, ...)` and (c) a heavy-tailed t distribution
    `rng.standard_t(2, ...)`. For each, compare the standard deviation of the 10,000 sample means with that of the
    10,000 sample medians. Which estimator wins in each case, and what does that suggest about when to use each?
12. **(C)** A product has 100 reviews: 50 one-star and 50 five-star. Compute the mean and median rating. Write a
    one-sentence summary that a shopper would find more useful than either.

Solutions: [solutions/solutions.md](solutions/solutions.md)
