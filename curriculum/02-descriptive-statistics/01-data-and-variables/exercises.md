# 02.01 Exercises — Data and variables

`make lesson L=02-descriptive-statistics/01-data-and-variables`

## ★ Warm-up

1. **(C)** A study measures 300 patients' blood pressure to learn about all adults with diabetes. Name the population,
   the sample, one parameter and one statistic.
2. **(C)** Classify each variable (nominal / ordinal / discrete / continuous) and its level of measurement: shoe size;
   eye colour; Olympic medal; temperature in °F; number of emails per day; time to finish a marathon; postcode;
   pain rating 0–10.
3. **(P)** Print the frequency and relative-frequency tables of `tutoring` in `students.csv`.

## ★★ Core

4. **(C)** For each, say whether it's an observational study or an experiment, and whether it could show causation:
   (a) comparing exam scores of students who chose online vs in-person classes; (b) randomly assigning half of a class
   to use flashcards; (c) an A/B test that shows each website visitor one of two button colours at random.
5. **(C)** Name the bias: (a) a gym surveys its members about how healthy they feel; (b) a product's online reviews are
   mostly 1 or 5 stars; (c) studying WWII planes that returned to find where to add armour.
6. **(C)** Give a variable in `students.csv` that could confound the tutoring–score association, and explain the
   mechanism in two sentences.
7. **(P)** Build a table of `program` with frequency, relative frequency and percent, sorted alphabetically. Add a row
   for the total.
8. **(P)** Build a frequency table of `year` with a cumulative relative-frequency column. What share of students is in
   year 3 or above?
9. **(P)** Implement `frequency_table(values)` (returns a dict category → count) and `relative_frequencies(values)`
   (dict category → proportion) in `statkit/descriptive.py`, using loops and a dict. Run `make test M=descriptive`
   (only these two tests will pass for now).

## ★★★ Challenge

10. **(P)** Treat `students.csv` as the population. Draw 1,000 random samples of size 10, 40 and 160 and compute each
    sample's mean `exam_score`. How does the spread of the sample means change with sample size? Plot the three
    histograms on one figure.
11. **(P)** Create the untidy table `{"student": [1, 2, 3], "math_2024": [60, 70, 80], "math_2025": [65, 72, 90]}`
    and turn it into a tidy table with columns `student`, `year`, `score` using `melt`.
12. **(P)** Change the non-response simulation in `examples.py` so the reply probability depends on scores more steeply
    (divide by 4 instead of 8). What happens to the bias? Then make it not depend on scores at all. What happens now?

Solutions: [solutions/solutions.py](solutions/solutions.py) and [solutions/solutions.md](solutions/solutions.md)
