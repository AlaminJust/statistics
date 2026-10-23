# 02.04 Solutions

1. n = 7, positions 0–6. Q1: h = 1.5 → between 5 and 5 = **5**. Median: h = 3 → **6**. Q3: h = 4.5 → 8 + 0.5 × 1 = **8.5**.
   Five-number summary **3, 5, 6, 8.5, 30**; IQR = **3.5**. Upper fence = 8.5 + 5.25 = 13.75 < 30, so **30 is an outlier**.
2. P10: h = 0.9 → 11 + 0.9 × 2 = **12.8**. Q1: h = 2.25 → 14 + 0.25 × 2 = **14.5**. Median: h = 4.5 → (18 + 21)/2 = **19.5**.
   Q3: h = 6.75 → 22 + 0.75 × 3 = **24.25**.
3. Ana: z = (650 − 500)/100 = **1.5**. Ben: z = (27 − 20)/5 = **1.4**. Ana did slightly better.
4. | Variable | min | Q1 | median | Q3 | max | IQR | outliers |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | study_hours | 1.8 | 6.68 | 9.6 | 13.23 | 26.3 | 6.55 | 3 |
   | sleep_hours | 4.3 | 6.28 | 7.0 | 7.6 | 10.0 | 1.33 | 1 |
   | commute_minutes | 0 | 9 | 18 | 33.25 | 108 | 24.25 | **11** |
   | exam_score | 24 | 55 | 64 | 74 | 100 | 19 | 1 |

   commute_minutes has the most, because it is strongly right-skewed: a long tail always produces many points beyond
   the upper fence. They are real values, not errors.
5. See `../solutions/statkit_descriptive_reference.py`.
6. Student 79 (Arts, year 3): commute_minutes = 108, **z = 4.14**. Their other values are ordinary (z = 0.17 and 1.25).
7. 1 − 1/1.5² = 1 − 0.444 = **at least 55.6%**. Actual for commute_minutes: **90%**. Chebyshev is a guarantee for
   the worst case, and real data are usually far better behaved.
8. IQR rule: **11**; |z| > 3: **1**; robust |z| > 3: **15**. The classic z uses a standard deviation inflated by the tail,
   so it flags very little. The robust z uses the median and MAD, which assume a *symmetric* spread, so on a skewed
   variable it flags many ordinary tail values. Outlier rules embody assumptions about shape. On skewed data, transform
   first (exercise 11) or use domain knowledge.
9. (a) Typo (2,500 m², or a unit mix-up with ft²): check the source and fix. (b) A real but unusual property, such as an
   estate or a warehouse: keep it, use robust summaries, and maybe report results with and without it. (c) A different
   population, such as commercial buildings in a residential dataset: remove it and document the rule.
10. z = (1/s)·x − x̄/s is a linear transformation a·x + b with a = 1/s and b = −x̄/s. Mean: (1/s)x̄ − x̄/s = 0.
    sd: |a|·s = (1/s)·s = 1. ∎
11. Skewness goes from 1.37 to **−0.56** (now slightly left-skewed: many short commutes, including 0, are stretched out by
    the log). 1.5 × IQR outliers fall from 11 to **3**.
12. ```python
    def percentile_rank(values, x):
        return 100 * sum(v < x for v in values) / len(values)
    ```
    A score of 90 has percentile rank **95**: 95% of students scored below 90.
