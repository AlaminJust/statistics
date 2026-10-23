# 02.01 — Data and variables

**Time:** about 3 hours · **Prerequisites:** Modules 00–01

## Why this matters

Every statistical decision starts with two questions: **what kind of variable is this?** and **where did the data
come from?** The answers decide which summaries make sense (you can't average "Science" and "Arts"), which chart to
draw, which model to use (regression for numbers, classification for categories), and whether any conclusion can be
trusted at all. A model trained on a biased sample gives confident wrong answers, and ML is full of such cases.

## Learning objectives

- Distinguish population and sample, parameter and statistic.
- Classify variables as categorical (nominal, ordinal) or numeric (discrete, continuous).
- Explain the four levels of measurement and what operations each allows.
- Tell observational studies from experiments and explain why only experiments easily show causation.
- Name common sampling methods and biases.
- Build frequency and relative-frequency tables.

## Before you start

A survey about smartphone use is posted on a tech website and gets 50,000 replies. Is that better than 1,000 people
chosen at random? Write your answer and reason, then revisit it in section 5.

---

## 1. Population, sample, parameter, statistic

| Term | Meaning | Example |
| --- | --- | --- |
| **Population** | every unit you want to draw conclusions about | all university students in the country |
| **Sample** | the units you actually measured | 200 students in `students.csv` |
| **Parameter** | a number describing the population (usually unknown) | the true average exam score, μ |
| **Statistic** | a number computed from the sample | the sample mean x̄ = 63.93 |
| **Unit / observation** | one thing measured; one row of data | one student |

The central problem of statistics: **we know the statistic, we want the parameter.** Different samples give different
statistics. In `examples.py`, treating our 200 students as a population, one random sample of 20 has mean 69.7 while
the "population" mean is 63.93. Across 1,000 random samples of 20, the means range from 54.9 to 72.8. Measuring how far a
statistic is likely to be from the parameter is the subject of Module 05.

## 2. Types of variables

```text
variable
├── categorical (qualitative): values are labels
│   ├── nominal: no natural order            program: Science / Arts / Commerce
│   │   └── binary: exactly two categories   tutoring: yes / no
│   └── ordinal: ordered labels              satisfaction: low / medium / high; year: 1st–4th
└── numeric (quantitative): values are numbers with units
    ├── discrete: countable values           commute_minutes (whole minutes), number of children
    └── continuous: any value in a range     study_hours, sleep_hours, height
```

**Worked example 1: classify the columns of `students.csv`.**

| Column | Type | Why |
| --- | --- | --- |
| student_id | identifier (not a variable to analyse) | numbers, but averaging IDs is meaningless |
| program | categorical, nominal | labels with no order |
| year | categorical ordinal, or discrete numeric | 1st < 2nd < 3rd < 4th; treat as ordinal for counts, numeric if "one more year" has a constant meaning |
| study_hours | numeric, continuous | measured to 0.1 hour |
| sleep_hours | numeric, continuous | |
| commute_minutes | numeric, discrete (recorded); continuous in reality | time is continuous, but it was rounded to whole minutes |
| tutoring | categorical, binary | yes / no |
| exam_score | numeric, discrete | whole points 0–100; often treated as continuous because it has many values |

Two lessons from the table: **numbers aren't always numeric variables** (IDs, postcodes, phone numbers), and the same
variable can reasonably be treated two ways. Say which you chose and why.

## 3. Levels of measurement

| Level | What you can do | Example | Meaningful summaries |
| --- | --- | --- | --- |
| Nominal | = ≠ | program, blood type | counts, proportions, mode |
| Ordinal | + order < > | satisfaction rating, year | + median, percentiles |
| Interval | + differences | temperature in °C, calendar year | + mean, standard deviation |
| Ratio | + true zero, ratios | study hours, income, kelvin | + "twice as much", geometric mean |

Test for ratio: does "twice as much" make sense? 20 hours of study is twice 10 hours (ratio). 20 °C is **not** twice as
hot as 10 °C, because 0 °C isn't "no temperature" (interval).

Rating scales (1–5 stars) are ordinal. People often average them anyway; that assumes the gap between 1 and 2 stars
equals the gap between 4 and 5. Be aware you're making that assumption.

## 4. Where data come from: observational studies vs experiments

| | Observational study | Randomised experiment |
| --- | --- | --- |
| Who decides the "treatment"? | the units themselves (students choose tutoring) | the researcher, at random |
| Can show association? | yes | yes |
| Can show causation? | only with strong extra assumptions (Module 09) | yes, if done properly |

In `students.csv`, students who attend tutoring score about 8 points higher on average (69.7 vs 61.5). That's
observational: maybe tutoring helps, or maybe motivated students both choose tutoring *and* study more. A variable
that affects both the "treatment" and the outcome is a **confounder**. Randomly assigning tutoring would break that link.

## 5. Sampling methods and bias

| Method | How | Strength / weakness |
| --- | --- | --- |
| Simple random sample | every unit equally likely, chosen by chance | the gold standard; needs a complete list of units |
| Stratified | split into groups (strata), random sample from each | guarantees each group is represented |
| Cluster | randomly pick whole groups (e.g. classes), measure everyone in them | cheap; less precise if clusters differ |
| Systematic | every k-th unit from a list | easy; risky if the list has a pattern |
| Convenience | whoever is easy to reach | cheap; usually biased |

| Bias | What happens | Example |
| --- | --- | --- |
| Selection bias | some units are more likely to be included | online survey reaches only internet users |
| Non-response bias | the people who reply differ from those who don't | only very happy or very angry customers fill in reviews |
| Survivorship bias | you only see the "survivors" | studying only successful startups to find what makes success |
| Measurement bias | the measuring tool is off | self-reported study hours are usually exaggerated |

`examples.py` simulates non-response bias: if students with higher scores are more likely to answer a survey about
their scores, the 107 who reply average **71.4** points, while the true mean is **63.9**. A random sample of the
same size averages 63.8.

Answer to "Before you start": the 50,000 web replies are a **self-selected convenience sample** of people who visit a
tech site and choose to answer. They likely use phones more than average. A larger biased sample gives a more
*precise* wrong answer. **Sample size reduces random error, not bias.** 1,000 random people are better.

ML version: a face-recognition model trained mostly on one group of people performs worse on others. That's
selection bias in the training data, and no amount of extra data from the same group fixes it.

## 6. Tidy data

Analysis is easiest when data are **tidy**:

1. each variable is a column;
2. each observation is a row;
3. each value is a cell.

`students.csv` is tidy. A spreadsheet with one column per exam date ("score_jan", "score_feb", …) is not: "date" is a
variable spread across column names. pandas' `melt` converts it (exercise 11).

## 7. Frequency tables: summarising a categorical variable

**Worked example 2: program.** Count each category, then divide by n = 200.

| program | frequency | relative frequency | percent |
| --- | --- | --- | --- |
| Science | 79 | 79 / 200 = 0.395 | 39.5% |
| Arts | 72 | 0.360 | 36.0% |
| Commerce | 49 | 0.245 | 24.5% |
| **total** | **200** | **1.000** | **100%** |

Relative frequencies always sum to 1. For an ordinal variable, also add a **cumulative** column. For `year`:

| year | frequency | relative | cumulative relative |
| --- | --- | --- | --- |
| 1 | 55 | 0.275 | 0.275 |
| 2 | 42 | 0.210 | 0.485 |
| 3 | 44 | 0.220 | 0.705 |
| 4 | 59 | 0.295 | 1.000 |

The cumulative column says, for example, 48.5% of students are in year 2 or below. The only "average" for a nominal
variable is the **mode**, the most common category (Science).

```python
df["program"].value_counts()                 # frequencies
df["program"].value_counts(normalize=True)   # relative frequencies
```

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Averaging codes (e.g. program coded 1, 2, 3) | Use counts and proportions for nominal data |
| "Big sample, so it's unbiased" | Size cuts random error, not bias |
| Concluding causation from observational data | Say "associated with" unless the treatment was randomised |
| Treating IDs or postcodes as numeric | They're labels |

## Summary

- Population/parameter (unknown truth) vs sample/statistic (what we compute).
- Categorical (nominal, ordinal, binary) vs numeric (discrete, continuous). The type decides the summary and the chart.
- Levels: nominal < ordinal < interval < ratio.
- Experiments randomise and can show causation; observational studies show association.
- Random sampling avoids bias; larger samples only reduce random error.

## Self-test

1. Give one parameter and one statistic for a study of 500 voters.
2. Classify: shoe size, eye colour, Olympic medal (gold/silver/bronze), temperature in °F, number of emails per day.
3. Why is a convenience sample of 10,000 not automatically better than a random sample of 500?
4. What is a confounder? Give an example from `students.csv`.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
