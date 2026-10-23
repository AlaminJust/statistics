# 02.05 — Visualising data

**Time:** about 4 hours · **Prerequisites:** 00.02, 02.01–02.04

## Why this matters

Summary numbers can hide almost anything: two datasets can share a mean, a standard deviation and even a correlation
while looking completely different (you'll see the famous Anscombe example in 02.06). **Always plot before you
summarise, and before you model.** A good chart exposes skew, outliers, groups, data errors and relationships at a
glance. A bad chart can mislead, accidentally or on purpose. Every paper you write will be judged partly by its
figures.

## Learning objectives

- Choose the right chart for the variables involved.
- Draw and read histograms (and choose bins), box, violin and strip plots, bar charts, scatter plots, ECDFs and line charts.
- Recognise the shapes symmetric, skewed, bimodal and uniform.
- Spot and avoid misleading design: truncated axes, missing labels, overplotting.
- Produce clean, labelled figures with matplotlib and seaborn.

## Before you start

Run `examples.py` and open `figures/generated/04_bar_truncated_axis.png`. Before reading on: how big is the difference
between the programs' mean scores, judging by the left chart? By the right chart?

---

## 1. Which chart? Decide from the variable types

| You want to show | Variables | Chart |
| --- | --- | --- |
| Distribution of one numeric variable | 1 numeric | histogram, box plot, ECDF |
| Counts of categories | 1 categorical | bar chart (sorted) |
| A numeric variable across groups | numeric + categorical | box / violin / strip plots side by side; overlaid ECDFs |
| Relationship between two numeric variables | 2 numeric | scatter plot (+ trend line) |
| Two categorical variables | 2 categorical | grouped or stacked bar chart; table of proportions |
| Change over time | numeric + time | line chart |
| Many numeric variables at once | several numeric | pair plot; correlation heat map (02.06) |

## 2. Histograms

A histogram splits the number line into equal **bins** and draws a bar for how many values fall in each.
Bars **touch**, because the variable is continuous (unlike a bar chart of categories).

**The bin count changes the story** (`01_histogram_bins.png`):

| Bins | What you see for study_hours |
| --- | --- |
| 5 | a rough lump, with the skew barely visible |
| 20 | the shape: a peak around 8 hours and a long right tail |
| 80 | spiky noise; each bar holds only a few students |

Start with an automatic rule and then adjust: `bins="fd"` (Freedman–Diaconis, based on the IQR) suggests **11** bins for
study hours. Try a few values. A real feature survives changes in bin width; noise doesn't.

**Reading the shape** (`02_distribution_shapes.png`):

| Shape | What it looks like | Typical cause |
| --- | --- | --- |
| Symmetric | mirror image around the centre | many small independent effects (heights, measurement error) |
| Right-skewed | long tail to the right | a floor at zero with no ceiling (income, waiting times, commute) |
| Left-skewed | long tail to the left | a ceiling (scores on an easy exam, age at retirement) |
| Bimodal | two peaks | **two different groups mixed together**: find the grouping variable |
| Uniform | flat | rounding, random IDs, simulated noise |

## 3. Comparing groups: box, violin and strip plots

`03_groups_box_violin_strip.png` shows `exam_score` by program three ways:

- **Box plot:** median, quartiles, whiskers and outliers (02.04). Compact and good for many groups, but it hides the shape
  (a bimodal group looks like a normal one).
- **Violin plot:** a smoothed outline of the shape, mirrored, with quartile lines. Shows bimodality. **Caution:** the
  smoothing spills past the real data range; the violins here extend above 100 and below 20, which are impossible or
  unobserved scores.
- **Strip plot:** every observation as a dot, "jittered" sideways so they don't overlap. Best for small groups (under
  about 100), and the most honest, because nothing is summarised away.

The three programs look very similar. Science is a little less spread out, matching its lower standard deviation (02.03).

**Rule:** when n is small, show the raw points, alone or on top of a box plot.

## 4. Bar charts for categories, and the truncated-axis trap

A bar's **length** encodes its value, so the axis **must start at zero**. In `04_bar_truncated_axis.png` the left
chart starts at 62.5. Commerce's bar looks about four times as tall as Science's, yet the means are 65.0 and 63.1, a
difference of 3%. That's the answer to "Before you start".

Bar-chart checklist:

- Start at zero. (If small differences matter, plot points or differences instead of bars.)
- Sort bars by value, unless the categories have a natural order (years, ratings).
- Label the values directly when there are few bars (`05_bar_categorical.png`).
- Horizontal bars make long category names readable.
- Avoid pie charts: people judge angles far worse than lengths. Never use 3-D bars or pies.

## 5. Scatter plots

Each point is one observation, with x on one axis and y on the other (`06_scatter.png`). Look for:

| Feature | Question |
| --- | --- |
| Direction | Does y tend to rise or fall with x? |
| Form | A straight line, a curve, clusters? |
| Strength | Do the points hug a line or form a cloud? |
| Unusual points | Any far from the pattern? |

Practical tips:

- **Overplotting.** With many points, use transparency (`alpha=0.3`), smaller markers, or a 2-D histogram (`hexbin`).
- **Trend line.** A least-squares line (`np.polyfit(x, y, 1)`) summarises direction. Here it is about
  43.1 + 2.05 × hours; Module 06 derives it.
- **Third variable.** Colour by a category: tutoring students sit slightly above the others at every level of study hours.
- Put the variable you think *explains* the other on the x-axis.

## 6. The ECDF: the best plot you've never heard of

The **empirical cumulative distribution function** plots, for every value x, the proportion of data ≤ x.

**Worked example.** Data 3, 1, 4, 1, 5. Sorted: 1, 1, 3, 4, 5. The ECDF steps up by 1/5 = 0.2 at each value:

| x | 1 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- |
| proportion ≤ x | 0.4 | 0.6 | 0.8 | 1.0 |

(It jumps by 0.4 at x = 1 because two values equal 1.)

Why it's useful (`07_ecdf.png`):

- **No bins to choose.** Every data point is shown.
- **Read percentiles directly.** Where the curve crosses 0.5 is the median; 0.25 is Q1.
- **Compare groups exactly.** If one group's curve lies to the right of another's, its values are generally larger.
  The tutoring curve sits to the right of the no-tutoring curve almost everywhere.

The ECDF reappears in Module 05 (the Kolmogorov–Smirnov test) and Module 08 (calibration).

## 7. Line charts for time

Connect points in time order. A line implies continuity, so **use lines only when the x-axis is ordered** (time, dose,
iteration), never for categories. Add a moving average to show the trend through noise (`08_line_time_series.png`).
ML training curves (loss against epoch) are line charts.

## 8. Many variables: the pair plot

`sns.pairplot` draws a scatter plot for every pair of numeric variables and a histogram for each one
(`09_pairplot.png`). It's the fastest first look at a new dataset. Here only study_hours vs exam_score shows a strong
pattern.

## 9. A checklist for every figure you produce

1. **Title that states the finding** ("More study, higher scores"), not just the topic ("Scatter plot").
2. **Axis labels with units** ("study hours per week").
3. **Honest scales:** bars start at zero; don't use dual y-axes that invite false comparisons.
4. **Readable:** font size ≥ 10 pt in the final size, no clutter, no 3-D.
5. **Colour with meaning:** one colour per category, consistent across figures, and colour-blind safe (seaborn's
   `"colorblind"` palette).
6. **Show the data** when n is small; show uncertainty when you show estimates (Module 05).
7. **Reproducible:** the figure is saved by a script, never edited by hand.

---

## matplotlib and seaborn in 10 lines

```python
fig, ax = plt.subplots(figsize=(6, 4))                 # one figure, one axes
ax.hist(df["study_hours"], bins=20, edgecolor="white")  # draw on the axes
ax.set_title("Most students study 5–12 hours a week")   # title = the finding
ax.set_xlabel("study hours per week")
ax.set_ylabel("students")
fig.tight_layout()
fig.savefig("figure.png", dpi=150)

sns.boxplot(data=df, x="program", y="exam_score", ax=ax)   # seaborn: give it the table + column names
```

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Bars not starting at zero | Start at zero, or use points |
| Histogram with default bins only | Try several bin widths |
| Line chart for categories | Bar chart or points |
| Unlabelled axes, no units | Label everything |
| 10,000 opaque points | `alpha`, smaller markers or `hexbin` |
| Trusting violin edges | Smoothing extends past the data; check the real min and max |

## Summary

- Choose the chart from the variable types (table in section 1).
- Histograms show shape; try several bin counts.
- Box plots compare groups; violins show shape; strip plots show every point.
- Bars start at zero. Scatter plots show relationships. ECDFs need no bins.
- Every figure: a finding as the title, labelled axes with units, honest scales, saved by code.

## Self-test

1. Which chart would you draw for (a) the distribution of incomes, (b) the number of students per year, (c) height vs
   weight, (d) daily website visits over a year?
2. A histogram has two clear peaks. What should you investigate?
3. Why must the y-axis of a bar chart start at zero, but not necessarily the y-axis of a scatter plot?
4. On an ECDF, how do you read the median?

Next: [exercises.md](exercises.md)
