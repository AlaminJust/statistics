# Study Plan — 34 weeks, about 8 hours a week

Start date: the Monday you begin Module 00. Each week has a fixed rhythm (see [docs/study-method.md](docs/study-method.md)):

| Day | Activity | Time |
| --- | --- | --- |
| Mon | Read `lesson.md`, run `examples.py`, write `notes.md` in your own words | 2 h |
| Tue | Warm-up and core exercises by hand (paper or `notes.md`) | 1.5 h |
| Wed | Code exercises in `workspace/.../exercises.py` | 1.5 h |
| Thu | Implement the week's `statkit` functions until `make test` passes | 1.5 h |
| Fri–Sun | Challenge exercise or mini-project, weekly journal, commit + push | 1.5 h |

**Rule:** do not open `solutions/` until you have made a real attempt and committed it. Then compare, and commit
your corrections separately, so your git history shows how you improved.

---

## Module 00 — Setup and study system (week 0)

| Week | Lessons | Output you commit | Tag when done |
| --- | --- | --- | --- |
| 0 | 00.01 Environment and git · 00.02 Python for data | Working `.venv`, first journal entry, first `statkit` function | `m00-done` |

## Module 01 — Math toolkit (weeks 1–2)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 1 | 01.01 Notation and summation · 01.02 Functions, exponents and logs · 01.03 Derivatives and optimisation | Notes, exercises, `statkit/mathkit.py` part 1 | |
| 2 | 01.04 Integrals and area · 01.05 Vectors and matrices | Notes, exercises, mini-project: gradient descent from scratch | `m01-done` |

## Module 02 — Descriptive statistics (weeks 3–5)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 3 | 02.01 Data and variables · 02.02 Measures of centre | `statkit/descriptive.py`: mean, median, mode, weighted and trimmed mean | |
| 4 | 02.03 Measures of spread · 02.04 Position, shape and outliers | variance, std, IQR, percentiles, z-scores, skewness | |
| 5 | 02.05 Visualising data · 02.06 Two variables and correlation | covariance, correlation; mini-project: EDA report | `m02-done` |

## Module 03 — Probability (weeks 6–8)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 6 | 03.01 Sample spaces and axioms · 03.02 Counting | Simulations of dice, cards, birthdays | |
| 7 | 03.03 Conditional probability and independence · 03.04 Bayes' theorem | Medical-test and Naive Bayes examples | |
| 8 | 03.05 Random variables and expectation · 03.06 Joint distributions and covariance | Mini-project: Monty Hall + spam filter from scratch | `m03-done` |

## Module 04 — Distributions (weeks 9–11)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 9 | 04.01 Discrete distributions · 04.02 Continuous distributions | `statkit/distributions.py`: PMFs, PDFs, CDFs, samplers | |
| 10 | 04.03 Normal and multivariate normal · 04.04 Law of large numbers and CLT | CLT simulator | |
| 11 | 04.05 Sampling distributions (t, χ², F) · 04.06 Entropy, cross-entropy, KL | Mini-project: distribution explorer | `m04-done` |

## Module 05 — Statistical inference (weeks 12–15)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 12 | 05.01 Estimators, bias and variance · 05.02 Maximum likelihood | MLE by hand and with `scipy.optimize` | |
| 13 | 05.03 Confidence intervals · 05.04 The bootstrap | `statkit/inference.py`: intervals, bootstrap | |
| 14 | 05.05 Hypothesis testing · 05.06 Common tests (t, χ², ANOVA, non-parametric) | Tests from scratch vs SciPy | |
| 15 | 05.07 Power and effect size · 05.08 Multiple testing and model comparison | Mini-project: A/B results analysis | `m05-done` |

## Module 06 — Regression and GLMs (weeks 16–19)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 16 | 06.01 Simple linear regression · 06.02 Multiple regression in matrix form | `statkit/regression.py`: OLS | |
| 17 | 06.03 Diagnostics and assumptions | Residual plots, VIF | |
| 18 | 06.04 Logistic regression and GLMs · 06.05 Regularisation | Logistic regression by gradient descent; ridge, lasso | |
| 19 | 06.06 Splines and mixed models | Mini-project: regression modelling report | `m06-done` |

## Module 07 — Bayesian statistics (weeks 20–23)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 20 | 07.01 Bayesian thinking and conjugate priors | Beta–Binomial A/B test | |
| 21 | 07.02 MCMC and PyMC | Metropolis sampler from scratch | |
| 22 | 07.03 Hierarchical models · 07.04 Model checking and comparison | Eight Schools, radon model | |
| 23 | 07.05 Latent variables, EM, variational inference | Mini-project: Bayesian analysis report | `m07-done` |

## Module 08 — Statistical learning and ML evaluation (weeks 24–27)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 24 | 08.01 Generalisation and bias–variance | Double-descent experiment | |
| 25 | 08.02 Cross-validation and leakage · 08.03 Metrics and calibration | Leakage demo, reliability diagrams | |
| 26 | 08.04 Uncertainty and conformal prediction | Conformal prediction from scratch | |
| 27 | 08.05 PCA and high-dimensional data | Mini-project: trustworthy classifier | `m08-done` |

## Module 09 — Experiments, causal inference and research methods (weeks 28–31)

| Week | Lessons | Output you commit | Tag |
| --- | --- | --- | --- |
| 28 | 09.01 Experimental design and A/B tests | Experiment plan + peeking simulation | |
| 29 | 09.02 Potential outcomes and DAGs | DAGs for three real questions | |
| 30 | 09.03 Observational causal methods | Matching, IPW, diff-in-diff | |
| 31 | 09.04 Reproducible ML research | Seed-variance study, paper audit | `m09-done` |

## Module 10 — Capstone (weeks 32–34)

| Week | Work | Output you commit | Tag |
| --- | --- | --- | --- |
| 32 | Choose project, write the plan and hypotheses before touching results | `projects/capstone/PLAN.md` | |
| 33 | Analysis | Notebooks and code | |
| 34 | Write the 4–6 page report, self-review with the rubric | `projects/capstone/report.md` | `v1.0-course-complete` |

---

## If you fall behind

- Missing one week is normal. Do not skip lessons; move the whole plan one week later.
- If a lesson takes longer than two weeks, write down exactly what is confusing in the journal and ask for help with
  that specific question.
- Each module ends with a **checkpoint** (in its README). Only move on when you can do the checkpoint without notes.
