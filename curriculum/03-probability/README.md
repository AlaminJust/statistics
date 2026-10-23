# Module 03 — Probability

**Weeks 6–8 · about 24 hours** · Status: **outline**. Full lessons (lesson.md, examples.py, exercises, solutions)
are written in the same format as Modules 00–02.

Probability is the language of uncertainty: every model in ML outputs probabilities or is trained with a probabilistic
loss, and every statistical test is a probability statement. This module builds probability from scratch, always
checking the maths against simulation.

## Lessons

| # | Lesson | You will be able to | Key topics |
| --- | --- | --- | --- |
| 03.01 | Sample spaces and axioms | Model a random experiment and compute basic probabilities | outcomes, events, set operations (∪ ∩ complement), the three axioms, the addition rule, equally likely outcomes, simulation as a check |
| 03.02 | Counting | Count outcomes without listing them | multiplication rule, permutations, combinations, n choose k, the birthday problem, sampling with and without replacement |
| 03.03 | Conditional probability and independence | Update probabilities when information arrives | P(A \| B), the multiplication rule, independence vs mutual exclusivity, the law of total probability, tree diagrams, conditional independence |
| 03.04 | Bayes' theorem | Reverse a conditional probability | prior, likelihood, posterior, the base-rate fallacy, medical tests, odds form of Bayes, Naive Bayes classification |
| 03.05 | Random variables and expectation | Describe a random quantity and its average and spread | discrete random variables, PMFs, CDFs, E[X], linearity of expectation, Var(X), E[g(X)], the indicator trick |
| 03.06 | Joint distributions and covariance | Work with two random variables at once | joint and marginal PMFs, conditional distributions, independence of random variables, Cov(X, Y), Var(X + Y), correlation of random variables |

## statkit functions (`statkit/probability.py`)

`n_choose_k`, `permutations`, `simulate_probability(event, trials, rng)`, `bayes_posterior(prior, sensitivity,
specificity)`, `expectation(values, probs)`, `variance_rv(values, probs)`, `joint_to_marginals(table)`,
`covariance_rv(joint, x_values, y_values)`.

## Mini-project: simulations and a spam filter from scratch

1. Verify the Monty Hall answer (switching wins 2/3) and the birthday problem (23 people → P(shared birthday) ≈ 0.507)
   by simulation and by exact calculation.
2. Build a Naive Bayes spam classifier on the
   [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (downloaded into `data/raw/`),
   using only `collections.Counter` and logs (Lesson 01.02). Compare its accuracy with scikit-learn's `MultinomialNB`.

## Checkpoint (without notes)

1. P(at least one six in four rolls of a die)?
2. A test is 99% sensitive and 95% specific, and prevalence is 1%. P(disease | positive)?
3. Show that E[aX + b] = aE[X] + b, and compute Var(X) for a fair die.
4. Explain the difference between independent and mutually exclusive events.
5. `make test M=probability` passes.

## Resources for this module

[Introduction to Probability — Blitzstein and Hwang](http://probabilitybook.net) chapters 1–4 ·
[Harvard Stat 110](https://projects.iq.harvard.edu/stat110) lectures 1–12 ·
[Seeing Theory: basic and compound probability](https://seeing-theory.brown.edu/) ·
[3Blue1Brown: Bayes' theorem](https://www.3blue1brown.com/topics/probability)
