# Module 03 — Probability

**Weeks 6–8 · about 24 hours**

Probability is the language of uncertainty. Every ML model outputs probabilities or is trained with a probabilistic
loss, and every statistical test is a probability statement. This module builds probability from three axioms, and
checks every result against a simulation.

## Lessons

| # | Lesson | Key idea | statkit functions |
| --- | --- | --- | --- |
| 03.01 | [Sample spaces and axioms](01-sample-spaces-and-axioms/lesson.md) | Three axioms imply all the rules; check everything by simulation | `simulate_probability` |
| 03.02 | [Counting](02-counting/lesson.md) | Order? Replacement? Then n!, P(n, k) or C(n, k) | `permutations`, `n_choose_k` |
| 03.03 | [Conditional probability and independence](03-conditional-probability-independence/lesson.md) | P(A \| B) restricts to B; disjoint ≠ independent | `total_probability` |
| 03.04 | [Bayes' theorem](04-bayes-theorem/lesson.md) | posterior ∝ likelihood × prior; base rates matter | `bayes_posterior` |
| 03.05 | [Random variables and expectation](05-random-variables-expectation/lesson.md) | Linearity of expectation always holds | `expectation`, `variance_rv` |
| 03.06 | [Joint distributions and covariance](06-joint-distributions-covariance/lesson.md) | Var(X̄) = σ²/n | `joint_to_marginals`, `covariance_rv` |

Run `make test M=probability` as you implement each function.

## Mini-project: simulations and a spam filter from scratch

Brief in [projects/m03-spam-filter/README.md](../../projects/m03-spam-filter/README.md).

## Checkpoint (without notes)

1. P(at least one six in four rolls of a die)?
2. A test is 99% sensitive and 95% specific, and prevalence is 1%. P(disease | positive)? Explain with natural frequencies.
3. Show that E[aX + b] = aE[X] + b, and compute Var(X) for a fair die.
4. Explain the difference between independent and mutually exclusive events.
5. Derive Var(X̄) = σ²/n for i.i.d. observations.
6. `make test M=probability` passes.

Pass all six → `git tag -a m03-done -m "Module 03 complete"`.

## Resources

[Introduction to Probability — Blitzstein and Hwang](http://probabilitybook.net) chapters 1–4 and 7 ·
[Harvard Stat 110](https://projects.iq.harvard.edu/stat110) lectures 1–12 ·
[Seeing Theory: basic and compound probability](https://seeing-theory.brown.edu/) ·
[3Blue1Brown: Bayes' theorem](https://www.3blue1brown.com/topics/probability)
