# 04.01 — Discrete distributions

**Time:** about 4 hours · **Prerequisites:** Module 03

## Why this matters

A handful of named distributions cover most count and yes/no data. Each one comes from a simple **story**: if your data
follow that story, you know the formulas for free. In ML, the Bernoulli is the model behind binary classifiers and
their loss, the categorical is behind softmax, and the Poisson is the basis of count models (Module 06). Once you can
recognise the story, choosing a model becomes much easier.

## Learning objectives

- Recognise the stories behind the Bernoulli, binomial, geometric, Poisson, categorical and multinomial distributions.
- Compute their PMFs, means and variances by hand and with `scipy.stats`.
- Use the Poisson approximation to the binomial.
- Check whether count data look Poisson.

## Before you start

A support desk receives on average 4 emails an hour. Guess the probability that a whole hour passes with no email.

---

## 1. The distributions at a glance

| Distribution | Story | Values | PMF P(X = k) | Mean | Variance |
| --- | --- | --- | --- | --- | --- |
| Bernoulli(p) | one yes/no trial | 0, 1 | pᵏ(1 − p)¹⁻ᵏ | p | p(1 − p) |
| Binomial(n, p) | number of successes in n independent trials | 0, …, n | C(n, k) pᵏ (1 − p)ⁿ⁻ᵏ | np | np(1 − p) |
| Geometric(p) | number of trials until the first success | 1, 2, … | (1 − p)ᵏ⁻¹ p | 1/p | (1 − p)/p² |
| Poisson(λ) | number of events in a fixed time or space, at rate λ | 0, 1, 2, … | λᵏ e^(−λ) / k! | λ | λ |
| Categorical(p₁, …, p_K) | one draw from K categories | 1, …, K | p_k | — | — |
| Multinomial(n, p₁, …, p_K) | counts in each category from n draws | count vectors | n!/(k₁!…k_K!) Π p_jᵏʲ | np_j | np_j(1 − p_j) |

`scipy.stats` provides each one, with the same methods: `pmf`, `cdf` (P(X ≤ k)), `sf` (P(X > k)), `mean`, `var`,
`rvs` (samples) and `ppf` (quantiles).

## 2. Bernoulli and binomial

A **Bernoulli** variable is a single 0/1 trial with P(1) = p. Lesson 03.05 derived its mean p and variance p(1 − p).

A **binomial** variable counts the successes in n *independent* Bernoulli(p) trials. The PMF comes from Lesson 03.02:
choose which k of the n trials succeed (C(n, k) ways), and each such sequence has probability pᵏ(1 − p)ⁿ⁻ᵏ.

**Worked example 2.** A player makes 70% of free throws, independently. In 10 attempts:

- P(exactly 7) = C(10, 7) × 0.7⁷ × 0.3³ = 120 × 0.08235 × 0.027 = **0.2668**.
- P(8 or more) = `stats.binom.sf(7, 10, 0.7)` = **0.3828**. (`sf(k)` is P(X > k), so use 7 to get "≥ 8".)
- P(5 or fewer) = `stats.binom.cdf(5, 10, 0.7)` = **0.1503**.
- Mean np = **7**, variance np(1 − p) = 2.1, sd **1.45**.

Even for a 70% shooter, 7 out of 10 happens only about a quarter of the time. Real performance varies a lot from game
to game, so be careful judging anyone on 10 attempts.

**Check the assumptions** before using the binomial: a fixed number of trials, two outcomes, the same p every time, and
independence. "Hot hands" or fatigue would break the last two.

## 3. Geometric: waiting for the first success

**Worked example 3.** Roll a die until the first six. P(first six on roll 3) = (5/6)² × (1/6) = **0.1157**: two failures,
then a success. P(more than 10 rolls needed) = (5/6)¹⁰ = **0.1615**. On average you wait 1/p = **6** rolls.

Like its continuous cousin the exponential (Lesson 04.02), the geometric is **memoryless**: having failed 5 times
doesn't make a success more likely next time (the gambler's fallacy, Lesson 03.03).

## 4. Poisson: counts of rare, independent events

A **Poisson(λ)** variable counts events that happen independently at a constant average rate λ per interval: emails
per hour, typos per page, goals per match, mutations per gene, server requests per second.

**Worked example 4.** Emails arrive at λ = 4 per hour.

| k | P(X = k) = 4ᵏ e⁻⁴ / k! |
| --- | --- |
| 0 | e⁻⁴ = **0.0183** |
| 2 | 16 e⁻⁴ / 2 = 0.1465 |
| 4 | 256 e⁻⁴ / 24 = 0.1954 |
| 8 | 0.0298 |

Answer to "Before you start": **1.8%**. P(X ≤ 2) = 0.2381 and P(X ≥ 8) = 0.0511.
**Rates scale with the interval:** in 30 minutes the rate is 2, so P(no email in half an hour) = e⁻² = 0.135.

**A signature property: mean = variance = λ.** If count data have a variance much larger than the mean
(*overdispersion*, common in practice), a Poisson model is too simple. Module 06 uses the negative binomial instead.

### The Poisson as a limit of the binomial

With many trials and a small success probability (n large, p small, np = λ moderate), Binomial(n, p) ≈ Poisson(np).

**Worked example 5.** Binomial(1000, 0.003) vs Poisson(3):

| k | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| binomial | 0.0496 | 0.1491 | 0.2242 | 0.2244 | 0.1683 | 0.1009 |
| Poisson | 0.0498 | 0.1494 | 0.2240 | 0.2240 | 0.1680 | 0.1008 |

That's why the Poisson fits "rare events among many opportunities": accidents among many drivers, defects among many
parts.

**Worked example 6: checking a fit.** For 380 (simulated) football matches, estimate λ by the mean number of goals,
2.63, then compare observed counts with 380 × Poisson(2.63) probabilities:

| Goals | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed | 24 | 65 | 106 | 83 | 54 | 34 | 11 | 3 |
| expected | 27.4 | 72.1 | 94.8 | 83.0 | 54.6 | 28.7 | 12.6 | 4.7 |

Close everywhere, and the sample variance (2.29) is near the mean (2.63). Module 05 turns "close" into a formal test
(the χ² goodness-of-fit test).

## 5. Categorical and multinomial: more than two outcomes

A **categorical** variable takes one of K categories with probabilities p₁, …, p_K (summing to 1). It's the
distribution a classifier's softmax output describes (Lesson 01.02). The **multinomial** counts how many of n
independent categorical draws fall in each category, generalising the binomial.

**Worked example 7.** With p = (0.5, 0.3, 0.2), P(5 A, 3 B, 2 C in 10 draws) = 10!/(5! 3! 2!) × 0.5⁵ × 0.3³ × 0.2² =
2520 × 0.03125 × 0.027 × 0.04 = **0.0850**.

## 6. Choosing a discrete model

| Your data | Try |
| --- | --- |
| a single yes/no outcome | Bernoulli |
| successes out of a known number of tries | binomial |
| the number of tries until something happens | geometric |
| a count with no fixed maximum, events at a steady rate | Poisson (check mean ≈ variance) |
| a count with variance ≫ mean | negative binomial (Module 06) |
| one label out of K | categorical |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| `binom.sf(8, …)` for P(X ≥ 8) | `sf(k)` = P(X > k), so use `sf(7, …)` |
| Binomial for dependent trials | Check independence and a constant p |
| Poisson for overdispersed counts | Compare mean and variance first |
| Forgetting to rescale λ for a different interval | λ per hour × hours |

## Summary

- Bernoulli (one trial) → binomial (count in n trials): mean np, variance np(1 − p).
- Geometric: trials until the first success, mean 1/p, memoryless.
- Poisson(λ): counts at rate λ, mean = variance = λ; the limit of the binomial for large n and small p.
- Categorical and multinomial: more than two categories.
- `scipy.stats`: `pmf`, `cdf`, `sf`, `ppf`, `rvs`, `mean`, `var`.

## Self-test

1. A multiple-choice test has 20 questions with 4 options each. Guessing every answer, what are the mean and sd of the
   number correct?
2. A website gets 2 sign-ups per hour on average. P(at least one sign-up in the next 30 minutes)?
3. Why is the geometric distribution called memoryless?
4. Which distribution: the number of heads in 50 flips; the number of meteors seen in an hour; the flip on which the
   first head appears?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
