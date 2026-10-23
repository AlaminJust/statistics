# 03.05 — Random variables and expectation

**Time:** about 4 hours · **Prerequisites:** 03.01–03.03, 02.02–02.03, 01.04

## Why this matters

A **random variable** turns outcomes into numbers: the number of heads, a customer's spend, a model's prediction error.
Once outcomes are numbers we can average them. The **expected value** is the long-run average and the **variance** is
the long-run spread. These two ideas define almost everything that follows: the expected loss that ML minimises (the
"risk"), the bias and variance of estimators (Module 05), and the mean and variance of every distribution in Module 04.

## Learning objectives

- Define a discrete random variable by its PMF and CDF.
- Compute E[X], E[g(X)], Var(X) and sd(X).
- Use linearity of expectation, including the indicator trick.
- Apply E[aX + b] = aE[X] + b and Var(aX + b) = a²Var(X).
- Explain why E[g(X)] ≠ g(E[X]) in general.

## Before you start

You shuffle 52 cards and count how many are still in their original position. On average, how many do you expect?
(Less than 1? About 1? More?)

---

## 1. Random variables, PMFs and CDFs

A **random variable** X is a function from the sample space to numbers. Roll two dice; X = their sum maps (3, 5) to 8.
Capital letters (X) are random variables; lower case (x) are the values they can take.

A **discrete** random variable takes a countable set of values, and is described by its **probability mass function**
(PMF):

$$p(x) = P(X = x), \qquad p(x) \ge 0, \qquad \sum_x p(x) = 1.$$

The **cumulative distribution function** (CDF) is F(x) = P(X ≤ x). It rises in steps from 0 to 1.

**Fair die.** p(x) = 1/6 for x = 1, …, 6. F(3) = 1/2, F(5) = 5/6 (`die_pmf_cdf.png`).

Continuous random variables (Module 04) use a density f(x) instead, with probabilities as areas (Lesson 01.04), and
Σ becomes ∫ in every formula below.

## 2. Expected value

$$E[X] = \sum_x x\,p(x) \qquad\Big(\text{continuous: } E[X] = \int x f(x)\,dx\Big).$$

It's the **probability-weighted average** of the values (a weighted mean, Lesson 02.02), and the long-run average over
many repetitions.

**Worked example 1.** Fair die: E[X] = (1 + 2 + 3 + 4 + 5 + 6)/6 = **3.5**. You can never roll 3.5; an expected value
need not be a possible value. 500,000 simulated rolls average 3.5016.

**Worked example 2: should you play?** A game costs $2 and pays $10 with probability 0.1. Profit X is 8 with
probability 0.1 and −2 with probability 0.9.
E[X] = 8 × 0.1 + (−2) × 0.9 = 0.8 − 1.8 = **−$1 per game**. A million simulated games lose $1.003 per game on average.
Casinos, insurers and lotteries all work this way: the expected value is in their favour.

### Expectation of a function

$$E[g(X)] = \sum_x g(x)\,p(x).$$

Apply g to each value, then average with the same probabilities. For the die, E[X²] = (1 + 4 + 9 + 16 + 25 + 36)/6 =
91/6 ≈ **15.17**.

**Warning: E[g(X)] ≠ g(E[X]) in general.** (E[X])² = 3.5² = 12.25 ≠ 15.17. And for g(x) = 1/x, E[1/X] = 0.408 but
1/E[X] = 0.286. Equality holds when g is linear. For curved g, *Jensen's inequality* says which way the gap goes: for
convex g (curving up, like x²), E[g(X)] ≥ g(E[X]). That's why the sample standard deviation is biased low (Lesson
02.03, exercise 11), and it's behind the ELBO in Module 07.

## 3. Linearity of expectation: the most useful fact in probability

For any random variables X and Y and constants a, b:

$$E[aX + b] = a\,E[X] + b, \qquad E[X + Y] = E[X] + E[Y].$$

**It holds whether or not X and Y are independent.** That makes it extraordinarily powerful.

**Worked example 3.** E[sum of two dice] = 3.5 + 3.5 = **7** (simulated: 6.996). No need for the 36-outcome table.
E[3X + 2] for one die = 3 × 3.5 + 2 = **12.5**.

### The indicator trick

An **indicator** I_A is 1 if event A happens and 0 otherwise. Its expectation is a probability:
E[I_A] = 1 × P(A) + 0 × (1 − P(A)) = **P(A)**. (Lesson 00.02: the mean of a 0/1 mask is a proportion.)

To find the expected *count* of something, write the count as a sum of indicators and use linearity.

**Worked example 4: fixed points of a shuffle.** Let X = number of cards still in their original position after a
random shuffle of n cards. Write X = I₁ + … + Iₙ, where Iⱼ = 1 if card j stays in place. P(card j stays) = 1/n, so

$$E[X] = \sum_{j=1}^n E[I_j] = n \times \frac{1}{n} = 1.$$

Answer to "Before you start": **exactly 1, for any n**, even though the indicators are dependent. `examples.py` checks
n = 3, …, 6 by listing every permutation, and 50,000 shuffles of 52 cards average 0.9996.

**Worked example 5: the binomial mean.** The number of heads in n flips with P(heads) = p is a sum of n indicators,
each with mean p, so E[heads] = **np**. For n = 20 and p = 0.3, E = 6 (simulated: 5.99).

## 4. Variance and standard deviation

$$\operatorname{Var}(X) = E\big[(X - \mu)^2\big] = E[X^2] - (E[X])^2, \qquad \operatorname{sd}(X) = \sqrt{\operatorname{Var}(X)},$$

where μ = E[X]. It's the expected squared distance from the mean: the population version of Lesson 02.03's s².

**Worked example 1, continued.** Var(die) = 91/6 − 49/4 = 182/12 − 147/12 = **35/12 ≈ 2.917**, and sd ≈ **1.708**.
Simulated: 2.916.

The shortcut form E[X²] − (E[X])² comes from expanding the square and using linearity:
E[X² − 2μX + μ²] = E[X²] − 2μ² + μ² = E[X²] − μ². (Compare Lesson 01.01, worked example 4.)

**Bernoulli(p)** (a single 0/1 trial): E[X²] = p because X² = X, so Var = p − p² = **p(1 − p)**. It's largest at
p = 0.5 (Lesson 01.03, exercise 5): a fair coin is the most uncertain.

### Rules for variance

| Rule | Formula | Note |
| --- | --- | --- |
| Shift | Var(X + b) = Var(X) | spread doesn't move |
| Scale | Var(aX) = a² Var(X) | sd scales by \|a\| |
| Sum of **independent** variables | Var(X + Y) = Var(X) + Var(Y) | needs independence (or zero covariance), Lesson 03.06 |

Var(3X + 2) for a die = 9 × 35/12 = **26.25** (simulated: 26.24).

The binomial variance follows from the sum rule: n independent Bernoulli trials give Var = **np(1 − p)**. For n = 20
and p = 0.3 that's 4.2 (simulated: 4.20).

## 5. Expected value as a decision rule, and its limits

Maximising expected value is a good rule when you face the same gamble many times: the long-run average wins. It's a
poor guide for one-off gambles with catastrophic downside. An insurance policy has negative expected value for the
buyer, yet buying it is rational, because the spread (risk) matters too. In ML the **risk** of a model is its expected
loss, E[L(Y, f(X))], and training minimises an estimate of it, the average loss on the training data.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| E[X²] = (E[X])² | Var(X) = E[X²] − (E[X])² ≥ 0, so E[X²] ≥ (E[X])² |
| E[XY] = E[X]E[Y] always | Only for independent (or uncorrelated) variables, Lesson 03.06 |
| Var(X + Y) = Var(X) + Var(Y) always | Needs zero covariance |
| Var(aX) = a Var(X) | a² Var(X) |
| Thinking linearity needs independence | E[X + Y] = E[X] + E[Y] always |

## Summary

- PMF p(x) = P(X = x); CDF F(x) = P(X ≤ x).
- E[X] = Σ x p(x); E[g(X)] = Σ g(x) p(x); generally E[g(X)] ≠ g(E[X]).
- Linearity: E[aX + bY + c] = aE[X] + bE[Y] + c, always. Indicators turn counts into sums of probabilities.
- Var(X) = E[X²] − (E[X])²; Var(aX + b) = a²Var(X).
- Bernoulli: mean p, variance p(1 − p). Binomial: np and np(1 − p).

## Self-test

1. X takes values 0, 1, 2 with probabilities 0.5, 0.3, 0.2. Find E[X], E[X²] and Var(X).
2. E[X] = 4 and Var(X) = 9. Find E[2X − 1] and sd(2X − 1).
3. A box has 10 tickets numbered 1–10. You draw 3 without replacement. What is the expected sum?
4. Why is E[X²] ≥ (E[X])² always?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
