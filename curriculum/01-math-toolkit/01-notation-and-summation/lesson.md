# 01.01 — Notation and summation

**Time:** about 3 hours · **Prerequisites:** arithmetic, Module 00

## Why this matters

Open any statistics book or ML paper and you will see lines like

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad\text{or}\qquad L(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i).$$

They look intimidating, but they are just compact instructions: *"loop over the data and add things up."* Once you can
read Σ (sigma) fluently, about half of the mathematics in this course becomes readable.

## Learning objectives

- Read and write variables with subscripts: x₁, x₂, …, xₙ.
- Expand and compute sums written with Σ, including double sums.
- Use the three rules of summation to simplify expressions.
- Read product notation Π.
- Recognise the common symbols used in statistics and ML.

## Before you start

Guess: if every data value is reduced by the average, what do the results add up to? Write your guess down.

---

## 1. Data with subscripts

We write a dataset of n numbers as x₁, x₂, …, xₙ. The small number is the **index**: it says *which* observation.

| Notation | Meaning | Example data: 4, 7, 1, 8 |
| --- | --- | --- |
| n | number of observations | n = 4 |
| xᵢ | the i-th value | x₁ = 4, x₃ = 1 |
| xₙ | the last value | x₄ = 8 |

In code, `x[i]` is the same idea, but Python counts from 0, so x₁ is `x[0]`. Maths usually counts from 1.

## 2. Σ — the summation sign

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n$$

Read it as: *"for i from 1 to n, add up xᵢ."* It is exactly this loop:

```python
total = 0
for i in range(1, n + 1):
    total += x[i - 1]        # -1 because Python counts from 0
```

The parts: the **index** (i), its **start** (below Σ), its **end** (above Σ), and the **term** (what's being added).

### Worked example 1: a sum of a formula

$$\sum_{i=1}^{4} (2i + 1)$$

Step by step, substitute i = 1, 2, 3, 4 into the term:

| i | 2i + 1 |
| --- | --- |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| **sum** | **24** |

### Worked example 2: sums over data

Data x = (4, 7, 1, 8).

- Σxᵢ = 4 + 7 + 1 + 8 = **20**
- Σxᵢ² = 16 + 49 + 1 + 64 = **130** (square first, then add)
- (Σxᵢ)² = 20² = **400** (add first, then square)

**Σxᵢ² and (Σxᵢ)² are different.** Mixing them up is the most common notation mistake in statistics. Read the brackets
carefully: whatever is inside the Σ is computed for every i *before* adding.

### The mean

The **mean** (average) is the sum divided by how many:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{20}{4} = 5.$$

x̄ is read "x-bar". A bar over a letter almost always means "average of".

## 3. The three rules of summation

These let you simplify formulas without computing anything.

| Rule | Formula | Why it's true |
| --- | --- | --- |
| **Constants factor out** | Σ c·xᵢ = c · Σ xᵢ | c·x₁ + c·x₂ + … = c(x₁ + x₂ + …) |
| **Sums split** | Σ (xᵢ + yᵢ) = Σ xᵢ + Σ yᵢ | Reorder the additions |
| **Sum of a constant** | Σᵢ₌₁ⁿ c = n·c | Adding c to itself n times |

**Warning:** there is no rule for products. Σ xᵢyᵢ ≠ (Σ xᵢ)(Σ yᵢ) in general.

### Worked example 3: deviations from the mean always sum to zero

This answers "Before you start". A **deviation** is xᵢ − x̄. Using the rules:

$$\sum_{i=1}^{n} (x_i - \bar{x}) = \sum_{i=1}^{n} x_i - \sum_{i=1}^{n} \bar{x} = n\bar{x} - n\bar{x} = 0.$$

Step 1 used "sums split"; step 2 used Σxᵢ = n·x̄ (from the definition of the mean) and "sum of a constant" (x̄ is the
same for every i). Check with the data: deviations are (−1, 2, −4, 3) and −1 + 2 − 4 + 3 = 0. ✓

This is why we can't measure spread by averaging deviations — they always cancel. Module 02 squares them instead.

### Worked example 4: a shortcut you'll meet again

The sum of squared deviations can be computed two ways:

$$\sum (x_i - \bar{x})^2 = \sum x_i^2 - n\bar{x}^2.$$

Proof: expand the square, (xᵢ − x̄)² = xᵢ² − 2x̄xᵢ + x̄², then apply the rules:
Σxᵢ² − 2x̄Σxᵢ + n x̄² = Σxᵢ² − 2x̄(n x̄) + n x̄² = Σxᵢ² − n x̄².

Check: left side = 1 + 4 + 16 + 9 = 30. Right side = 130 − 4 × 25 = 30. ✓

## 4. Double sums

When data have two indexes (rows i and columns j, like a table), sums can be nested:

$$\sum_{i=1}^{2}\sum_{j=1}^{3} i \cdot j$$

It's a nested loop: for each i, loop over all j.

| | j = 1 | j = 2 | j = 3 | row sum |
| --- | --- | --- | --- | --- |
| i = 1 | 1 | 2 | 3 | 6 |
| i = 2 | 2 | 4 | 6 | 12 |
| | | | | **18** |

Shortcut: since the term factors as (i)(j), the sum factors: (1 + 2)(1 + 2 + 3) = 3 × 6 = 18. This only works when the
term is a product of an i-part and a j-part.

## 5. Π — the product sign

$$\prod_{i=1}^{n} x_i = x_1 \times x_2 \times \dots \times x_n$$

Example: Π from i = 1 to 4 of i = 1 × 2 × 3 × 4 = 24 (this is 4!, "4 factorial").

Why statisticians care: the probability that several **independent** events all happen is the product of their
probabilities. If three independent events have probabilities 0.5, 0.2 and 0.1, all three happen with probability
0.5 × 0.2 × 0.1 = 0.01. The **likelihood** in Module 05 is a product like this over the whole dataset, and Lesson 01.02
shows why we take its logarithm.

## 6. Symbols you will meet

| Symbol | Read as | Meaning | Example |
| --- | --- | --- | --- |
| x̄ | x-bar | sample mean | x̄ = 5 |
| μ, σ | mu, sigma | population mean and standard deviation | μ = 100 |
| θ | theta | a model parameter (generic) | θ = probability of heads |
| θ̂ | theta-hat | an *estimate* of θ from data | θ̂ = 0.53 |
| ŷ | y-hat | a prediction | ŷ = model output |
| ∈ | in | is an element of | 3 ∈ {1, 2, 3} |
| ≈ | approximately | | π ≈ 3.14 |
| \|x\| | absolute value | distance from 0 | \|−4\| = 4 |
| ∝ | proportional to | equal up to a constant factor | p(θ \| data) ∝ … |
| argmin | arg-min | the input that makes something smallest | argminₘ (m − 3)² = 3 |
| ⇒ | implies | | |

Convention: **Greek letters for unknown truths, Latin letters (or hats) for things computed from data.**

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Σxᵢ² = (Σxᵢ)² | Square-then-add ≠ add-then-square |
| Σ xᵢyᵢ = Σxᵢ · Σyᵢ | No product rule for sums |
| Forgetting Σᵢ₌₁ⁿ c = n·c (writing just c) | A constant inside Σ is added n times |
| Off-by-one in code | Maths starts at i = 1, Python at index 0 |

## Summary

- Σ = loop and add; Π = loop and multiply.
- Rules: constants factor out, sums split, Σ c = n·c.
- Deviations from the mean sum to 0; Σ(xᵢ − x̄)² = Σxᵢ² − n x̄².
- Greek = truth, hat = estimate.

## Self-test

1. Compute Σ from i = 1 to 5 of i².
2. For data (2, 4, 9), compute Σxᵢ, Σxᵢ², (Σxᵢ)² and x̄.
3. Simplify Σ(3xᵢ + 2) for i = 1 to n in terms of Σxᵢ and n.
4. What is the difference between θ and θ̂?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
