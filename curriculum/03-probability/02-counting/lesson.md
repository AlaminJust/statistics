# 03.02 — Counting

**Time:** about 3 hours · **Prerequisites:** 03.01

## Why this matters

With equally likely outcomes, P(A) = (ways A can happen) / (all possible ways). Real sample spaces are far too big to
list: a deck can be shuffled in about 8 × 10⁶⁷ orders. Counting rules let you find those sizes without listing. They
also give you the binomial coefficients behind the binomial distribution (Module 04), and a feel for combinatorial
explosion: why you can't try every subset of features, and why grid searches get expensive fast.

## Learning objectives

- Apply the multiplication rule.
- Count ordered arrangements (permutations) and unordered selections (combinations).
- Decide whether order matters and whether sampling is with or without replacement.
- Compute probabilities by counting, including the birthday problem.

## Before you start

How many people must be in a room before it's more likely than not that two share a birthday? Guess a number.

---

## 1. The multiplication rule

If a task has k steps, with n₁ ways to do step 1, n₂ ways for step 2 (whatever happened in step 1), and so on, the
total number of ways is n₁ × n₂ × … × n_k.

**Worked example 1.**

| Task | Count |
| --- | --- |
| Outfits from 3 shirts and 4 trousers | 3 × 4 = **12** |
| 4-letter passwords (26 letters, repeats allowed) | 26⁴ = **456,976** |
| 4-character passwords (letters and digits) | 36⁴ = **1,679,616** |
| 8-character passwords (a–z, A–Z, 0–9) | 62⁸ ≈ **2.18 × 10¹⁴** |

Each extra character multiplies the count: that's why long passwords are hard to guess.

## 2. Permutations: order matters

A **permutation** is an ordered arrangement.

- All n objects in a row: **n! = n × (n − 1) × … × 1**. Five books can be shelved in 5! = 120 orders (0! = 1 by
  convention).
- k of n objects in order:

$$P(n, k) = n \times (n-1) \times \dots \times (n - k + 1) = \frac{n!}{(n-k)!}.$$

**Worked example 2.** Gold, silver and bronze among 8 runners: 8 choices for gold, then 7 for silver, then 6 for bronze:
8 × 7 × 6 = **336**. (`math.perm(8, 3)`, and listing them all by brute force agrees.)

## 3. Combinations: order doesn't matter

A **combination** is an unordered selection: a team, a hand of cards, a subset of features.

$$\binom{n}{k} = \frac{n!}{k!\,(n-k)!} \qquad \text{read "n choose k".}$$

Why: each unordered group of k can be arranged in k! orders, so divide the permutation count by k!.

**Worked example 3.** A 3-person team from 8 runners: 336 ordered podiums / 3! orders each = 336 / 6 = **56** teams.
The UK lottery (6 numbers from 49): C(49, 6) = **13,983,816** tickets, so P(jackpot) ≈ 7.2 × 10⁻⁸.

Useful facts:

- C(n, 0) = C(n, n) = 1 and C(n, 1) = n.
- **Symmetry:** C(n, k) = C(n, n − k). Choosing who's in is the same as choosing who's out.
- **Pascal's rule:** C(n, k) = C(n − 1, k − 1) + C(n − 1, k). This builds Pascal's triangle (printed by `examples.py`).
- Σₖ C(n, k) = 2ⁿ: the total number of subsets of n items.

## 4. Which formula? Two questions

| | Order matters | Order doesn't matter |
| --- | --- | --- |
| **With replacement** (repeats allowed) | nᵏ (passwords, dice sequences) | C(n + k − 1, k) (scoops of ice cream; rare in this course) |
| **Without replacement** | n!/(n − k)! (podium) | C(n, k) (teams, hands, samples) |

Ask: (1) Would swapping two picks give a different result? (2) Can the same item be picked twice?

## 5. Probabilities by counting

**Worked example 4: exactly 2 heads in 5 flips.** There are 2⁵ = 32 equally likely sequences. A sequence with exactly 2
heads is fixed by choosing *which* 2 of the 5 positions are heads: C(5, 2) = 10.

P = 10/32 = **0.3125**. In general, P(k heads in n fair flips) = C(n, k)/2ⁿ. That's the binomial distribution (Module 04).

**Worked example 5: quality control.** A box holds 10 items, 3 of them defective. You take 4 at random without
replacement. P(exactly 2 defective)?

- All samples: C(10, 4) = 210.
- Favourable: choose 2 of the 3 defectives **and** 2 of the 7 good ones: C(3, 2) × C(7, 2) = 3 × 21 = 63.
- P = 63/210 = **0.30**. Simulation: 0.2986.

This is the *hypergeometric* distribution: sampling without replacement from two groups. Survey sampling and card
games use it.

**Worked example 6: the birthday problem.** Ignore leap years and assume all 365 days are equally likely. With n people,
compute the complement, "all birthdays differ":

$$P(\text{all different}) = \frac{365}{365} \times \frac{364}{365} \times \dots \times \frac{365 - n + 1}{365}.$$

| People | P(at least one shared birthday) |
| --- | --- |
| 10 | 0.117 |
| **23** | **0.507** |
| 30 | 0.706 |
| 50 | 0.970 |
| 70 | 0.999 |

Answer to "Before you start": **23**, far fewer than most people guess. The reason is that 23 people form
C(23, 2) = 253 *pairs*, and each pair is a chance for a match. In computing the same idea gives "birthday attacks" on
hash functions. In data work it explains why duplicate IDs appear much sooner than you'd expect when IDs are random.

## 6. Combinatorial explosion in ML

| Problem | Count |
| --- | --- |
| Feature subsets of 10 features | 2¹⁰ = 1,024 |
| … of 20 features | 2²⁰ ≈ 1 million |
| … of 40 features | 2⁴⁰ ≈ 1.1 × 10¹² |
| Grid of 5 × 6 × 4 × 3 hyperparameter values | 360 combinations, × 5 CV folds = 1,800 model fits |

That's why "try every subset" is impossible beyond a few dozen features, and why ML uses greedy selection, the lasso
(Module 06) or random search instead.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Using permutations for a team or hand | Teams are unordered: C(n, k) |
| Forgetting that a draw without replacement removes an item | The next step has one fewer choice |
| Double-counting in "2 defective and 2 good" | Multiply the separate choices, C(3,2) × C(7,2) |
| Direct calculation of "at least one match" | Use the complement |

## Summary

- Multiplication rule: multiply the counts of independent steps.
- n! arrangements; P(n, k) = n!/(n − k)! ordered selections; C(n, k) = n!/(k!(n − k)!) unordered.
- Decide order and replacement first.
- P = favourable/total for equally likely outcomes; use complements for "at least one".

## Self-test

1. How many ways can 10 people be seated in a row? How many ways can a committee of 4 be chosen from them?
2. Why is C(n, k) = C(n, n − k)?
3. P(a 5-card poker hand is all hearts)?
4. Why does the birthday probability grow so fast with n?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
