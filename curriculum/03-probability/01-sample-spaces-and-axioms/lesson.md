# 03.01 — Sample spaces and the axioms of probability

**Time:** about 3 hours · **Prerequisites:** Modules 01–02

## Why this matters

Probability is the mathematics of uncertainty. Every classifier outputs probabilities, every loss in Module 04 is a
probability statement, and every p-value and confidence interval in Module 05 is built on the rules in this lesson.
They come down to three axioms and a few consequences. Learn them precisely and you can derive everything else.

## Learning objectives

- Describe a random experiment by its sample space and events.
- Combine events with union, intersection and complement.
- State the three axioms and derive the complement and addition rules from them.
- Compute probabilities of equally likely outcomes by counting.
- Check any probability calculation by simulation.

## Before you start

Roll a fair die four times. Guess the probability of getting at least one six. More than 50% or less?

---

## 1. Experiments, outcomes, sample spaces, events

| Term | Meaning | Die example |
| --- | --- | --- |
| **Random experiment** | a process with an uncertain result | roll a die once |
| **Outcome** | one possible result | 4 |
| **Sample space** S | the set of all outcomes | S = {1, 2, 3, 4, 5, 6} |
| **Event** | a set of outcomes (a subset of S) | "even" = {2, 4, 6} |

An event **happens** if the outcome is one of its elements. Sample spaces can be:

- finite: a die, a card, a coin;
- countably infinite: the number of emails tomorrow, {0, 1, 2, …};
- continuous: a waiting time, any number in [0, ∞). Probabilities then come from areas (Lesson 01.04).

## 2. Combining events

| Operation | Symbol | Read as | Die example with A = even, B = {5, 6} |
| --- | --- | --- | --- |
| union | A ∪ B | A **or** B (or both) | {2, 4, 5, 6} |
| intersection | A ∩ B | A **and** B | {6} |
| complement | Aᶜ | **not** A | {1, 3, 5} |
| disjoint | A ∩ B = ∅ | can't both happen | "even" and "odd" |

A Venn diagram helps: two overlapping circles inside a rectangle S. The union is everything in either circle, and the
intersection is the overlap.

## 3. The three axioms (Kolmogorov, 1933)

A probability P assigns a number to every event so that:

1. **Non-negativity:** P(A) ≥ 0 for every event A.
2. **Normalisation:** P(S) = 1. Something in the sample space certainly happens.
3. **Additivity:** if A and B are disjoint, P(A ∪ B) = P(A) + P(B). (The same holds for any countable list of
   mutually disjoint events.)

That's all. Everything else is derived.

### Consequences (with proofs)

**Complement rule:** P(Aᶜ) = 1 − P(A).
*Proof:* A and Aᶜ are disjoint and A ∪ Aᶜ = S. By axioms 3 and 2, P(A) + P(Aᶜ) = 1.

**Impossible event:** P(∅) = 0, since ∅ = Sᶜ.

**Bounded:** 0 ≤ P(A) ≤ 1.

**Monotone:** if A ⊆ B, then P(A) ≤ P(B).

**Addition rule (inclusion–exclusion):**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B).$$

*Proof sketch:* adding P(A) and P(B) counts the overlap A ∩ B twice, so subtract it once. Formally, split A ∪ B into the
disjoint pieces A and (B without A), then apply axiom 3.

## 4. Equally likely outcomes: probability by counting

If S has N equally likely outcomes, then

$$P(A) = \frac{\text{number of outcomes in } A}{N}.$$

**Worked example 1: one die.** A = even = {2, 4, 6}, B = "more than 4" = {5, 6}.

- P(A) = 3/6 = 1/2 and P(B) = 2/6 = 1/3.
- A ∩ B = {6}, so P(A ∩ B) = 1/6.
- Addition rule: P(A ∪ B) = 1/2 + 1/3 − 1/6 = **2/3**. Check by listing: A ∪ B = {2, 4, 5, 6}, which is 4/6 = 2/3 ✓.
- Simulation with 100,000 rolls gives 0.6655.

**Worked example 2: two dice.** The sample space is the 36 ordered pairs (first die, second die), all equally likely.
Count the pairs for each sum:

| Sum | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pairs | 1 | 2 | 3 | 4 | 5 | **6** | 5 | 4 | 3 | 2 | 1 |

P(sum = 7) = 6/36 = **1/6**, the most likely sum. P(sum = 2) = 1/36.

**Careful:** the *sums* 2, …, 12 are not equally likely. Equally likely outcomes must be the basic pairs. Treating
(1, 2) and (2, 1) as the same outcome is a classic mistake.

**Worked example 3: the complement trick.** P(at least one six in four rolls). "At least one" has many cases (one six,
two sixes, …), but its complement, "no sixes", is a single case: each roll avoids six with probability 5/6, and
(anticipating Lesson 03.03) the rolls are independent.

P(at least one six) = 1 − (5/6)⁴ = 1 − 0.4823 = **0.5177**. Simulation gives 0.5166.

So yes, slightly more than 50%. In the 1600s gamblers bet on this, and it led Pascal and Fermat to found probability
theory. **Rule of thumb: for "at least one", compute the complement.**

**Worked example 4: cards.** A standard deck has 52 cards. P(heart) = 13/52 = 1/4, P(king) = 4/52 = 1/13, and
P(heart and king) = 1/52 (the king of hearts).
P(heart or king) = 13/52 + 4/52 − 1/52 = 16/52 = **4/13 ≈ 0.308**.

## 5. What does a probability *mean*?

| Interpretation | P(A) is… | Example |
| --- | --- | --- |
| **Frequentist** | the long-run share of times A happens in repeated trials | P(heads) = 0.5: in many flips, about half are heads |
| **Bayesian** | a degree of belief, updated with evidence | "70% chance it rains tomorrow"; "80% sure this code has a bug" |

Both obey the same axioms. Frequentist methods dominate Modules 05–06; Bayesian methods are Module 07.

`examples.py` shows the frequentist idea: the running share of heads wanders early on (0.500 after 10 flips, 0.492 after
1,000) and settles towards 0.5 (0.5055 after 10,000). Module 04 proves this as the **law of large numbers**.

## 6. Simulation: your universal checking tool

Any probability you calculate can be estimated by simulating the experiment many times and counting:

```python
rng = np.random.default_rng(2026)
rolls = rng.integers(1, 7, size=(200_000, 4))      # 200,000 experiments of 4 rolls
estimate = (rolls == 6).any(axis=1).mean()         # share of experiments with at least one six
```

The estimate's typical error shrinks like 1/√(number of trials) (Module 04), so 100,000 trials give about 2–3 correct
decimal places. **Habit for this module: compute exactly, then confirm by simulation.** When the two disagree, one of
them has a bug, and finding it teaches you something.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| P(A or B) = P(A) + P(B) for overlapping events | Subtract P(A ∩ B) |
| Treating unequal outcomes as equally likely (sums of dice) | Use the basic equally likely outcomes (ordered pairs) |
| Listing every case for "at least one" | 1 − P(none) |
| A probability above 1 or below 0 | Something is wrong; recheck |

## Summary

- Sample space S = all outcomes; an event = a subset of S.
- Axioms: P ≥ 0, P(S) = 1, disjoint events add.
- Derived: P(Aᶜ) = 1 − P(A); P(A ∪ B) = P(A) + P(B) − P(A ∩ B).
- Equally likely outcomes: P(A) = |A| / |S|.
- Always check by simulation.

## Self-test

1. Write the sample space for flipping a coin three times. How many outcomes are there?
2. P(A) = 0.6, P(B) = 0.5, P(A ∩ B) = 0.3. Find P(A ∪ B) and P(neither).
3. Can two events with P(A) = 0.6 and P(B) = 0.7 be disjoint? Why?
4. P(at least one head in 10 flips)?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
