# 03.03 — Conditional probability and independence

**Time:** about 4 hours · **Prerequisites:** 03.01, 03.02, 02.06 (contingency tables)

## Why this matters

Information changes probabilities. The chance a patient has a disease changes once you see a test result. The chance
an email is spam changes once you see the word "lottery". A classifier's output is a conditional probability,
P(class | features). **Independence**, the special case where information *doesn't* change anything, is the assumption
behind the product formula for likelihoods (Module 05) and behind Naive Bayes. Getting conditioning right is the single
most important skill in probability, and getting it wrong causes famous errors in courts, medicine and ML.

## Learning objectives

- Compute P(A | B) from its definition, from counts and from tables.
- Use the multiplication rule and tree diagrams for sequential experiments.
- Test independence and distinguish it from mutual exclusivity.
- Apply the law of total probability.
- Explain conditional independence and the gambler's fallacy.

## Before you start

A fair coin lands heads five times in a row. Is the next flip more likely to be tails, "to balance out"? Write your answer.

---

## 1. Conditional probability

The probability of A **given** that B happened:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0.$$

Picture it as **shrinking the sample space** to B and asking what fraction of B is also in A.

**Worked example 1: two dice.**

- P(sum = 8 | first die = 3). Given the first die is 3, there are 6 equally likely outcomes (3, 1), …, (3, 6), and
  only (3, 5) sums to 8. **1/6**.
- P(first die = 3 | sum = 8). The sum-8 outcomes are (2, 6), (3, 5), (4, 4), (5, 3), (6, 2). One of the five has
  first die 3. **1/5**.

The two answers differ: **P(A | B) ≠ P(B | A) in general.** Confusing them is the *prosecutor's fallacy*:
"P(this evidence | innocent) is tiny" does not mean "P(innocent | this evidence) is tiny". Lesson 03.04 shows how to get
from one to the other.

**Worked example 2: from a table.** From `students.csv` (Lesson 02.06): 28 of the 79 Science students attend
tutoring, so P(tutoring | Science) = 28/79 = **0.354**. But 28 of the 60 tutoring students are in Science, so
P(Science | tutoring) = 28/60 = **0.467**. The numerator is the same; the denominator is the conditioning group.

## 2. The multiplication rule and tree diagrams

Rearranging the definition gives the **multiplication rule**:

$$P(A \cap B) = P(B)\,P(A \mid B) = P(A)\,P(B \mid A).$$

For sequences: P(A₁ ∩ A₂ ∩ A₃) = P(A₁) P(A₂ | A₁) P(A₃ | A₁ ∩ A₂).

**Worked example 3: two aces.** Draw two cards without replacement.
P(first ace) = 4/52. *Given* the first was an ace, 3 aces remain among 51 cards: P(second ace | first ace) = 3/51.

P(both aces) = 4/52 × 3/51 = **1/221 ≈ 0.00452** (simulated: 0.00458).
With replacement it would be (4/52)² = 0.00592: the draws would be independent.

A **tree diagram** draws this: one branch per outcome at each stage, labelled with the conditional probability. Multiply
along a path to get the probability of that path, and add paths that lead to the same event.

```text
              ┌─ ace (3/51) ─────► both aces: 4/52 × 3/51
  ace (4/52) ─┤
              └─ not (48/51)
              ┌─ ace (4/51)
 not (48/52) ─┤
              └─ not (47/51)
```

## 3. Independence

A and B are **independent** if knowing one happened doesn't change the probability of the other:

$$P(A \mid B) = P(A) \iff P(A \cap B) = P(A)\,P(B).$$

**Worked example 4: one die.** A = even = {2, 4, 6}.

| B | P(A ∩ B) | P(A) P(B) | Verdict |
| --- | --- | --- | --- |
| {1, 2} | P({2}) = 1/6 | 1/2 × 1/3 = 1/6 | **independent** |
| {4, 5, 6} | P({4, 6}) = 1/3 | 1/2 × 1/2 = 1/4 | dependent |

Independence is about probabilities, not about physical connection. You check it with the formula.

### Independent vs mutually exclusive: opposites, not synonyms

| | Mutually exclusive (disjoint) | Independent |
| --- | --- | --- |
| Definition | P(A ∩ B) = 0 | P(A ∩ B) = P(A)P(B) |
| If A happens… | B **can't** happen | B's chance is **unchanged** |
| Example | "even" and "odd" on one roll | heads on flip 1 and heads on flip 2 |

Two disjoint events with positive probabilities are always **dependent**: learning A happened tells you B didn't.

### Several events

Events A₁, …, Aₙ are (mutually) independent if *every* subset multiplies, e.g. P(A₁ ∩ A₂ ∩ A₃) = P(A₁)P(A₂)P(A₃).
Repeated coin flips, dice rolls and **observations sampled at random** are modelled this way. That is why the
probability of a whole dataset is a product (the likelihood, Module 05), and why "at least one six in four rolls" was
1 − (5/6)⁴ in Lesson 03.01.

### The gambler's fallacy

Answer to "Before you start": **no**. Independent flips have no memory, so P(heads) is still 0.5. In `examples.py`,
across a million flips there were 31,183 runs of five or more heads, and the next flip was heads **50.03%** of the time.
"Balancing out" happens only in *proportions* over very long runs (Lesson 03.01), and only because new flips dilute
the early streak, not because the coin compensates.

## 4. The law of total probability

If B₁, …, B_k split the sample space into disjoint pieces (a **partition**), then

$$P(A) = \sum_{j=1}^k P(A \mid B_j)\,P(B_j).$$

Split by cases, compute each case, then take a weighted average of the cases.

**Worked example 5: two machines.** Machine 1 makes 60% of the parts with a 2% defect rate; machine 2 makes 40% with a
5% defect rate.

P(defect) = P(defect | M1)P(M1) + P(defect | M2)P(M2) = 0.02 × 0.6 + 0.05 × 0.4 = 0.012 + 0.020 = **0.032**.
Simulation: 0.0317.

This is a weighted mean (Lesson 02.02), with the case probabilities as weights. Lesson 03.04 uses it as the denominator
of Bayes' theorem.

## 5. Conditional independence

A and B are **conditionally independent given C** if P(A ∩ B | C) = P(A | C) P(B | C).

**Worked example 7.** Two students from the same school take a test. The pass rate is 90% in good schools and 40% in
weak ones, and half the schools are good. Within a school, the two results are independent.

| | P(B passes) | P(B passes \| A passes) |
| --- | --- | --- |
| overall | 0.649 | **0.746**: dependent |
| within good schools | 0.900 | 0.900: independent |
| within weak schools | 0.398 | 0.400: independent |

A's pass is evidence that the school is good, which raises B's chance, so the two results are dependent *overall*.
Once you know the school, A tells you nothing more about B. A shared cause creates dependence; conditioning on that cause
removes it. This is the confounding of Lesson 02.06, written as probability, and the core idea of causal diagrams
(Module 09).

**Naive Bayes** assumes the words in an email are conditionally independent *given* the class (spam or not). The
assumption is false, but the classifier still works surprisingly well (Lesson 03.04).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| P(A \| B) = P(B \| A) | Different denominators; use Bayes (03.04) |
| "Disjoint" means "independent" | Disjoint events are dependent (if both have P > 0) |
| P(A ∩ B) = P(A)P(B) without checking | True only for independent events; otherwise use P(A)P(B \| A) |
| Gambler's fallacy | Independent trials have no memory |
| Assuming independence for repeated measurements of the same person | Same-unit measurements are usually dependent (Module 06: mixed models) |

## Summary

- P(A | B) = P(A ∩ B)/P(B): restrict to B.
- Multiplication rule: P(A ∩ B) = P(A) P(B | A). Trees multiply along paths.
- Independence: P(A ∩ B) = P(A)P(B). Disjoint events are not independent.
- Total probability: P(A) = Σ P(A | Bⱼ) P(Bⱼ).
- Conditional independence: a shared cause creates dependence that disappears when you condition on it.

## Self-test

1. P(A) = 0.5, P(B) = 0.4, P(A ∩ B) = 0.2. Find P(A | B). Are A and B independent?
2. Draw 3 cards without replacement. P(all three are hearts)?
3. Can two independent events both with probability 0.5 be disjoint?
4. 30% of emails are spam; 80% of spam and 10% of non-spam contain a link. P(an email contains a link)?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
