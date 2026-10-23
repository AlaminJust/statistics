# 03.03 Exercises — Conditional probability and independence

`make lesson L=03-probability/03-conditional-probability-independence`

## ★ Warm-up

1. **(M)** P(A) = 0.5, P(B) = 0.4, P(A ∩ B) = 0.2. Find P(A | B) and P(B | A). Are A and B independent?
2. **(M)** Three cards are drawn without replacement. P(all three are hearts)?
3. **(M)** Two cards without replacement: P(at least one ace)? (Use the complement.)

## ★★ Core

4. **(M)** 30% of emails are spam. 80% of spam and 10% of legitimate emails contain a link. Find P(an email contains a
   link) using the law of total probability. Draw the tree.
5. **(M)** An urn has 3 red and 2 blue balls. Two are drawn without replacement. Show that P(second ball is red) = 3/5,
   the same as for the first ball. Why does that make sense?
6. **(M)** A family has two children, each equally likely to be a boy or a girl, independently. Find
   (a) P(both boys | the older is a boy) and (b) P(both boys | at least one is a boy). Why do they differ?
7. **(M)** A system has three independent components, each working with probability 0.95. Find P(system works) if they
   are (a) in series (all must work) and (b) in parallel (at least one must work).
8. **(P)** Using `students.csv`, check whether `tutoring` and `program` look independent: compare P(tutoring) with
   P(tutoring | program) for each program. Then do the same for "exam_score ≥ 70" and "study_hours ≥ 12". Which pair is
   clearly dependent?
9. **(P)** Verify exercise 6(b) by simulating 100,000 two-child families.

## ★★★ Challenge

10. **(M)** Show that if A and B are independent, then so are A and Bᶜ. (Hint: P(A) = P(A ∩ B) + P(A ∩ Bᶜ).)
11. **(M)** **Pairwise but not mutually independent.** Flip two fair coins. A = first is heads, B = second is heads,
    C = the two coins match. Show that every pair is independent but P(A ∩ B ∩ C) ≠ P(A)P(B)P(C).
12. **(P)** **Monty Hall.** There are three doors: a car is behind one and goats behind the others. You pick door 1. The
    host, who knows where the car is, opens another door with a goat behind it. Should you switch? Simulate 100,000 games
    for "stay" and for "switch", then explain the answer with conditional probability.

Solutions: [solutions/solutions.md](solutions/solutions.md)
