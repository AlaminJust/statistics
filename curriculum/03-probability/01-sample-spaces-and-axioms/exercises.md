# 03.01 Exercises — Sample spaces and axioms

`make lesson L=03-probability/01-sample-spaces-and-axioms`

## ★ Warm-up

1. **(M)** List the sample space for three coin flips. Find P(exactly two heads) and P(at least one head).
2. **(M)** P(A) = 0.6, P(B) = 0.5, P(A ∩ B) = 0.3. Find P(A ∪ B), P(Aᶜ) and P(neither A nor B).
3. **(C)** Can events with P(A) = 0.6 and P(B) = 0.7 be disjoint? Explain using the axioms.

## ★★ Core

4. **(M)** Two dice. Find P(sum ≥ 10), P(at least one die shows 1) and P(the two dice differ).
5. **(M)** From the axioms, prove that if A ⊆ B then P(A) ≤ P(B). (Hint: write B as A ∪ (B ∩ Aᶜ).)
6. **(M)** Prove inclusion–exclusion for three events: P(A ∪ B ∪ C) = P(A) + P(B) + P(C) − P(A∩B) − P(A∩C) − P(B∩C) + P(A∩B∩C).
   Check it on one die with A = {1, 2, 3}, B = {2, 4, 6}, C = {3, 6}.
7. **(M)** P(at least one head in 10 flips)? P(at least one six in 24 throws of two dice, where a "double six" counts)?
   (This second one is de Méré's famous bet. Compare with 1/2.)
8. **(P)** Implement `simulate_probability(event, trials, rng)` in `statkit/probability.py`: it calls `event(rng)` (a
   function that runs one experiment and returns True or False) `trials` times and returns the share of True results.
   Use it to check exercises 4 and 7.
9. **(P)** Simulate 10,000 flips and plot the running share of heads with a log x-axis. Repeat with 5 different seeds on
   the same axes. What do the five curves have in common for large n?

## ★★★ Challenge

10. **(M)** A card is drawn from a deck. Find P(face card or red). (Face cards: J, Q, K.)
11. **(P)** **Bertrand's box.** Three boxes hold two gold coins, two silver coins, and one of each. You pick a box at
    random and draw a coin: it's gold. What is the probability that the other coin in the same box is gold? Guess, then
    simulate. (You'll be able to prove it in Lesson 03.04.)
12. **(C)** A weather forecaster says "70% chance of rain" every day for a year. How would you check whether the forecaster
    is *calibrated*? Connect your answer to the frequentist interpretation. (Module 08 calls this a reliability diagram.)

Solutions: [solutions/solutions.md](solutions/solutions.md)
