# 03.02 Exercises — Counting

`make lesson L=03-probability/02-counting`

## ★ Warm-up

1. **(M)** How many ways can 10 people sit in a row? How many 4-person committees can be chosen from them?
2. **(M)** How many distinct arrangements of the letters in STATS are there? (Hint: divide by the orders of repeated letters.)
   And of MISSISSIPPI's first 7 letters, MISSISS?
3. **(M)** Six fair coin flips: P(exactly 3 heads)?

## ★★ Core

4. **(M)** A club has 10 women and 8 men. In how many ways can it form a committee of 3 women and 2 men?
5. **(M)** A 5-card poker hand is dealt from a 52-card deck. Find (a) the number of possible hands, (b) P(all hearts),
   (c) P(a flush: all five cards of the same suit, any suit), (d) P(exactly one pair).
6. **(M)** Ten fair coin flips: P(8 or more heads)?
7. **(M)** A jar has 5 red, 3 blue and 4 green marbles. Five are drawn without replacement. P(2 red, 2 blue, 1 green)?
8. **(P)** Implement `n_choose_k(n, k)` (without `math.comb`; use a loop that multiplies and divides as it goes so
   intermediate numbers stay small) and `permutations(n, k)` in `statkit/probability.py`. Test with
   `make test M=probability`.
9. **(P)** Check exercises 5(b) and 7 by simulation with 100,000 trials.

## ★★★ Challenge

10. **(M)** Birthday variant: in a room of 30 people, what is P(at least one person shares *your* birthday)? Why is it so
    much smaller than P(some pair shares a birthday) = 0.706?
11. **(M)** Prove Pascal's rule C(n, k) = C(n − 1, k − 1) + C(n − 1, k) by a story: fix one person and split the
    committees by whether they include that person.
12. **(P)** You generate random 32-bit IDs (2³² ≈ 4.3 billion possible values). How many records can you have before
    P(at least one duplicate ID) passes 50%? Use the birthday formula with logs (sum of log(1 − i/N)) to avoid
    underflow. Repeat for 64-bit IDs. What does this mean for database design?

Solutions: [solutions/solutions.md](solutions/solutions.md)
