# 03.05 Exercises — Random variables and expectation

`make lesson L=03-probability/05-random-variables-expectation`

## ★ Warm-up

1. **(M)** X takes values 0, 1, 2 with probabilities 0.5, 0.3, 0.2. Find E[X], E[X²], Var(X) and the CDF.
2. **(M)** E[X] = 4 and Var(X) = 9. Find E[2X − 1], Var(2X − 1) and sd(2X − 1).
3. **(M)** On an American roulette wheel (38 pockets, 18 red), a $1 bet on red wins $1 or loses $1. Find the expected
   profit per bet.

## ★★ Core

4. **(M)** An insurer sells a $300 policy that pays $20,000 with probability 0.01 and nothing otherwise. Find the
   insurer's expected profit per policy and its standard deviation. Why does the insurer need many customers?
5. **(M)** Use indicators to find the expected number of distinct faces in 6 rolls of a die. (Hint: Iⱼ = 1 if face j
   appears at least once.)
6. **(M)** Find E[max of two dice] from the 36-outcome table. Compare with E[one die] = 3.5.
7. **(M)** 100 fair coin flips: find the mean, variance and standard deviation of the number of heads. About how many
   heads would surprise you?
8. **(P)** Implement `expectation(values, probs)` and `variance_rv(values, probs)` in `statkit/probability.py`. Check the
   die (3.5 and 35/12) and exercise 1. Raise `ValueError` if the probabilities don't sum to 1 (within 1e-9).
9. **(P)** Simulate exercises 5 and 6 with 100,000 trials each.

## ★★★ Challenge

10. **(M)** **Coupon collector.** How many rolls of a die do you need, on average, to see all six faces? (Hint: after you
    have seen k distinct faces, the wait for a new one is geometric with success probability (6 − k)/6 and mean
    6/(6 − k).) Simulate to check.
11. **(M)** Prove E[aX + bY] = aE[X] + bE[Y] for discrete X and Y using the joint PMF. Notice where (if anywhere) you used
    independence.
12. **(P)** **St. Petersburg paradox.** Flip a coin until the first heads; if that happens on flip k you win 2ᵏ dollars.
    Show that E[winnings] is infinite. Then simulate 10⁶ games and plot the running average. What happens, and why do
    people refuse to pay much to play?

Solutions: [solutions/solutions.md](solutions/solutions.md)
