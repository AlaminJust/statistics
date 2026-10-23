# 04.01 Exercises — Discrete distributions

`make lesson L=04-distributions/01-discrete-distributions`

## ★ Warm-up

1. **(M)** A 20-question multiple-choice test has 4 options per question. Guessing every answer, find the mean and sd of
   the number correct, P(0 correct) and P(10 or more correct).
2. **(M)** A website gets 2 sign-ups per hour on average. P(at least one sign-up in the next 30 minutes)?
3. **(C)** Name the distribution: (a) heads in 50 flips; (b) meteors seen in an hour; (c) the flip on which the first head
   appears; (d) which of 5 products a customer buys.

## ★★ Core

4. **(M)** By hand, then with scipy: P(exactly 3 heads in 5 fair flips); P(at least 2 defective in 8 items when each is
   defective with probability 0.1, independently).
5. **(M)** Accidents at a junction follow a Poisson distribution with mean 3 per month. Find P(no accidents in a month),
   P(more than 4 in a month) and P(at most 10 in a 4-month period).
6. **(M)** A salesperson closes each call with probability 0.2, independently. P(the first sale comes within the first 4
   calls)? How many calls does the first sale take on average?
7. **(M)** A factory makes 500 parts a day, each defective with probability 0.004. Compute P(4 or more defective) exactly
   (binomial) and with the Poisson approximation.
8. **(P)** Implement `binomial_pmf(k, n, p)`, `poisson_pmf(k, lam)` and `geometric_pmf(k, p)` in
   `statkit/distributions.py` (use `math.comb` or your `n_choose_k`, and `math.exp`). Check that each PMF sums to 1
   (Poisson: sum k = 0…100).
9. **(P)** Simulate 100,000 draws from Binomial(20, 0.25) with NumPy and plot their histogram against the exact PMF.

## ★★★ Challenge

10. **(M)** Show that the Poisson PMF sums to 1, using the series eˣ = Σ xᵏ/k!. Then show that its mean is λ.
11. **(M)** Starting from the binomial PMF with p = λ/n, show that it tends to the Poisson PMF as n → ∞. (Hint:
    (1 − λ/n)ⁿ → e^(−λ), and n(n − 1)…(n − k + 1)/nᵏ → 1.)
12. **(P)** **Overdispersion.** Simulate counts where each of 1,000 customers has their own rate λᵢ drawn from a gamma
    distribution with mean 3 (shape 1.5, scale 2), and then the count is Poisson(λᵢ). Compare the mean and variance of
    the counts. Why does a single Poisson fit badly, and what kind of real data behaves like this?

Solutions: [solutions/solutions.md](solutions/solutions.md)
