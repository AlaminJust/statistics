# 04.04 Exercises — Law of large numbers and the CLT

`make lesson L=04-distributions/04-lln-and-clt`

## ★ Warm-up

1. **(M)** Exam scores have mean 64 and sd 14. Use the CLT to approximate P(the mean of 49 random students exceeds 67).
2. **(M)** A coin is flipped 10,000 times. Give the approximate sd of the number of heads and of the proportion of heads.
3. **(C)** Explain the difference between the LLN and the CLT in two sentences, without formulas.

## ★★ Core

4. **(M)** You roll 100 dice. Approximate P(sum ≥ 370), with a continuity correction. (One die: mean 3.5, variance 35/12.)
5. **(M)** A candidate truly has 52% support. In a random poll of 1,000 voters, what is the probability the poll shows
   them below 50%?
6. **(M)** A website converts 2% of visitors. Of 5,000 visitors, approximate P(120 or more conversions) with the CLT, and
   compare with the exact binomial.
7. **(M)** How many test examples does Hoeffding require to estimate accuracy within ±0.01 with 99% confidence? Compare
   with the CLT answer for the worst case p = 0.5.
8. **(M)** Commute times have mean 23.5 and sd 20.4 minutes. Use Markov and then Chebyshev to bound P(commute ≥ 100).
   What share of `students.csv` actually commutes ≥ 100 minutes?
9. **(P)** Pick a non-normal distribution (uniform, Poisson(1), or Beta(0.5, 0.5)). For n = 1, 2, 5, 10 and 30, simulate
   50,000 sample means, then plot their histograms with the CLT normal curve overlaid. Which distribution converges
   fastest, and why?

## ★★★ Challenge

10. **(P)** **Monte Carlo π.** Draw n points uniformly in the unit square and estimate π as 4 × (share inside the quarter
    circle). For n = 10², 10⁴ and 10⁶, compute the estimate and a CLT-based 95% interval. How does the error scale with n?
11. **(P)** **When n = 30 is not enough.** Simulate means of n draws from a log-normal distribution
    (`rng.lognormal(0, 1.5, ...)`, very skewed) for n = 30, 100 and 1,000. Report the skewness of the mean and both tail
    probabilities, P(Z < −1.96) and P(Z > 1.96). How large does n need to be for the tails to look normal?
12. **(M)** Show that the Hoeffding bound for a test-set accuracy gives the same n whatever the true accuracy is, while
    the CLT-based n depends on p. When does that make Hoeffding much more conservative?

Solutions: [solutions/solutions.md](solutions/solutions.md)
