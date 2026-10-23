# 04.02 Exercises — Continuous distributions

`make lesson L=04-distributions/02-continuous-distributions`

## ★ Warm-up

1. **(M)** X ~ Uniform(2, 8). Find P(3 < X < 5), E[X] and Var(X).
2. **(M)** Light bulbs fail at rate 1 per 1,000 hours (exponential). P(a bulb lasts more than 2,000 hours)? Its median
   lifetime?
3. **(P)** What do `stats.norm.ppf(0.95)` and `stats.norm.cdf(-1)` return? Explain each in words.

## ★★ Core

4. **(M)** Call-centre waiting times are exponential with mean 8 minutes. Find P(wait > 10), P(2 < wait < 5) and the wait
   that only 1% of callers exceed.
5. **(M)** Customers arrive at rate 3 per 10 minutes (Poisson). Use the exponential to find P(no customer in the next 2
   minutes). Check it with the Poisson PMF.
6. **(M)** For Beta(2, 5), find the mean, P(X < 0.5) and the central 90% interval (scipy).
7. **(M)** A Gamma(shape 3, scale 2) is the sum of three independent exponential waits with mean 2. Find its mean,
   variance and P(X > 10). Check by simulating sums of three exponentials.
8. **(P)** Implement `exponential_cdf(x, rate)` and `sample_inverse_cdf(inverse_cdf, n, rng)` (draw n uniforms and
   apply `inverse_cdf`) in `statkit/distributions.py`. Use them to draw 100,000 exponential samples with mean 8, and
   compare their histogram with the PDF.
9. **(P)** Fit an exponential to `commute_minutes` by setting the rate to 1/mean. Plot the data's ECDF and the fitted CDF
   on one figure. Does it fit?

## ★★★ Challenge

10. **(M)** Show that the median of an Exponential(λ) is ln(2)/λ, and that its variance is 1/λ². (Hint: E[X²] = 2/λ²,
    by integration by parts or from the gamma.)
11. **(P)** Use inverse-CDF sampling to draw from the density f(x) = 3x² on [0, 1]. Check that the sample mean is close to
    3/4.
12. **(P)** **The probability integral transform.** Draw 10,000 values from a Gamma(4, 2.5), apply that distribution's
    own CDF to them, and plot a histogram of the results. What shape do you get, and why? What would you see if you used
    the *wrong* CDF, say Exponential(mean 10)? (This is the idea behind calibration plots.)

Solutions: [solutions/solutions.md](solutions/solutions.md)
