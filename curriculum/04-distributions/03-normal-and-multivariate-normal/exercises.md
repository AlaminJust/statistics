# 04.03 Exercises — The normal and the multivariate normal

`make lesson L=04-distributions/03-normal-and-multivariate-normal`

## ★ Warm-up

1. **(M)** Heights ~ N(170, 8²) cm. Find the share taller than 186 cm, the share between 160 and 180 cm, and the 25th and
   95th percentiles. Use z-scores first, then check with scipy.
2. **(M)** X ~ N(5, 9) and Y ~ N(3, 16), independent. Give the distribution of X − Y and P(X > Y).
3. **(C)** Without calculating: which is more unusual, a single adult over 2 m tall or a basketball team of five whose
   *average* height is over 2 m? Why?

## ★★ Core

4. **(M)** Cereal boxes are filled with weight ~ N(μ, 4²) grams. With μ = 500, what share weigh under 490 g? What μ makes
   only 1% of boxes weigh under 500 g?
5. **(M)** A student's three exam scores are independent N(20, 3²). P(total > 66)?
6. **(M)** A survey asks 400 people a yes/no question when the true "yes" share is 20%. Use the normal approximation (with
   continuity correction) to find P(70 or fewer say yes), and compare with the exact binomial.
7. **(M)** Fathers' and sons' heights are bivariate normal: fathers mean 175, sons mean 177, both sd 7, correlation 0.5.
   A father is 189 cm tall. Find the expected height of his son, the conditional sd, and P(son > 189). Explain the
   result in terms of regression to the mean.
8. **(P)** Implement `normal_pdf(x, mu, sigma)` and `normal_cdf(x, mu, sigma)` in `statkit/distributions.py`, using
   Φ(z) = (1 + erf(z/√2))/2 with `math.erf`. Check against scipy at several points.
9. **(P)** Make Q–Q plots for `study_hours`, `log(study_hours)` and `sqrt(study_hours)`. Which transformation makes it
   closest to normal?

## ★★★ Challenge

10. **(P)** Implement `normal_ppf(q, mu, sigma)` by bisection on your `normal_cdf` (search z in [−10, 10] until the
    interval is shorter than 1e-10). Check that `normal_ppf(0.975)` ≈ 1.959964.
11. **(M)** For a bivariate normal, show that the conditional-mean formula gives a slope equal to Cov(X, Y)/Var(X). In
    Module 06 this becomes the least-squares slope.
12. **(P)** Generate 3-D normal data with covariance [[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]] using your own Cholesky-based
    sampler (`np.linalg.cholesky` is allowed). Verify the sample covariance matrix, and show that a linear combination
    w = (1, −1, 2) has variance **w**ᵀΣ**w**.

Solutions: [solutions/solutions.md](solutions/solutions.md)
