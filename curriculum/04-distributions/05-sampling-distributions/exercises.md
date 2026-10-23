# 04.05 Exercises — Sampling distributions

`make lesson L=04-distributions/05-sampling-distributions`

## ★ Warm-up

1. **(M)** A sample of 16 has s = 12. Find the estimated standard error of the mean and the t critical value for a 95%
   interval.
2. **(M)** Give the mean and variance of a χ² distribution with 20 df.
3. **(C)** Explain the difference between a standard deviation and a standard error, using `exam_score` as an example
   (sd 13.84, n = 200).

## ★★ Core

4. **(M)** Group A (n = 50) has s = 10 and group B (n = 40) has s = 12. Find the standard error of x̄_A − x̄_B.
5. **(M)** A survey of 600 people finds p̂ = 0.42. Estimate the standard error of p̂. How many people would halve it?
6. **(M)** For a t statistic with 5 df, find P(T > 2.5) and compare with P(Z > 2.5). By what factor does the normal
   understate the tail?
7. **(M)** Normal data, n = 10, s² = 196. Using (n − 1)s²/σ² ~ χ²₉, find the interval of σ² values for which the
   observed s² is not in the outer 2.5% tails. (This is a 95% confidence interval for σ², Module 05.) How wide is it?
8. **(P)** Simulate the sampling distribution of the sample *range* (max − min) for n = 10 standard normals. What is its
   standard error? Is its distribution symmetric?
9. **(P)** Treat `students.csv` as a population. Draw 10,000 samples of size 25 *without replacement*, and for each
   compute the t statistic (x̄ − μ)/(s/5) with μ = the population mean of `exam_score`. Compare the share with |t| > 2.064
   (t₂₄ critical value) with 0.05.

## ★★★ Challenge

10. **(M)** Show that if T ~ t_k then T² ~ F(1, k), using the definitions of t and F.
11. **(P)** **Robustness of t.** Repeat the lesson's "Before you start" simulation (n = 10, using s, critical value 2.262)
    with data drawn from (a) a uniform, (b) an exponential and (c) a t distribution with 3 df instead of the normal. How
    far does the false-positive rate move from 5% in each case?
12. **(P)** Estimate by simulation the standard error of the sample correlation r for n = 30 bivariate-normal pairs with
    ρ = 0, 0.5 and 0.9. Compare with the approximation (1 − ρ²)/√n. Why does r get *more* precise as |ρ| grows?

Solutions: [solutions/solutions.md](solutions/solutions.md)
