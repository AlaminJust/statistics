# 05.01 Exercises — Estimators, bias and variance

`make lesson L=05-inference/01-estimators-bias-variance`

## ★ Warm-up

1. **(M)** Estimator A has bias 0.5 and variance 1; estimator B has bias 0 and variance 1.5. Which has the lower MSE?
2. **(M)** X₁, X₂ are i.i.d. with mean μ and variance σ². Show that θ̂ = (X₁ + 2X₂)/3 is unbiased for μ, and find its
   variance. Is it better or worse than X̄?
3. **(C)** Explain bias and variance with the dartboard picture. Draw the four combinations (low/high bias × low/high
   variance).

## ★★ Core

4. **(M)** For i.i.d. Bernoulli(p) data, show that p̂ = X̄ is unbiased and find its MSE. What is the MSE when p = 0.5 and
   n = 100?
5. **(M)** Show that the "plus four" estimator p̃ = (X + 2)/(n + 4), where X is the number of successes, is biased. Find
   its bias as a function of p and n. For which p does it have lower MSE than p̂ when n = 10? (Compute the MSE of both
   on a grid of p, in code.)
6. **(P)** Repeat worked example 2 with Normal(5, 1) data and with Laplace data (`rng.laplace(5, 1, ...)`). Which of
   mean, median and midrange wins in each case?
7. **(P)** Simulate the bias and MSE of the sample standard deviation s as an estimator of σ = 2 for n = 2, 5, 10, 50.
   Does the bias vanish as n grows? Is s consistent?
8. **(P)** Implement `standard_error(values)` (= s/√n) in `statkit/inference.py`. Use it to report the mean exam score
   of each program with its SE.

## ★★★ Challenge

9. **(M)** Prove that E[max of n Uniform(0, θ)] = θ n/(n + 1). (Hint: P(max ≤ x) = (x/θ)ⁿ; differentiate to get the
   density, then integrate x times it.)
10. **(M)** For the shrinkage estimator c·X̄, show MSE(c) = (1 − c)²μ² + c²σ²/n, and find the c that minimises it.
11. **(P)** **James–Stein in action.** Generate 10 true means μᵢ ~ N(0, 1), and for each one observation
    xᵢ ~ N(μᵢ, 1). Compare the total squared error Σ(xᵢ − μᵢ)² of the raw estimates with that of the James–Stein
    estimate (1 − (k − 2)/Σxᵢ²)·xᵢ, where k = 10, averaged over 10,000 repetitions. Which wins, and by how much?

Solutions: [solutions/solutions.md](solutions/solutions.md)
