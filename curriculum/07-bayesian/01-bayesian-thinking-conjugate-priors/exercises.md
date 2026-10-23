# 07.01 Exercises — Bayesian thinking and conjugate priors

`make lesson L=07-bayesian/01-bayesian-thinking-conjugate-priors`

## ★ Warm-up

1. **(M)** Prior Beta(2, 2); data 3 successes in 12. Give the posterior, its mean, its mode (MAP) and a 95% credible
   interval (scipy).
2. **(C)** Explain the difference between a 95% credible interval and a 95% confidence interval to a manager in three
   sentences.
3. **(M)** Show that the Beta(a, b) prior is equivalent to having seen a − 1 successes and b − 1 failures before, in the
   sense that the posterior *mode* equals the MLE computed on the pooled counts.

## ★★ Core

4. **(P)** A small drug trial: 15 of 20 treated patients and 9 of 20 controls recover. With Beta(1, 1) priors, compute
   P(treatment recovery rate > control rate) and a 95% credible interval for the difference.
5. **(M)** A factory inspects 50 items and finds 0 defective. With a Beta(1, 1) prior, give the posterior mean defect
   rate and the 95% upper credible bound. Compare with the frequentist "rule of three" (3/n).
6. **(M)** A person's IQ prior is N(100, 15²). One test with measurement sd 5 gives 130. Compute the posterior mean and
   sd. Why isn't the best estimate 130?
7. **(M)** An ad has 3 clicks in 100 impressions. Using a Beta(2, 98) prior (similar ads average 2%), find the posterior
   mean click rate and 95% interval. Compare with the raw rate.
8. **(P)** Implement `beta_binomial_update(alpha, beta, successes, failures)` → (alpha', beta') and
   `normal_normal_update(prior_mean, prior_sd, data, sigma)` → (posterior mean, posterior sd) in `statkit/bayes.py`.
9. **(P)** In `students.csv`, with Beta(1, 1) priors, compute P(tutored students are more likely than others to score
   ≥ 70) and plot the two posteriors.

## ★★★ Challenge

10. **(M)** Derive the Normal–Normal posterior by completing the square in the exponent of likelihood × prior.
11. **(P)** **Prior sensitivity.** For the drug trial in exercise 4, compute P(treatment better) under priors Beta(1, 1),
    Beta(0.5, 0.5), Beta(5, 5) and Beta(20, 20) for both arms. How robust is the conclusion?
12. **(P)** **Thompson sampling.** Three ad variants have true click rates 0.04, 0.05 and 0.06. Each round, draw one sample
    from each variant's Beta posterior, show the variant with the largest draw, observe a click (or not) and update.
    Run 10,000 rounds and plot how often each variant is shown over time. Why does this balance exploration and
    exploitation automatically?

Solutions: [solutions/solutions.md](solutions/solutions.md)
