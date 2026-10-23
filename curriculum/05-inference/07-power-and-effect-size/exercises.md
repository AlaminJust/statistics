# 05.07 Exercises — Power and effect size

`make lesson L=05-inference/07-power-and-effect-size`

## ★ Warm-up

1. **(M)** Compute Cohen's d for two groups with means 72 and 66 and pooled sd 12. How would you describe it?
2. **(M)** Using n ≈ 2(z_{α/2} + z_β)²/d², find n per group for 90% power to detect d = 0.4 at α = 0.05. Check with
   statsmodels.
3. **(C)** A study with 15 people per group finds p = 0.30. What would you tell the authors? What power did they have
   for a medium effect (d = 0.5)?

## ★★ Core

4. **(M)** Compare the sample size needed to detect d = 0.3 with 80% power (a) in a two-group design and (b) in a paired
   design where the *differences* have effect size 0.3. Why is the paired design cheaper?
5. **(M)** How does n change if you (a) use α = 0.01 instead of 0.05, or (b) want 90% power instead of 80%, for d = 0.5?
6. **(P)** An app's crash rate is 5%. A fix should lower it to 4%. How many users per arm does an A/B test need for 80% power?
7. **(P)** Implement `cohens_d(x, y)` (pooled sd) and `sample_size_two_means(delta, sd, alpha=0.05, power=0.8)` (the
   normal-approximation formula, rounded up) in `statkit/inference.py`.
8. **(P)** **Explain a Type II error.** In Lesson 05.05, exercise 7, sleep's correlation with score (r = 0.128, n = 200)
   was not significant, even though sleep truly affects scores in `make_students.py`. Using Fisher's z (atanh(r) has SE
   1/√(n − 3)), compute the power that n = 200 had to detect r = 0.128, and the n needed for 80% power.
9. **(P)** Write a simulation-based power function for the **Mann–Whitney** test with log-normal data (σ = 1) where
   group B is multiplied by 1.3. Find the n per group that gives 80% power.

## ★★★ Challenge

10. **(P)** Reproduce the winner's-curse simulation for n = 20, 50, 100 and 400 per group (true d = 0.2). Plot the average
    significant estimate against n, with a horizontal line at the truth. At what n does the exaggeration largely
    disappear?
11. **(M)** Derive the sample-size formula. Under H₁ the test statistic is approximately N(d√(n/2), 1). Power is
    P(statistic > z_{1−α/2}) (ignoring the other tail); set it equal to 1 − β and solve for n.
12. **(C)** A colleague runs an A/B test with 1,000 users per arm, sees no significant difference, and concludes "the
    new design doesn't matter". Using the minimum-detectable-effect idea, write a two-paragraph reply.

Solutions: [solutions/solutions.md](solutions/solutions.md)
