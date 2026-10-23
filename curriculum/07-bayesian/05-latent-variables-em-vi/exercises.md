# 07.05 Exercises — Latent variables, EM and variational inference

`make lesson L=07-bayesian/05-latent-variables-em-vi`

## ★ Warm-up

1. **(M)** A two-component mixture has π = (0.7, 0.3), means (0, 4) and sds (1, 1). Compute the responsibility of
   component 2 for x = 2 and x = 3.
2. **(C)** What's the difference between the soft assignments of EM and the hard assignments of k-means? When does it
   matter?
3. **(C)** Write the ELBO identity, and explain in words why maximising the ELBO makes q close to the posterior.

## ★★ Core

4. **(P)** Implement `em_gaussian_mixture(x, k=2, iterations=500, rng=None, n_init=5)` in `statkit/bayes.py`. Return
   (weights, means, sds, log-likelihood) for the best of n_init random starts (initial means drawn from the data).
   Check it on the lesson's data.
5. **(P)** Fit mixtures with k = 1, 2, 3 and 4 components to the lesson's study-hours data (sklearn `GaussianMixture`).
   Compare their BIC. Which k does it choose?
6. **(P)** Apply your EM to `commute_minutes` from `students.csv` with k = 2. Do the components mean anything? (Remember
   how commute was generated.) What does this tell you about over-interpreting mixture fits?
7. **(P)** In the ADVI example, switch to `method="fullrank_advi"`. Does it recover the b₀–b₁ correlation and the NUTS sd?
   Then standardise `hours` before fitting with mean-field ADVI. Why does that help?
8. **(P)** Fit a GP to `exam_score` vs `study_hours` (scikit-learn, RBF + WhiteKernel, `normalize_y=True`). Plot the mean
   and 95% band. Is the fitted function close to linear? Compare its predictions at 5, 15 and 30 hours with the linear
   regression of Lesson 06.01.

## ★★★ Challenge

9. **(M)** Derive the M-step update for μ_k by maximising Σᵢ Σ_k rᵢₖ log[π_k N(xᵢ | μ_k, σ_k²)] with respect to μ_k.
10. **(M)** For the model θ ~ N(0, 1), yᵢ ~ N(θ, 1), compute the ELBO for q = N(m, s²) in closed form, and show that it's
    maximised at the exact posterior.
11. **(P)** **Missing data with EM.** Generate 300 bivariate-normal points (ρ = 0.7) and delete 30% of the second
    coordinates at random. Estimate the means, variances and correlation by EM (E-step: fill in the expected missing
    values and their squares from the conditional normal; M-step: recompute the moments). Compare with simply dropping
    incomplete rows.

Solutions: [solutions/solutions.md](solutions/solutions.md)
