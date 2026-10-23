# 07.05 — Latent variables, EM and variational inference

**Time:** about 5 hours · **Prerequisites:** 04.06 (KL), 05.02 (MLE), 07.01–07.04

## Why this matters

Many models contain **latent** (hidden) variables: which group a data point belongs to, a topic behind a document, the
"code" behind an image. Fitting them leads to two of the most important algorithms in ML. The **EM algorithm**
alternates between guessing the hidden variables and updating the parameters; it fits Gaussian mixtures, hidden Markov
models and models with missing data. **Variational inference** (VI) replaces sampling with optimisation: it finds the
simple distribution closest (in KL) to the posterior by maximising the **ELBO**, the objective behind variational
autoencoders and scalable Bayesian deep learning. This lesson also introduces **Gaussian processes**, Bayesian priors
over whole functions.

## Learning objectives

- Write a mixture model with latent group labels.
- Derive and implement the EM algorithm's E-step and M-step for a Gaussian mixture.
- Derive the ELBO and explain log p(x) = ELBO + KL(q ‖ posterior).
- Run ADVI in PyMC and explain the limits of mean-field VI.
- Fit a Gaussian process regression and interpret its uncertainty.

## Before you start

A histogram of weekly study hours has two humps. You suspect two kinds of student (casual and intensive), but nobody
recorded who's who. How could you estimate each group's average, and the share of each type?

---

## 1. A mixture model with latent labels

Each student i belongs to a hidden group zᵢ ∈ {1, 2} with probabilities π₁ and π₂. Given the group,
xᵢ ~ N(μ_k, σ_k²). The marginal density is a **mixture**:

$$p(x) = \pi_1 N(x \mid \mu_1, \sigma_1^2) + \pi_2 N(x \mid \mu_2, \sigma_2^2).$$

If we knew the zᵢ, fitting would be easy (the group means and sds). If we knew the parameters, we could compute each
zᵢ's probability by Bayes' theorem. EM alternates between the two.

## 2. The EM algorithm

**E-step:** for each point, compute the **responsibility** of each component, the posterior probability that the point
came from it:

$$r_{ik} = \frac{\pi_k\, N(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j\, N(x_i \mid \mu_j, \sigma_j^2)}.$$

**M-step:** refit each component by weighted MLE, with the responsibilities as weights:

$$\pi_k = \frac{1}{n}\sum_i r_{ik}, \qquad \mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \qquad \sigma_k^2 = \frac{\sum_i r_{ik}(x_i - \mu_k)^2}{\sum_i r_{ik}}.$$

Repeat until the log-likelihood stops increasing. **EM never decreases the log-likelihood** (section 4 shows why).

**Worked example 1 (Before you start).** 500 students; 35% "intensive" with hours ~ N(18, 3²), the rest
N(8, 2²); labels hidden. EM converges in 49 iterations, with the log-likelihood rising from −1826.5 to −1421.9:

| | Estimate | Truth |
| --- | --- | --- |
| weights | 0.672 / 0.328 | 0.65 / 0.35 |
| means | 8.11 / 18.05 | 8 / 18 |
| sds | 2.01 / 3.31 | 2 / 3 |

98.6% of students are assigned to their true group. A student who studies 13 hours has P(intensive) = 0.68: a *soft*
assignment that reflects the overlap. scikit-learn's `GaussianMixture` gives the same answer.

**Worked example 2: local optima.** Start with two *identical* means and EM can't break the symmetry: both components
stay at 11.37, with a log-likelihood of −1543.4 against −1421.9 for the good solution. EM finds a *local* maximum.
**Run it from several random starts and keep the best** (`GaussianMixture(n_init=10)`).

k-means clustering is the limiting case of EM with hard assignments and equal, shrinking variances.

## 3. The ELBO

For a model with latent θ (or z) and data x, take *any* distribution q(θ). Then

$$\log p(x) = \underbrace{E_q[\log p(x, \theta)] - E_q[\log q(\theta)]}_{\text{ELBO}(q)} + \mathrm{KL}\big(q(\theta) \,\|\, p(\theta \mid x)\big).$$

*Derivation:* write log p(x) = E_q[log p(x, θ) − log p(θ | x)], then add and subtract E_q[log q(θ)].

Because KL ≥ 0 (Lesson 04.06), the ELBO is a **lower bound** on the log evidence, hence the name *evidence lower bound*.
Maximising the ELBO over q is the same as minimising KL(q ‖ posterior).

**Worked example 3, an exact check.** Model θ ~ N(0, 1), yᵢ | θ ~ N(θ, 1), 10 observations; log evidence −12.025.

| q | ELBO | gap = KL(q ‖ posterior) |
| --- | --- | --- |
| the exact posterior | −12.025 | 0.000 |
| right width, wrong mean | −13.404 | 1.379 |
| right mean, too wide | −15.803 | 3.778 |

**EM is coordinate ascent on the ELBO:** the E-step sets q(z) to the exact posterior of the latent labels (making the KL
zero), and the M-step maximises the ELBO over the parameters. Each step can only raise the ELBO, and at the end of each
E-step the ELBO equals the log-likelihood, so the log-likelihood never decreases.

**VAEs** use exactly this objective. With an encoder q(z | x) and a decoder p(x | z),
ELBO = E_q[log p(x | z)] − KL(q(z | x) ‖ p(z)). The first term is the reconstruction loss; the second is the KL
regulariser towards the N(0, I) prior, computed with the normal KL formula of Lesson 04.06.

## 4. Variational inference in practice: ADVI

When the posterior is too expensive to sample, choose a simple family for q, usually **mean-field**: independent
normals for every parameter. Then maximise the ELBO by stochastic gradient ascent. PyMC's ADVI does this automatically:

```python
approx = pm.fit(30_000, method="advi")
draws = approx.sample(4000)
```

**Worked example 4: NUTS vs ADVI on a logistic regression** (300 students, pass ~ study hours; true slope 0.4):

| Method | b₁ mean | b₁ sd | corr(b₀, b₁) |
| --- | --- | --- | --- |
| NUTS (exact up to Monte Carlo error) | 0.481 | 0.058 | **−0.96** |
| mean-field ADVI | 0.366 | **0.015** | −0.02 |

Mean-field ADVI assumes b₀ and b₁ are independent, but they are strongly negatively correlated (a steeper slope must be
offset by a lower intercept). Unable to represent that, it settles for a narrow approximation: the **sd is 4 times too
small** and the mean is shifted. This is the mode-seeking, uncertainty-underestimating behaviour of reverse KL that
Lesson 04.06 predicted.

| | MCMC (NUTS) | VI (ADVI) |
| --- | --- | --- |
| accuracy | asymptotically exact | approximate (often too confident) |
| speed | slower | fast, scales to big data and big models |
| diagnostics | R̂, ESS, divergences | ELBO convergence only |
| use when | you need reliable uncertainty | you need speed (huge data, deep models), or an initial exploration |

Full-rank VI (`method="fullrank_advi"`) captures correlations at a higher cost. Normalising flows and other richer
families go further.

## 5. Gaussian processes: priors over functions

A **Gaussian process** (GP) is a distribution over functions: any finite set of function values is multivariate normal
(Lesson 04.03), with covariances given by a **kernel**. The RBF kernel k(x, x′) = exp(−(x − x′)²/(2ℓ²)) says that nearby
inputs have similar outputs; the length-scale ℓ says how quickly that similarity decays. Conditioning on data uses the
multivariate-normal conditional formulas, which gives a posterior mean and an uncertainty band at every x.

**Worked example 5.** 25 noisy points from sin(x) + 0.3x on [0, 10]. The learned kernel has length-scale 1.63 and noise
level 0.025. At x = 5 (inside the data) the prediction is **0.43 ± 0.40**; at x = 12 (outside) it's **1.40 ± 1.90**. The
GP *knows* it doesn't know beyond the data (`em_and_gp.png`). GPs power **Bayesian optimisation**, the standard method for
tuning expensive ML hyperparameters, and they're useful whenever you have few, expensive observations.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Running EM once | Several random starts; keep the best log-likelihood |
| Hard-assigning points whose responsibility is near 0.5 | Keep soft assignments; they encode real uncertainty |
| Trusting mean-field VI intervals | Check against MCMC on a subset, or use full-rank VI |
| Reading the ELBO as a log-likelihood | It's a lower bound; the gap is KL(q ‖ posterior) |
| GP extrapolation as if it were knowledge | Look at the band: far from data the GP reverts to its prior |

## Summary

- Mixtures: hidden labels z; p(x) = Σ π_k N(x | μ_k, σ_k²).
- EM: E-step = responsibilities (posterior of z); M-step = weighted MLE; the likelihood never decreases; local optima.
- log p(x) = ELBO + KL(q ‖ posterior); maximising the ELBO = best approximation q.
- ADVI: fast, mean-field, often overconfident; MCMC is slower but reliable.
- VAEs maximise an ELBO; GPs give calibrated uncertainty that grows away from the data.

## Self-test

1. What do the responsibilities rᵢₖ mean?
2. Why can't EM decrease the log-likelihood?
3. What does the gap between log p(x) and the ELBO measure?
4. Why does mean-field VI underestimate uncertainty when parameters are correlated?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
