# 07.03 — Hierarchical models

**Time:** about 5 hours · **Prerequisites:** 06.06 (mixed models), 07.01, 07.02

## Why this matters

Data often come from many related groups: schools, hospitals, products, users, countries, experiments. Treating each
group separately wastes information and overreacts to noise in small groups. Treating them as identical ignores real
differences. A **hierarchical** (multilevel) model does both at once: each group gets its own parameter, and those
parameters come from a shared distribution whose spread is estimated from the data. The model decides **how much to
pool**. It's the Bayesian version of Lesson 06.06's mixed models, it handles few groups and complex structures better,
and it underlies modern meta-analysis, small-area estimation, recommender systems and A/B testing across many segments.

## Learning objectives

- Contrast no pooling, complete pooling and partial pooling.
- Write a hierarchical model with hyperpriors, and fit it in PyMC.
- Interpret τ (the between-group spread) and the shrinkage of group estimates.
- Recognise the funnel geometry and fix divergences with a non-centred parameterisation.

## Before you start

A coaching programme raised SAT scores by an estimated 28 points in school A (standard error 15). The other seven
schools saw effects between −3 and 18. What's your best estimate of the *true* effect in school A?

---

## 1. The Eight Schools data

A classic dataset (Rubin, 1981): the estimated effect of SAT coaching in eight schools, each with a known standard error.

| School | A | B | C | D | E | F | G | H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| effect yⱼ | 28 | 8 | −3 | 7 | −1 | 1 | 18 | 12 |
| SE σⱼ | 15 | 10 | 16 | 11 | 9 | 11 | 10 | 18 |

Each school's estimate is very noisy: school A's 95% interval is [−1.4, 57.4].

## 2. Three ways to analyse it

**No pooling:** take each yⱼ at face value. School A's effect is 28. Every estimate keeps all its noise.

**Complete pooling:** assume one common effect and take the precision-weighted mean: **7.7 ± 4.1**. A test of
homogeneity gives χ² = 4.7 on 7 df, p = 0.70: the data are perfectly consistent with all schools having the same effect.
But "can't rule out equal" is not "equal".

**Partial pooling: the hierarchical model.**

$$y_j \sim N(\theta_j, \sigma_j^2), \qquad \theta_j \sim N(\mu, \tau^2), \qquad \mu \sim N(0, 5^2), \qquad \tau \sim \text{Half-Cauchy}(5).$$

- θⱼ: school j's true effect.
- μ: the average effect across schools; τ: how much true effects vary between schools.
- μ and τ are **hyperparameters** with their own priors (**hyperpriors**), and they're learned from all eight schools.

If τ is small, the θⱼ are pulled strongly together (close to complete pooling). If τ is large, they stay near their own
yⱼ (no pooling). The data decide.

## 3. Results

| School | Raw yⱼ | Posterior mean of θⱼ | 95% interval |
| --- | --- | --- | --- |
| A | 28 | **6.2** | [−2.9, 20.1] |
| B | 8 | 4.9 | [−4.3, 14.7] |
| C | −3 | 4.0 | [−6.7, 13.7] |
| D | 7 | 4.9 | [−4.2, 14.3] |
| E | −1 | 3.7 | [−6.6, 12.2] |
| F | 1 | 4.1 | [−5.9, 12.9] |
| G | 18 | 6.2 | [−2.3, 18.2] |
| H | 12 | 4.8 | [−5.2, 15.8] |

μ = 4.4; τ has posterior median **2.7** with 95% interval [0.14, 11.5].

Answer to "Before you start": school A's best estimate is about **6**, not 28. The data show the effects vary little
between schools relative to the noise, so each extreme estimate is mostly luck. P(school A's true effect > 20) = **0.03**
under the hierarchical model, against 0.70 if you trusted A's estimate alone (`eight_schools.png`).

The shrinkage is the same formula as Lesson 06.06, θ̂ⱼ ≈ wⱼyⱼ + (1 − wⱼ)μ with wⱼ = τ²/(τ² + σⱼ²), but now
**τ's uncertainty is averaged over** instead of plugged in. With only 8 groups, τ is very uncertain (its interval spans
0.1 to 11.5), and the Bayesian model carries that uncertainty into every θⱼ. That's its key advantage over the
point-estimate mixed models of Lesson 06.06, especially with few groups.

**The prior on μ matters here.** Normal(0, 5²), the prior used in the standard PyMC example, is fairly informative
relative to the data and pulls μ to 4.4, below the complete-pooling estimate of 7.7. Exercise 6 asks you to check the
sensitivity.

## 4. The funnel and the non-centred parameterisation

**Centred** model: θⱼ ~ N(μ, τ²) directly. When τ is small, the θⱼ must all lie very close to μ; when τ is large, they
can spread out. The posterior has a **funnel** shape in (log τ, θⱼ) (`eight_schools.png`, right). NUTS can't use one
step size for both the wide mouth and the narrow neck, so it produces **divergences** near the neck and fails to
explore small τ.

**Non-centred** model: write θⱼ = μ + τzⱼ with zⱼ ~ N(0, 1). It's mathematically the same model, but the sampler now
explores zⱼ, which has the same shape whatever τ is.

| Parameterisation | Divergences | ESS(τ) | R̂(τ) |
| --- | --- | --- | --- |
| centred | **137** | **32** | **1.10**: results not trustworthy |
| non-centred | **0** | 2,210 | 1.001 |

The rule of thumb: when groups are small or noisy (the data say little about each θⱼ), use the **non-centred** form.
With lots of data per group, the centred form can work better.

```python
with pm.Model():
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    z = pm.Normal("z", 0, 1, shape=8)
    theta = pm.Deterministic("theta", mu + tau * z)
    pm.Normal("y", theta, sigma, observed=y)
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)
```

## 5. Choosing priors for hierarchical models

- **Group-level sd τ:** use a half-normal, half-Student-t or half-Cauchy prior that allows 0 and moderate values.
  Avoid "uninformative" inverse-gamma priors with tiny parameters, which can distort the results when groups are few.
- **Scale:** priors should be on a sensible scale for the data. Standardise outcomes, or reason in the data's units
  ("effects of more than about 50 SAT points are implausible").
- **Check with simulations** of the prior predictive distribution (Lesson 07.04).

## 6. Where hierarchical models are used

| Field | Groups | What gets pooled |
| --- | --- | --- |
| meta-analysis | studies | effect estimates (the Eight Schools is a meta-analysis) |
| A/B testing | segments, countries | the treatment effect per segment |
| sport | players | batting averages, shooting percentages |
| health | hospitals, regions | mortality rates, disease prevalence |
| recommender systems | users and items | user preferences and item quality |
| ML experiments | datasets, seeds | model performance differences |

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Reporting the noisiest group's raw estimate as "the best" | Partial pooling |
| Concluding "all groups are equal" from a non-significant homogeneity test | Estimate τ, with its uncertainty |
| Ignoring divergences | Non-centred parameterisation, higher `target_accept` |
| Vague inverse-gamma priors on variances | Half-normal or half-Cauchy on the sd |
| Too few groups to learn τ, without saying so | Report τ's wide interval; consider an informative prior |

## Summary

- No pooling (each group alone), complete pooling (all equal), partial pooling (hierarchical).
- θⱼ ~ N(μ, τ²): τ controls the amount of pooling and is learned from the data.
- Shrinkage is strongest for noisy groups; Bayesian models average over τ's uncertainty.
- Funnels cause divergences; fix them with θⱼ = μ + τzⱼ (non-centred).

## Self-test

1. What happens to the θⱼ estimates as τ → 0? As τ → ∞?
2. Why does school A shrink more than school E, relatively?
3. What is a divergence, and why does the centred model have so many?
4. Give an example from your own field where partial pooling would help.

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
