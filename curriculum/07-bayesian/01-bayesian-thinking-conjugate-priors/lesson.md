# 07.01 — Bayesian thinking and conjugate priors

**Time:** about 4 hours · **Prerequisites:** 03.04, 04.01–04.03, 05.02, 05.03

## Why this matters

Frequentist methods treat the parameter as fixed and the data as random, which leads to careful but awkward statements
like "95% of such intervals would contain the truth". **Bayesian inference** treats the unknown parameter as uncertain,
describes that uncertainty with a probability distribution, and updates it with data using Bayes' theorem (Lesson
03.04). The result answers the questions people actually ask: "What's the probability that version B is better?" "What
range of values is plausible?" It handles small data and prior knowledge naturally, and its ideas power modern ML:
regularisation as priors, uncertainty in deep learning, VAEs and Bayesian optimisation.

## Learning objectives

- Write posterior ∝ likelihood × prior for a parameter.
- Update Beta–Binomial, Normal–Normal and Gamma–Poisson models by hand.
- Summarise a posterior with its mean, mode (MAP), credible intervals and probabilities of hypotheses.
- Explain how the prior's influence shrinks as data grow.
- Compute posterior predictive distributions.
- Run a Bayesian A/B test and compare it with the frequentist analysis.

## Before you start

A coin landed heads 7 times in 10. What is the probability that the coin favours heads (p > 0.5)? A frequentist can't
answer that directly. Guess the Bayesian answer.

---

## 1. Bayes' theorem for parameters

$$\underbrace{p(\theta \mid \text{data})}_{\text{posterior}} = \frac{\overbrace{p(\text{data} \mid \theta)}^{\text{likelihood}}\;\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\text{data})}_{\text{evidence}}} \;\propto\; \text{likelihood} \times \text{prior}.$$

- The **likelihood** is the same as in Lesson 05.02.
- The **prior** encodes what you believe *before* seeing the data. It can be vague ("anything between 0 and 1"),
  informative ("about 0.5, give or take 0.1"), or regularising ("small values are more likely").
- The **posterior** combines both. It's a full distribution: you can read off a best guess, an interval, or the
  probability of any statement about θ.
- The evidence p(data) = ∫ likelihood × prior dθ only normalises, so it can usually be ignored, except for comparing
  models.

## 2. Beta–Binomial: the conjugate pair for a proportion

If the prior is **Beta(a, b)** and you observe k successes in n trials, the posterior is

$$\text{Beta}(a + k,\; b + n - k).$$

The prior's form is preserved (*conjugacy*): just add successes to a and failures to b. You can read a and b as
"pseudo-counts" of prior successes and failures.

*Derivation:* likelihood × prior ∝ pᵏ(1 − p)ⁿ⁻ᵏ × pᵃ⁻¹(1 − p)ᵇ⁻¹ = p^(a+k−1)(1 − p)^(b+n−k−1), which is the kernel of a
Beta(a + k, b + n − k).

**Worked example 1: 7 heads in 10, flat prior Beta(1, 1).** Posterior **Beta(8, 4)**:

| Summary | Value |
| --- | --- |
| posterior mean (a/(a + b)) | **0.667** |
| posterior mode = **MAP** estimate | 0.700, equal to the MLE (a flat prior adds nothing) |
| 95% **credible interval** | **[0.390, 0.891]** |
| P(p > 0.5 \| data) | **0.887** |

Answer to "Before you start": about **89%**. A numerical grid (likelihood × prior, normalised by integration) gives the
same posterior mean, 0.6667: conjugacy is a shortcut, not magic.

**A credible interval means what people want a confidence interval to mean:** given the model and the prior, there's
a 95% probability that p lies in [0.39, 0.89] (compare Lesson 05.03).

## 3. How much does the prior matter?

**Worked example 2.** Three priors and three amounts of data, all with 70% heads:

| Prior | 7/10 | 70/100 | 700/1000 |
| --- | --- | --- | --- |
| flat Beta(1, 1) | 0.667 [0.39, 0.89] | 0.696 [0.60, 0.78] | 0.700 [0.67, 0.73] |
| sceptical Beta(20, 20) | **0.540** [0.40, 0.68] | 0.643 [0.56, 0.72] | 0.692 [0.66, 0.72] |
| optimistic Beta(8, 2) | 0.750 [0.54, 0.91] | 0.709 [0.62, 0.79] | 0.701 [0.67, 0.73] |

With little data, the prior matters a lot; with lots of data, **everyone ends up in the same place**. A sceptical prior
acts like 40 earlier flips split 20–20.

**The posterior mean is a weighted average** (worked example 3). With Beta(a, b) and n observations:

$$E[p \mid \text{data}] = \underbrace{\frac{n}{a + b + n}}_{w}\,\hat{p} + (1 - w)\,\frac{a}{a + b}.$$

With Beta(20, 20) and n = 10, w = 0.2: 0.2 × 0.7 + 0.8 × 0.5 = **0.54**. That's shrinkage again (Lessons 05.01 and
06.06): the prior pulls noisy estimates towards a sensible centre.

**Sequential updating** (worked example 4): updating flip by flip, or all at once, gives the same posterior,
Beta(37, 15) after 50 flips. Today's posterior is tomorrow's prior.

## 4. The posterior predictive distribution

To predict new data, average the model's predictions over the posterior instead of plugging in a single estimate:

$$p(\tilde{y} \mid \text{data}) = \int p(\tilde{y} \mid \theta)\, p(\theta \mid \text{data})\, d\theta.$$

By simulation: draw θ from the posterior, then draw ỹ given θ, and repeat.

**Worked example 5.** P(at least 4 heads in the next 5 flips): posterior predictive **0.484**; plug-in with p̂ = 0.7:
0.528. The plug-in answer pretends p is known exactly. The posterior predictive accounts for the fact that p could
plausibly be anywhere from 0.4 to 0.9, which widens the predictions.

## 5. Normal–Normal: a mean with known σ

Prior μ ~ N(m₀, s₀²); data x₁…xₙ ~ N(μ, σ²) with σ known. The posterior is normal with

$$\frac{1}{s_n^2} = \frac{1}{s_0^2} + \frac{n}{\sigma^2}, \qquad m_n = s_n^2\left(\frac{m_0}{s_0^2} + \frac{n\bar{x}}{\sigma^2}\right).$$

**Precisions** (inverse variances) add, and the posterior mean is a precision-weighted average of the prior mean and
x̄.

**Worked example 6: mean exam score.** Prior N(50, 10²); data x̄ = 63.93, n = 200, σ = 14. Posterior N(63.79, 0.985²),
95% credible interval **[61.86, 65.73]**, almost identical to the frequentist interval [61.99, 65.87]. With 200
observations the data overwhelm a vague prior.

**Gamma–Poisson** (worked example 8): with a Gamma(α, rate β) prior on a Poisson rate, the posterior is
Gamma(α + Σyᵢ, β + n). For 77 goals in 38 matches with a Gamma(2, 1) prior: mean 2.03, 95% CI [1.60, 2.50].

## 6. A Bayesian A/B test

**Worked example 7** on `data/ab_test.csv` (the data of the m05 project). With Beta(1, 1) priors on each conversion
rate, the posteriors are Beta(1 + conversions, 1 + non-conversions). Draw 200,000 samples of each and compare:

| Question | Answer |
| --- | --- |
| P(B converts better than A) | **0.998** |
| lift B − A: mean and 95% credible interval | +1.21 points [0.37, 2.04] |
| P(lift > 0.5 points), the business threshold | **0.95** |
| expected loss if we ship B | 3 × 10⁻⁶ (vs 0.012 if we keep A) |

These statements go straight into a decision. Note that the frequentist test (p ≈ 0.005) gave a similar *conclusion*.
The difference is in what you can *say*.

**Caution:** Bayesian results don't protect you from peeking at results repeatedly (Lesson 05.08) if you then make
frequentist-style claims about error rates. And the result is only as good as the model and prior.

## 7. Frequentist vs Bayesian, honestly

| | Frequentist | Bayesian |
| --- | --- | --- |
| parameter | fixed, unknown | random (uncertain) |
| output | estimates, CIs, p-values | the posterior distribution |
| interval meaning | 95% of such intervals cover θ | 95% probability θ is in it (given the model and prior) |
| prior knowledge | not used formally | built in (and must be justified) |
| computation | often formulas | often MCMC (Lesson 07.02) |
| with lots of data and vague priors | usually the **same numbers** | |

Neither is "right". Good statisticians use both, and report the assumptions behind each.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Treating the prior as a nuisance to hide | State it and justify it, and check sensitivity (Lesson 07.04) |
| A prior that rules values out (zero density) | They can never be recovered by any amount of data |
| Reporting only the posterior mean | Report intervals and decision-relevant probabilities |
| Predicting with the plug-in estimate | Use the posterior predictive |
| "Bayesian, so peeking is fine" | Optional stopping still matters if you care about error rates |

## Summary

- posterior ∝ likelihood × prior.
- Beta(a, b) + k of n → Beta(a + k, b + n − k); Normal–Normal: precisions add; Gamma–Poisson: Gamma(α + Σy, β + n).
- Posterior mean = weighted average of the prior mean and the data estimate; the prior's weight shrinks as n grows.
- Credible intervals give direct probability statements; MAP with a flat prior = MLE.
- Posterior predictive: average predictions over the posterior.
- A/B tests: P(B > A), the lift distribution, expected loss.

## Self-test

1. Prior Beta(2, 2), data 3 successes in 12. What is the posterior? Its mean?
2. Why does a Beta(20, 20) prior pull estimates towards 0.5?
3. What's the difference between a 95% credible interval and a 95% confidence interval?
4. Why is the posterior predictive wider than the plug-in prediction?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
