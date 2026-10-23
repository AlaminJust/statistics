# 04.06 — Entropy, cross-entropy and KL divergence

**Time:** about 4 hours · **Prerequisites:** 01.02, 03.05, 04.01–04.03

## Why this matters

Open any deep-learning code and you'll find `CrossEntropyLoss`. Read about VAEs, diffusion models, knowledge
distillation or RLHF, and you'll meet the **KL divergence**. These come from information theory, which measures
uncertainty and how badly one probability distribution approximates another. This lesson shows that they are nothing
exotic: cross-entropy loss is exactly the negative log-likelihood of Lesson 05.02, and minimising it means making the
model's distribution as close as possible to the data's.

## Learning objectives

- Compute and interpret entropy in bits and nats.
- Compute cross-entropy and KL divergence, and explain why cross-entropy = entropy + KL.
- Show that binary cross-entropy loss is the negative Bernoulli log-likelihood.
- Explain the asymmetry of KL, and the difference between forward and reverse KL.
- Compute mutual information from a joint distribution.

## Before you start

Which is more "uncertain": a fair coin, or a die that shows 1 half the time and each other face 10% of the time? How
would you measure it?

---

## 1. Entropy: average surprise

The **surprise** (information content) of an outcome with probability p is −log p: rare outcomes are surprising, and
certain ones (p = 1) carry no surprise. **Entropy** is the average surprise:

$$H(p) = -\sum_x p(x)\log p(x).$$

With log base 2 the unit is **bits**; with natural logs (ln) it's **nats**. ML code uses nats. Terms with p(x) = 0 count
as 0.

**Worked example 1.**

| Distribution | Entropy (bits) |
| --- | --- |
| certain outcome | 0 |
| coin with P(heads) = 0.9 | 0.469 |
| fair coin | **1** |
| loaded die (0.5, then 0.1 each) | 2.161 |
| fair die | 2.585 = log₂ 6 |
| fair 8-sided die | 3 = log₂ 8 |

Interpretation: entropy is the average number of yes/no questions needed to identify the outcome with an optimal
strategy (and the shortest average code length, in bits per symbol). A fair 8-sided die needs exactly 3 questions.

Properties: H ≥ 0, and among distributions on K outcomes the **uniform** has the largest entropy, log K. Answer to
"Before you start": the loaded die has 2.16 bits, more than the fair coin's 1 bit. It's loaded, but it still has six
possible outcomes. The fair die (2.585) beats both.

## 2. Cross-entropy: the cost of using the wrong model

If outcomes really come from p but you predict (or encode) with a model q, your average surprise is the
**cross-entropy**:

$$H(p, q) = -\sum_x p(x)\log q(x).$$

**Worked example 2.** Weather: truth p = (sunny 0.6, cloudy 0.3, rainy 0.1).

| Model | H(p, model), bits | KL(p ‖ model), bits |
| --- | --- | --- |
| uniform q = (1/3, 1/3, 1/3) | 1.585 | 0.290 |
| better r = (0.5, 0.3, 0.2) | 1.353 | 0.058 |
| the truth p | 1.296 = H(p) | 0 |

The cross-entropy is always at least the entropy, and equals it only when the model is the truth.

## 3. KL divergence: the extra cost

$$\mathrm{KL}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p).$$

**Cross-entropy = entropy + KL.** The entropy H(p) is fixed by the data, so **minimising cross-entropy over models q
is the same as minimising KL(p ‖ q)**: making the model as close to the truth as possible.

Properties:

- KL ≥ 0, and KL = 0 only when p = q (Gibbs' inequality).
- **KL is not symmetric**: KL(p ‖ r) = 0.058 but KL(r ‖ p) = 0.069. So it's not a distance in the geometric sense; hence
  "divergence".
- KL is infinite if q(x) = 0 somewhere that p(x) > 0. A model that rules out something that actually happens is
  infinitely bad.

### KL between two normals

$$\mathrm{KL}\big(N(\mu_1, \sigma_1^2) \,\|\, N(\mu_2, \sigma_2^2)\big) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.$$

**Worked example 4.** KL(N(0, 1) ‖ N(1, 1)) = (Δμ)²/2 = **0.5 nats**; a Monte Carlo estimate, the average of
log p(x) − log q(x) over draws from p, gives 0.4997. Asymmetry again: KL(N(0, 1) ‖ N(0, 2²)) = 0.318, but
KL(N(0, 2²) ‖ N(0, 1)) = 0.807. Approximating a wide distribution with a narrow one is penalised more.

This closed form is the "KL term" in every VAE loss, where the encoder's normal is pulled towards a N(0, 1) prior.

## 4. Cross-entropy loss is negative log-likelihood

For a binary classifier predicting p̂ᵢ = P(yᵢ = 1), the **binary cross-entropy** (BCE) over n examples is

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^n \big[y_i\log\hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\big].$$

For each example only one term is non-zero: −log p̂ᵢ if yᵢ = 1, and −log(1 − p̂ᵢ) if yᵢ = 0. That's minus the log of the
probability the model gave to the **actual** label. So BCE is the **negative Bernoulli log-likelihood**, averaged.

**Worked example 3.** Labels y = (1, 0, 1, 1, 0), predictions p̂ = (0.9, 0.2, 0.6, 0.95, 0.4).

| Example | probability given to the true label | loss −ln(·) |
| --- | --- | --- |
| 1 | 0.9 | 0.105 |
| 2 | 1 − 0.2 = 0.8 | 0.223 |
| 3 | 0.6 | 0.511 |
| 4 | 0.95 | 0.051 |
| 5 | 1 − 0.4 = 0.6 | 0.511 |
| **mean** | | **0.280 nats** |

A confident wrong prediction (p̂ = 0.01 when y = 1) costs −ln 0.01 = 4.61 nats, as much as about 16 average examples.
Cross-entropy punishes overconfidence severely; that's why it produces probabilities worth calibrating (Module 08).

The multi-class version, −log q(true class), with q from a softmax (Lesson 01.02), is `CrossEntropyLoss`. Training
minimises the cross-entropy between the empirical label distribution and the model's, which is the same as
**maximum likelihood** (Module 05) and the same as minimising KL.

## 5. Forward vs reverse KL

When fitting a simple q to a complicated p, the direction matters (`entropy_and_kl.png`). **Worked example 6** fits one
normal to a target with two bumps, at −3 and +3:

| Objective | Best normal | Behaviour |
| --- | --- | --- |
| forward KL(p ‖ q) | mean 0.0, sd 3.15 | **mass-covering**: spreads out to cover both bumps, even putting mass in the empty middle |
| reverse KL(q ‖ p) | mean −3.0, sd 1.00 | **mode-seeking**: locks onto one bump and ignores the other |

Why: forward KL is huge wherever p > 0 but q ≈ 0, so q must cover everything p does. Reverse KL is huge wherever q > 0
but p ≈ 0, so q must avoid regions where p is empty. Maximum likelihood uses forward KL. Variational inference
(Module 07) uses reverse KL, which is why it tends to underestimate uncertainty.

## 6. Mutual information

How much does knowing X tell you about Y?

$$I(X; Y) = \sum_{x, y} p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)} = \mathrm{KL}\big(p(x, y) \,\|\, p(x)p(y)\big) = H(Y) - H(Y \mid X).$$

It's the KL divergence between the joint distribution and the "independent" one with the same marginals. It's zero
**exactly** when X and Y are independent, and unlike correlation it detects *any* kind of dependence, not just linear.

**Worked example 5.** For Lesson 03.06's studied/passed table, I(X; Y) = **0.1245 bits**. H(Y) = 1 bit, so knowing
whether a student studied removes about 12% of the uncertainty about passing, leaving H(Y | X) = 0.8755 bits. Mutual
information is used for feature selection, in decision trees (information gain) and in representation learning.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| Treating KL as symmetric, or as a distance | KL(p ‖ q) ≠ KL(q ‖ p) |
| Mixing bits and nats | log₂ gives bits, ln gives nats; 1 nat ≈ 1.443 bits |
| log(0) in cross-entropy | Clip predictions to [ε, 1 − ε]; libraries use log-softmax for stability |
| Thinking cross-entropy is unrelated to likelihood | It is the average negative log-likelihood |
| Correlation 0 ⇒ mutual information 0 | MI also catches non-linear dependence |

## Summary

- Entropy H(p) = −Σ p log p: average surprise; uniform is maximal, certainty is 0.
- Cross-entropy H(p, q) = −Σ p log q = H(p) + KL(p ‖ q).
- KL(p ‖ q) ≥ 0, asymmetric, and 0 only when p = q.
- BCE and softmax cross-entropy losses = negative log-likelihood = KL up to a constant.
- Forward KL covers modes; reverse KL seeks one mode.
- Mutual information = KL(joint ‖ product of marginals); zero ⇔ independent.

## Self-test

1. Entropy of a coin with P(heads) = 0.25, in bits?
2. A model predicts 0.7 for a positive example and 0.3 for a negative one. Total BCE in nats?
3. Why can't a good model ever assign probability 0 to an outcome that can happen?
4. Which KL direction does maximum likelihood minimise?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
