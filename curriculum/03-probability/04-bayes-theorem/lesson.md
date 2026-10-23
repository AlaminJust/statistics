# 03.04 — Bayes' theorem

**Time:** about 4 hours · **Prerequisites:** 03.03

## Why this matters

Bayes' theorem tells you how to **update a belief with evidence**: P(cause | evidence) from P(evidence | cause). It
explains why a positive test for a rare disease is usually a false alarm, it is the basis of the Naive Bayes classifier,
and it is the foundation of the whole of Bayesian statistics (Module 07). Every probabilistic classifier, from logistic
regression to a large language model, is in some sense computing a posterior.

## Learning objectives

- Derive Bayes' theorem from the definition of conditional probability.
- Compute posteriors with the formula, with natural frequencies and with the odds form.
- Explain the base-rate fallacy and why prevalence matters.
- Update sequentially as evidence accumulates.
- Build and run a small Naive Bayes classifier by hand and in log space.

## Before you start

A disease affects 1% of people. A test catches 99% of sick people and wrongly flags 5% of healthy people. You test
positive. Guess the probability that you have the disease.

---

## 1. The theorem

From the multiplication rule, P(A ∩ B) = P(A | B)P(B) = P(B | A)P(A). Divide by P(B):

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}.$$

With a partition A₁, …, A_k of hypotheses, the denominator comes from the law of total probability:

$$P(A_i \mid B) = \frac{P(B \mid A_i)\,P(A_i)}{\sum_j P(B \mid A_j)\,P(A_j)}.$$

| Term | Name | Meaning |
| --- | --- | --- |
| P(A) | **prior** | belief before seeing the evidence |
| P(B \| A) | **likelihood** | how probable the evidence is if A is true |
| P(B) | **evidence** (marginal likelihood) | how probable the evidence is overall |
| P(A \| B) | **posterior** | belief after seeing the evidence |

**posterior ∝ likelihood × prior.** The denominator just rescales so that the posteriors add to 1.

## 2. Worked example 1: the medical test

Notation: D = has the disease, + = tests positive.

| Quantity | Value | Name in medicine |
| --- | --- | --- |
| P(D) | 0.01 | prevalence |
| P(+ \| D) | 0.99 | sensitivity (true positive rate) |
| P(− \| not D) | 0.95 | specificity, so P(+ \| not D) = 0.05 (false positive rate) |

Step by step:

1. Evidence: P(+) = P(+ | D)P(D) + P(+ | not D)P(not D) = 0.99 × 0.01 + 0.05 × 0.99 = 0.0099 + 0.0495 = **0.0594**.
2. Posterior: P(D | +) = 0.0099 / 0.0594 = **0.167**.

Answer to "Before you start": only about **17%**. Most people guess 95% or 99%, confusing P(+ | D) with P(D | +).
Ignoring the prior is the **base-rate fallacy**.

## 3. Natural frequencies: the intuitive way

Picture 10,000 people:

```text
10,000 people
├── 100 sick (1%)
│   ├──  99 test positive   ← true positives
│   └──   1 test negative
└── 9,900 healthy
    ├── 495 test positive   ← false positives (5% of 9,900)
    └── 9,405 test negative
```

Of the 99 + 495 = 594 positives, only 99 are sick: 99/594 = **16.7%**. The false positives outnumber the true ones
because there are so many healthy people. A simulation of a million people agrees (0.164).

**Prevalence changes everything** (`posterior_vs_prevalence.png`):

| Prevalence | P(D \| +) |
| --- | --- |
| 0.1% | 0.019 |
| 1% | 0.167 |
| 10% | 0.687 |
| 30% | 0.895 |
| 50% | 0.952 |

The same test means very different things in mass screening (low prior) than in a clinic seeing patients with symptoms
(higher prior). In ML this is why **precision collapses on rare classes** even for an accurate classifier (Module 08).

## 4. Updating again: yesterday's posterior is today's prior

**Worked example 2.** You take a second, independent test and it's also positive. Start from prior 0.167:

P(D | ++) = 0.99 × 0.167 / (0.99 × 0.167 + 0.05 × 0.833) = **0.798**. A third positive: **0.987**.

Evidence accumulates. Each test multiplies the odds by the same factor (next section). Bayesian learning works this
way: start with a prior, update with each data point.

## 5. The odds form: the cleanest way to think

$$\underbrace{\frac{P(A \mid B)}{P(A^c \mid B)}}_{\text{posterior odds}} = \underbrace{\frac{P(A)}{P(A^c)}}_{\text{prior odds}} \times \underbrace{\frac{P(B \mid A)}{P(B \mid A^c)}}_{\text{likelihood ratio}}.$$

**Worked example 3.** Prior odds = 0.01/0.99 = 1/99 ≈ 0.0101. Likelihood ratio of a positive test = 0.99/0.05 = **19.8**.
Posterior odds = 19.8/99 = 0.2, so the probability is 0.2/1.2 = **0.167** ✓.

Taking logs gives **log posterior odds = log prior odds + log likelihood ratio**: evidence *adds* on the log-odds scale.
That is exactly how logistic regression and Naive Bayes combine features (Lessons 01.02 and 06.04).

## 6. Reversing conditionals in other settings

**Worked example 4.** In Lesson 03.03, machine 1 made 60% of parts (2% defective) and machine 2 made 40% (5% defective),
so P(defect) = 0.032. A part is defective. Which machine made it?

P(M2 | defect) = 0.4 × 0.05 / 0.032 = **0.625**, and P(M1 | defect) = 0.375. Machine 2 makes fewer parts but is
responsible for most of the defects.

**Worked example 6.** Bertrand's box (Lesson 03.01, exercise 11): P(gold–gold box | drew gold) =
(1/3 × 1) / (1/3 × 1 + 1/3 × 0 + 1/3 × ½) = (1/3)/(1/2) = **2/3**.

## 7. Naive Bayes: Bayes' theorem as a classifier

To classify an email with words w₁, …, w_m:

$$P(\text{spam} \mid w_1, \dots, w_m) \propto P(\text{spam}) \prod_{j=1}^m P(w_j \mid \text{spam}).$$

The **naive** assumption is that words are conditionally independent given the class (Lesson 03.03). It's false (e.g.
"free" and "prize" co-occur), but the classifier ranks emails well anyway.

**Worked example 5.** Learned from training data: P(spam) = 0.4; P("free" | spam) = 0.50, P("free" | ham) = 0.05;
P("meeting" | spam) = 0.05, P("meeting" | ham) = 0.30. A new email contains "free" and "meeting".

| Class | prior × likelihoods | unnormalised | posterior |
| --- | --- | --- | --- |
| spam | 0.4 × 0.50 × 0.05 | 0.0100 | 0.0100/0.0190 = **0.526** |
| ham | 0.6 × 0.05 × 0.30 | 0.0090 | **0.474** |

Spam, narrowly. "free" points strongly to spam (likelihood ratio 10), "meeting" points to ham (ratio 1/6), and the prior
leans towards ham. With hundreds of words these products underflow, so real implementations **add log-probabilities**
(Lesson 01.02): log scores −4.605 for spam vs −4.711 for ham. You'll build a full one in the module mini-project.

Two practical details you'll need: a word never seen in spam gets P = 0 and wipes out the whole product, so add a small
count to every word (**Laplace smoothing**: (count + 1)/(total + vocabulary size)); and compare log scores rather than
probabilities.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| P(D \| +) ≈ sensitivity | Combine with the prior; it can be far lower |
| Ignoring the base rate | Always ask "how common is this to begin with?" |
| Forgetting false positives in the denominator | P(+) = true positives + false positives |
| Multiplying many small probabilities | Add logs |
| Zero counts in Naive Bayes | Laplace smoothing |

## Summary

- P(A | B) = P(B | A)P(A)/P(B): posterior ∝ likelihood × prior.
- Natural frequencies make it intuitive: count true and false positives in a population of 10,000.
- Odds form: posterior odds = prior odds × likelihood ratio; in logs, evidence adds.
- Sequential updating: the posterior becomes the next prior.
- Naive Bayes: prior × product of per-feature likelihoods, computed in logs.

## Self-test

1. Prevalence 2%, sensitivity 90%, specificity 90%. P(D | +)?
2. What is the likelihood ratio of a negative result in worked example 1, P(− | D)/P(− | not D)?
3. Why can a very accurate classifier have low precision on a rare class?
4. Why does Naive Bayes need smoothing?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
