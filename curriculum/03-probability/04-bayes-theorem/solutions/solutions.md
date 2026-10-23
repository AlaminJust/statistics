# 03.04 Solutions

1. P(+) = 0.9 × 0.02 + 0.1 × 0.98 = 0.018 + 0.098 = 0.116, so P(D | +) = 0.018/0.116 ≈ **0.155**.
   Natural frequencies: 200 sick → 180 positive; 9,800 healthy → 980 positive; 180/1,160 = 0.155.
2. P(− | D) = 0.01 and P(− | not D) = 0.95. P(D | −) = 0.01 × 0.01 / (0.01 × 0.01 + 0.95 × 0.99) ≈ **0.000106**: about 1 in
   9,400. LR(negative) = 0.01/0.95 ≈ **0.0105**, so a negative result divides the odds by about 95.
3. When something is rare, most positive results come from the huge group that doesn't have it, even if each of them
   has only a small chance of a false alarm. So before trusting a positive, ask how common the condition was to begin
   with. A 95% accurate test for a 1-in-100 condition is still wrong most of the times it says "yes".
4. P(blue | says blue) = 0.8 × 0.15 / (0.8 × 0.15 + 0.2 × 0.85) = 0.12/0.29 ≈ **0.41**. Even with an 80% reliable
   witness, the taxi was more likely green.
5. 0.99 × 0.005 / (0.99 × 0.005 + 0.01 × 0.995) ≈ **0.332**. Two out of three positives are false. One test is not
   enough; a confirmatory test, independent of the first, is needed.
6. Likelihood of HH: fair 0.25, two-headed 1, biased 0.5625. With equal priors the posteriors are proportional to these:
   **fair 0.138, two-headed 0.552, biased 0.310**. P(next heads) = 0.138 × 0.5 + 0.552 × 1 + 0.310 × 0.75 ≈ **0.853**.
   (This is a *posterior predictive* probability, Module 07.)
7. Prior odds 1/99. After two positives: (1/99) × 19.8² = 392.04/99 = 3.96, so the probability is 3.96/4.96 ≈ **0.798** ✓.
8. ```python
   def bayes_posterior(prior, sensitivity, specificity):
       true_positive = sensitivity * prior
       false_positive = (1 - specificity) * (1 - prior)
       return true_positive / (true_positive + false_positive)
   ```
9. P(spam | email), counting only the words present:

   | Email | P(spam) |
   | --- | --- |
   | {free} | 0.870 |
   | {meeting} | 0.100 |
   | {free, prize} | 0.995 |
   | {meeting, prize} | 0.769 |

   "prize" has likelihood ratio 30 and overwhelms "meeting". (A more complete model would also use the *absence* of
   words; that's the Bernoulli version of Naive Bayes.)
10. Take logs of the odds form. For conditionally independent evidence B₁, …, B_k, P(B₁…B_k | A) = Π P(Bᵢ | A) and
    likewise given Aᶜ, so the combined LR is the product of the individual LRs, and the **log LRs add**:
    log-odds(A | all) = log-odds(A) + Σ log LRᵢ. This is a linear model in the evidence, which is why Naive Bayes and
    logistic regression look so similar.
11. ```python
    theta = np.linspace(0, 1, 101)
    post = np.ones_like(theta) / len(theta)
    for flip in flips:                        # flip = 1 for heads, 0 for tails
        post *= theta if flip else (1 - theta)
        post /= post.sum()
    ```
    The posterior starts flat, then narrows and centres near 0.7 as flips accumulate.
12. You need the **prior**: how many other people could have left the DNA. In a city of 10 million, a 1-in-a-million
    match rate means about 10 innocent people would also match. If nothing else points to the defendant, P(guilty |
    match) is roughly 1/11, so P(innocent | match) ≈ 0.9. Mixing up "P(match | innocent)" with "P(innocent | match)" is
    the **prosecutor's fallacy**.
