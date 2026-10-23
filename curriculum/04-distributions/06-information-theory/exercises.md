# 04.06 Exercises — Entropy, cross-entropy and KL divergence

`make lesson L=04-distributions/06-information-theory`

## ★ Warm-up

1. **(M)** Compute in bits: H(0.25, 0.75), H(0.5, 0.25, 0.25), and the entropy of a uniform distribution on 5 outcomes.
2. **(M)** A model predicts 0.7 for a positive example and 0.3 for a negative one. Compute the total BCE over the two
   examples, in nats.
3. **(C)** Why must a well-trained classifier never output probability exactly 0 for a class that can occur?

## ★★ Core

4. **(M)** For p = (0.5, 0.5) and q = (0.9, 0.1), compute KL(p ‖ q) and KL(q ‖ p) in bits. Which is larger, and why?
5. **(M)** A 3-class model outputs logits (2, 1, 0.1). Compute the softmax probabilities and the cross-entropy loss if the
   true class is the first one. What would the loss be if the true class were the third?
6. **(M)** Use the normal KL formula to compute KL(N(1, 1) ‖ N(0, 2²)) and KL(N(0, 2²) ‖ N(1, 1)).
7. **(P)** Implement `entropy(probs, base=2)`, `cross_entropy(p, q, base=2)` and `kl_divergence(p, q, base=2)` in
   `statkit/distributions.py`. Make `kl_divergence` return `math.inf` when q is 0 where p > 0. Check the weather example.
8. **(P)** In `students.csv`, compute the mutual information (bits) between "exam_score ≥ 70" and (a) tutoring, (b)
   "study_hours ≥ 12". Which feature is more informative about doing well? Does it match the correlations of Lesson 02.06?
9. **(P)** For a fair-coin label (y ~ Bernoulli(0.5)), plot the expected BCE loss of a constant prediction p̂ against p̂.
   Where is the minimum, and what value does the loss take there?

## ★★★ Challenge

10. **(M)** Prove Gibbs' inequality, KL(p ‖ q) ≥ 0, using ln x ≤ x − 1. (Hint: apply it to −KL = Σ p ln(q/p).)
11. **(M)** Show that minimising the average BCE over a dataset with respect to a single constant prediction p̂ gives
    p̂ = the share of positives. (Differentiate and set to zero.) Connect this to the MLE of a Bernoulli in Module 05.
12. **(P)** **Knowledge distillation.** A "teacher" model outputs logits (4, 2, 1, 0) for an image. Compute its softmax at
    temperature T = 1 and T = 4 (divide the logits by T first). Compute the entropy of each. Why do distillation methods
    train students to match the *higher-temperature* teacher distribution, using a KL loss?

Solutions: [solutions/solutions.md](solutions/solutions.md)
