# 04.06 Solutions

1. H(0.25, 0.75) ≈ **0.811** bits. H(0.5, 0.25, 0.25) = 0.5(1) + 2 × 0.25(2) = **1.5** bits. Uniform on 5: log₂ 5 ≈ **2.322** bits.
2. −ln 0.7 − ln(1 − 0.3) = 2 × 0.357 ≈ **0.713 nats** (0.357 per example on average).
3. Its loss on that class's examples would be −log 0 = ∞, and its KL from the truth would be infinite. Models output
   small positive probabilities (and loss functions clip them) to avoid this.
4. KL(p ‖ q) = 0.5 log₂(0.5/0.9) + 0.5 log₂(0.5/0.1) ≈ **0.737 bits**. KL(q ‖ p) = 0.9 log₂(0.9/0.5) + 0.1 log₂(0.1/0.5) ≈
   **0.531 bits**. KL(p ‖ q) is larger because q gives only 0.1 to "tails", which p says happens half the time, and
   −log of a small q is expensive.
5. Softmax ≈ (0.659, 0.242, 0.099). Loss for true class 1: −ln 0.659 ≈ **0.417**. For true class 3: −ln 0.099 ≈ **2.317**.
6. KL(N(1, 1) ‖ N(0, 4)) = ln 2 + (1 + 1)/8 − 0.5 ≈ **0.443**. KL(N(0, 4) ‖ N(1, 1)) = ln(1/2) + (4 + 1)/2 − 0.5 ≈ **1.307**.
7. ```python
   def kl_divergence(p, q, base=2):
       total = 0.0
       for pi, qi in zip(p, q):
           if pi == 0:
               continue
           if qi == 0:
               return math.inf
           total += pi * math.log(pi / qi, base)
       return total
   ```
8. (a) Tutoring: **0.027** bits. (b) Study ≥ 12: **0.187** bits, about 7 times more. This matches Lesson 02.06, where
   study hours were the strongest correlate of exam score.
9. The expected loss is −0.5 ln p̂ − 0.5 ln(1 − p̂). It's minimised at **p̂ = 0.5**, with value ln 2 ≈ 0.693 nats, the
   entropy of a fair coin. No model can beat the entropy of the labels without using features.
10. −KL = Σ p ln(q/p) ≤ Σ p (q/p − 1) = Σ q − Σ p = 1 − 1 = 0, so KL ≥ 0, with equality only when q/p = 1 everywhere. ∎
11. L(p̂) = −(1/n)Σ[yᵢ ln p̂ + (1 − yᵢ) ln(1 − p̂)] = −[ȳ ln p̂ + (1 − ȳ) ln(1 − p̂)]. Setting dL/dp̂ = −ȳ/p̂ + (1 − ȳ)/(1 − p̂) = 0
    gives **p̂ = ȳ**. This is the MLE of a Bernoulli probability (Module 05).
12. T = 1: (0.831, 0.112, 0.041, 0.015), entropy **0.86 bits**. T = 4: (0.409, 0.248, 0.193, 0.150), entropy **1.90 bits**.
    At T = 1 the teacher is nearly one-hot, so it tells the student little more than the label does. The softened
    distribution reveals the teacher's "dark knowledge", e.g. that class 2 is much more plausible than class 4 for this
    image. A KL loss between the softened teacher and student transfers that richer information.
