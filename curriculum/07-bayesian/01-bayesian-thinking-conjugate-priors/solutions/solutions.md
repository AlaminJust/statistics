# 07.01 Solutions

1. Posterior **Beta(5, 11)**: mean 5/16 = **0.3125**, mode (5 − 1)/(5 + 11 − 2) = **0.286**, 95% credible interval
   **[0.118, 0.551]**.
2. "Given our data and assumptions, there's a 95% chance the true value lies in the credible interval. A confidence
   interval is a statement about the method: if we repeated the study many times, 95% of the intervals built this way
   would contain the true value. With plenty of data and no strong prior information, the two intervals are usually
   almost the same."
3. Posterior mode = (a + k − 1)/(a + b + n − 2). The MLE on pooled counts with (a − 1) extra successes and (b − 1) extra
   failures is (k + a − 1)/(n + a + b − 2). They're the same. (A flat Beta(1, 1) adds zero pseudo-observations.)
4. Treatment posterior Beta(16, 6), control Beta(10, 12). P(treatment > control) ≈ **0.97**. 95% credible interval for the
   difference about **[−0.01, 0.53]**: probably better, but by an uncertain margin, which is typical of small trials.
5. Posterior Beta(1, 51): mean 1/52 ≈ **0.019**; 95% upper bound ≈ **0.057**. The rule of three gives 3/50 = 0.06, and the
   exact frequentist bound 1 − 0.05^(1/50) = 0.058. Zero observed defects doesn't mean a zero defect rate.
6. Precisions: 1/225 + 1/25 = 0.0444, so the posterior sd = **4.74**. Mean = (100/225 + 130/25)/0.0444 = **127.0**. One
   noisy measurement of an extreme value is pulled slightly towards the population average: regression to the mean,
   handled correctly.
7. Posterior Beta(5, 195): mean **0.025**, 95% interval **[0.008, 0.051]**. The raw rate of 3% is shrunk towards 2% because
   100 impressions are few. This is why ad platforms shrink new ads' rates before ranking them.
8. ```python
   def beta_binomial_update(alpha, beta, successes, failures):
       return alpha + successes, beta + failures

   def normal_normal_update(prior_mean, prior_sd, data, sigma):
       n = len(data)
       precision = 1 / prior_sd ** 2 + n / sigma ** 2
       mean = (prior_mean / prior_sd ** 2 + sum(data) / sigma ** 2) / precision
       return mean, precision ** -0.5
   ```
9. Tutored: 28 of 60 high → Beta(29, 33). Others: 37 of 140 → Beta(38, 104). P(tutored rate > other rate) ≈ **0.997**.
10. Collect the terms in μ in the exponent: −μ²/2 × (1/s₀² + n/σ²) + μ(m₀/s₀² + nx̄/σ²) + const. A normal density has
    exponent −(μ − m)²/(2s²) = −μ²/(2s²) + μm/s² + const. Matching coefficients gives 1/s² = 1/s₀² + n/σ² and
    m = s²(m₀/s₀² + nx̄/σ²). ∎
11. P(treatment better): Beta(1, 1) 0.972; Beta(0.5, 0.5) 0.975; Beta(5, 5) 0.945; Beta(20, 20) 0.868. The conclusion
    "probably better" holds, but a strongly sceptical prior (equivalent to 40 earlier patients per arm at a 50% rate)
    noticeably weakens it. With 20 patients per arm, report this sensitivity.
12. Over 10,000 rounds the variants were shown 3.1%, 9.1% and **87.9%** of the time. Early on, the posteriors are wide and
    every variant sometimes produces the largest draw (exploration). As evidence accumulates, the best variant's
    posterior dominates and is chosen almost always (exploitation). A variant is shown exactly as often as it's
    plausibly the best. This "Thompson sampling" is widely used for ads, recommendations and adaptive experiments.
