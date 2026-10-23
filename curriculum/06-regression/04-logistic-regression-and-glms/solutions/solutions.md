# 06.04 Solutions

1. OR = e^(−0.7) ≈ **0.50**: each one-unit increase in x halves the odds of the outcome, holding the other predictors fixed.
2. Linear predictor −5.629 + 4.916 + 1.285 = 0.572, so P = σ(0.572) ≈ **0.639**.
3. A 40% risk is odds of 0.4/0.6 = 0.667. Doubling the odds gives 1.333, which is a risk of 1.333/2.333 = 57%, not 80%.
   Odds and risk are close only when the risk is small (say under 10%). For communication, convert to risk differences or
   risk ratios at a realistic baseline.
4. Coefficients: study 0.45 and tutored 1.30 (both p < 0.01), sleep 0.44 (p = 0.051), Commerce −1.32 (**p = 0.022**),
   Science −0.64 (p = 0.18). The LR test for program as a whole: χ² = 5.7 on 2 df, **p = 0.057**. The surprise: program
   has *no* effect in `make_students.py`, so the Commerce dummy is a false positive. With several dummies, test the
   factor as a whole (the LR test) rather than reading off individual dummies. That's another form of multiple testing.
5. Adjusted OR for B = **1.145** (95% CI exp(0.044) to exp(0.227) = 1.045 to 1.255). Unadjusted OR 1.140. Rate ratio
   0.1086/0.0965 = 1.125. Adjustment barely matters because randomisation balanced device and pre_spend, though it can
   still shrink the SE (Module 09). The OR is close to the rate ratio because conversion is rare (about 10%).
6. ```python
   def fit_logistic_gd(X, y, learning_rate=0.1, steps=5000):
       X, y = np.asarray(X, float), np.asarray(y, float)
       beta = np.zeros(X.shape[1])
       for _ in range(steps):
           p = 1 / (1 + np.exp(-X @ beta))
           beta += learning_rate * X.T @ (y - p) / len(y)    # ascent on the mean log-likelihood
       return beta
   ```
   On standardised features it converges to the statsmodels coefficients (on the standardised scale).
7. The binned proportions follow the curves closely: low below about 7 hours, rising steeply between 8 and 15, near 1
   above 18. The logistic shape fits well here.
8. ℓ = Σ[yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)] with pᵢ = σ(ηᵢ), ηᵢ = **x**ᵢᵀ**β**. dℓ/dηᵢ = yᵢ/pᵢ · pᵢ(1 − pᵢ) −
   (1 − yᵢ)/(1 − pᵢ) · pᵢ(1 − pᵢ) = yᵢ(1 − pᵢ) − (1 − yᵢ)pᵢ = yᵢ − pᵢ. By the chain rule ∇ℓ = Σ(yᵢ − pᵢ)**x**ᵢ = Xᵀ(**y** − **p**). ∎
9. statsmodels warns "Perfect separation detected" and fails to converge, with coefficients running off to absurd values
   (slope ≈ 38). The MLE doesn't exist, because steeper and steeper curves keep increasing the likelihood. scikit-learn's
   default L2 penalty keeps the coefficients finite (slope ≈ 1.19). That's regularisation as a cure for separation
   (Lesson 06.05).
10. Poisson: coefficient 0.241, SE **0.048**, dispersion **3.5**, 95% CI [0.147, 0.335], which **misses** the truth
    ln 1.5 = 0.405. Negative binomial: same coefficient, SE **0.088**, CI [0.069, 0.413], which covers it. The estimate
    itself is low because of sampling noise; the overdispersed Poisson CI is too narrow to admit that.
11. With one binary x, the model reproduces the two group proportions exactly: β₀ = logit(p₀) and β₀ + β₁ = logit(p₁),
    so β₁ = logit(p₁) − logit(p₀) = log(odds₁/odds₀). Tutoring × high: OR = (28/32)/(37/103) = 2.436, and log 2.436 =
    **0.890**, the logistic coefficient ✓.
