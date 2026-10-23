# 09.03 Solutions

1. IPW: given the measured covariates, treatment is as good as random, and every unit has some chance of either
   treatment. DiD: without the treatment, treated and control groups would have followed parallel trends. IV: the
   instrument shifts the treatment and affects the outcome only through it, and is unrelated to the confounders. RD:
   units can't precisely manipulate the running variable, so those just either side of the cutoff are comparable.
2. Given its covariates, this unit was 95% likely to be treated. Comparable *untreated* units are rare, so if it's
   untreated it gets weight 1/0.05 = 20, and the control estimate leans heavily on a few such people (high variance; if
   e is near 1, positivity fails).
3. (58 − 50) − (45 − 40) = **+3**, assuming treated regions would have grown by 5 like the controls (parallel trends).
4. See `../solutions/statkit_causal_reference.py`.
5. With non-linear earnings (one seed): regression **+2,192**, IPW +1,383, AIPW +1,483. Averaged over 20 seeds: regression
   +2,456 (biased: the linear outcome model is wrong), IPW **+1,818**, AIPW **+1,860** (both close to the truth: the
   propensity model is right). This is the doubly-robust property: a wrong outcome model can be rescued by a right
   propensity model.
6. With an unmeasured motivation effect, everything is biased upwards: at strength 0.5, regression +1,976, IPW +1,673,
   AIPW +1,983 (noisy); at strength 1.0, about **+2,900, +2,700 and +3,000**, far from +1,800. No adjustment method can fix
   confounders you didn't measure. Sensitivity analysis asks how strong such a confounder would need to be to change the
   conclusion. Since motivation biases the estimate *upward* here, a strong enough negative confounder would be needed to
   push it to zero.
7. DiD estimates the true +4 **plus** the differential trend +2 ≈ **+6**. With several pre-periods, an event-study plot
   shows the treated–control gap already widening before the policy, so pre-period "effects" are non-zero, which signals
   violated parallel trends.
8. With a weak instrument, the median first-stage F is about **1** (far below 10). The 2SLS estimates scatter wildly (IQR
   −0.07 to +0.27), with a median of about 0.12, biased towards the confounded OLS value (0.16). Weak instruments give
   biased, unreliable IV estimates.
9. Retaking piles students up just above 70 and thins the region just below, which shows as a jump in the histogram at the
   cutoff. The students who retake are more motivated, so the units just above 70 are no longer comparable to those just
   below: the RD estimate is biased (here upwards).
10. E[TY/e(X)] = E[E[TY(1)/e(X) | X]] (since TY = TY(1)) = E[E[T | X] E[Y(1) | X]/e(X)] (no unmeasured confounding:
    T ⊥ Y(1) | X) = E[e(X) E[Y(1) | X]/e(X)] = E[Y(1)]. Positivity (e > 0) is needed to divide. ∎
11. For example: a study of screen time and adolescent depression using survey data. Estimand: the effect of each extra
    daily hour. Method: regression adjustment. Key assumption: no unmeasured confounding, but family stress could drive
    both. Robustness check: an E-value, a negative-control outcome, or a within-person (panel) analysis.
