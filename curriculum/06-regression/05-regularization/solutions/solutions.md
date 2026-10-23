# 06.05 Solutions

1. Adding a penalty can only make the fit to the *training* data worse, because OLS already minimises training error.
   The benefit of λ > 0 shows up only on new data, so λ must be chosen on held-out data.
2. The diamond has corners on the axes (where one coefficient is 0). An ellipse expanding from the OLS solution usually
   touches the diamond first at a corner. The circle is smooth everywhere, so the touching point usually has both
   coefficients non-zero.
3. Not much. With 200 observations per parameter, OLS has very low variance, so there's little to gain by trading bias
   for variance.
4. Minimise Σ(yᵢ − βxᵢ)² + λβ². Derivative: −2Σxᵢ(yᵢ − βxᵢ) + 2λβ = 0 ⇒ β = Σxᵢyᵢ/(Σxᵢ² + λ). With Σxᵢ² = n:
   β_ridge = (n/(n + λ)) β_OLS.
5. ```python
   def fit_ridge(X, y, lam):
       X, y = np.asarray(X, float), np.asarray(y, float)
       penalty = lam * np.eye(X.shape[1])
       penalty[0, 0] = 0.0                      # don't penalise the intercept
       return np.linalg.solve(X.T @ X + penalty, X.T @ y)
   ```
6. On all 442 rows with 10-fold CV: optimal α ≈ **1.20**; the one-SE rule picks α ≈ **7.9**, keeping only **4** features
   (bmi, bp, s3, s5), a simpler model that is almost as accurate.
7. Selection frequencies at α = 5 (200 bootstraps): bmi, bp and s5 **100%**, s3 **99%**, sex 82%, s6 38%, s2 16%, s4 9%,
   age 6%, s1 4%. The top four are stable. s1, s2 and s4 come and go: they're collinear, and the lasso picks among them
   almost at random.
8. | C | non-zero coefficients (of 30) | 5-fold CV accuracy |
   | --- | --- | --- |
   | 0.01 | 4 | 0.933 |
   | 0.1 | 8 | **0.972** |
   | 1 | 15 | 0.972 |
   | 10 | 24 | 0.963 |

   Eight features give the same accuracy as fifteen. Too little penalty (C = 10) slightly overfits.
9. Dense truth (seed 1): **ridge 3.32**, lasso 3.99, predicting the mean 3.95. With many small effects there's nothing
   sparse to select: the lasso zeroes out useful signal and does no better than the mean, while ridge shrinks
   everything a little and keeps the combined signal. Sparse truth → lasso; dense truth → ridge; unknown → elastic net
   with CV.
10. log posterior = −‖**y** − X**β**‖²/(2σ²) − ‖**β**‖²/(2τ²) + const. Maximising it is the same as minimising
    ‖**y** − X**β**‖² + (σ²/τ²)‖**β**‖², so **λ = σ²/τ²**. A tighter prior (small τ) means a bigger penalty.
11. Coordinate descent on the standardised diabetes data with α = 5 gives
    (0, −2.155, 24.216, 10.331, 0, 0, −7.027, 0, 21.229, 0), matching `Lasso(alpha=5)` to three decimals. The
    soft-threshold step is where the exact zeros come from.
12. The same data were used to choose the 7 genes *and* to test them, so the p-values ignore the search over thousands
    of candidates (Lesson 05.08). They are far too small. Valid alternatives: split the sample (select on one half, test
    on the other), selective-inference methods, stability selection, or a replication dataset.
