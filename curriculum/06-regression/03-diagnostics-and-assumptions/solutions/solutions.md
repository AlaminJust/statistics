# 06.03 Solutions

1. Heteroscedasticity: the error variance grows with the fitted value. Quick fix for inference: robust (HC3) SEs. A model
   fix: a log transformation of y (common for money, counts and times), or weighted least squares.
2. A high-leverage point pulls the fitted line towards itself, so its own residual ends up small even if it is wrong. Its
   influence shows in Cook's distance (or in how much the slope changes when you drop it), not in its residual.
3. SE inflation = √VIF = √20 ≈ **4.5 times**. VIF = 1/(1 − R²) = 20 ⇒ **R_j² = 0.95**: the other predictors explain 95% of
   this predictor's variation.
4. R² = **0.518**. Breusch–Pagan p = **0.0035**: some heteroscedasticity (residual spread grows with the fitted value).
   Maximum Cook's distance 0.024 and maximum leverage 0.13: no single point dominates. Conclusion: the linear model is
   reasonable, but use robust SEs.
5. VIFs: s1 **59**, s2 **39**, s3 15, s5 10, s4 9; the others below 2. s1 (total cholesterol), s2 (LDL), s3 (HDL) and s4
   (total/HDL) are algebraically linked, since total cholesterol is roughly LDL + HDL + a share of triglycerides, so they
   are close to linear combinations of one another. Their individual coefficients and p-values are unreliable (s1
   p = 0.06, s2 p = 0.16 even though cholesterol clearly matters). Robust SEs change little here; the problem is
   collinearity, not variance.
6. F = **2.02** on 4 and 431 df, p = **0.09**. These four don't add clear predictive information once s3 and s5 are in.
7. ```python
   def vif(X):
       X = np.asarray(X, float)
       n, p = X.shape
       out = []
       for j in range(p):
           others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
           beta = fit_ols(others, X[:, j])
           out.append(1 / (1 - r_squared(others, X[:, j], beta)))
       return out
   ```
8. Both points have the same leverage (**0.08**: 30 hours is far from the mean). (30, 100) follows the trend: studentised
   residual −0.46, Cook's D 0.009, slope 2.03 (barely changed). (30, 30) is far off the trend: residual **−7.5**, Cook's D
   **1.9**, and the slope drops to 1.77. Leverage alone isn't harmful; leverage plus a large residual is.
9. Hᵀ = (X(XᵀX)⁻¹Xᵀ)ᵀ = X(XᵀX)⁻¹Xᵀ = H, since XᵀX is symmetric. H² = X(XᵀX)⁻¹XᵀX(XᵀX)⁻¹Xᵀ = X(XᵀX)⁻¹Xᵀ = H.
   tr(H) = tr((XᵀX)⁻¹XᵀX) = tr(I_{p+1}) = p + 1. The leverages are the diagonal entries hᵢᵢ, which sum to p + 1, so they
   average (p + 1)/n.
10. Plain OLS rejects a true null **38%** of the time; cluster-robust SEs bring it down to about **6%**. Students in the
    same classroom share the classroom effect, so the 600 observations are not 600 independent pieces of evidence about
    the treatment. Treatment was assigned to 30 classrooms, and that's closer to the effective sample size. (Mixed models
    in Lesson 06.06 handle this structure directly.)
11. Raw revenue: R² 0.10, residual skew **1.83**, Breusch–Pagan p = 0.48. log(revenue): R² 0.13, residual skew **0.25**,
    but Breusch–Pagan p ≈ 10⁻¹⁵. The log fixes the skew (which matters for prediction intervals), but on the log scale
    the variance changes with pre_spend, so use robust SEs. Diagnostics often involve trade-offs like this rather than
    one perfect fix.
