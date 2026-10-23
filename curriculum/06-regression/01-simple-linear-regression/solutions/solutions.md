# 06.01 Solutions

1. b₁ = 0.6 × 8/2 = **2.4**; b₀ = 50 − 2.4 × 10 = **26**. ŷ = 26 + 2.4x, so at x = 13, ŷ = **57.2**.
2. x̄ = 3, ȳ = 4, Sxy = 6, Sxx = 10, so b₁ = **0.6** and b₀ = **2.2**. Residuals −0.8, 0.6, 1.0, −0.6, −0.2. SSE = 2.4, SST = 6,
   R² = **0.6**.
3. ∂SSE/∂b₀ = −2Σ(yᵢ − b₀ − b₁xᵢ) = 0 at the minimum, i.e. Σ residuals = 0. (This needs an intercept in the model.)
4. Slope **1.77** points per hour of sleep, 95% CI **[−0.15, 3.70]**; R² = **0.016**. The data were generated with a true
   effect of 2.0 per hour, so sleep matters. But sleep varies little between students (sd 1 hour), so it explains little
   of the *variance* in scores, and the CI is wide. A low R² does not mean an unimportant variable.
5. The slope is divided by 60 (0.0341 per minute) and the intercept is unchanged. R², t and p are identical. Units change
   coefficients, not the evidence.
6. Slope = **0.728** = r, intercept ≈ 0. With standardised variables, the regression slope *is* the correlation.
7. | hours | predicted mean | 95% CI (mean) | 95% prediction interval |
   | --- | --- | --- | --- |
   | 5 | 53.3 | [51.4, 55.3] | [34.5, 72.2] |
   | 10 | 63.6 | [62.2, 64.9] | [44.8, 82.4] |
   | 25 | 94.3 | [90.1, 98.5] | [75.1, **113.5**] |

   25 hours is near the edge of the data, so its intervals are widest, and the prediction interval goes above the
   maximum score of 100: the linear model doesn't know scores are capped.
8. ```python
   def fit_ols(X, y):
       X, y = np.asarray(X, float), np.asarray(y, float)
       return np.linalg.solve(X.T @ X, X.T @ y)

   def predict(X, beta):
       return np.asarray(X, float) @ beta

   def residuals(X, y, beta):
       return np.asarray(y, float) - predict(X, beta)

   def r_squared(X, y, beta):
       y = np.asarray(y, float)
       e = residuals(X, y, beta)
       return 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
   ```
9. With seed 0: patients with a first reading over 160 average about **169.1**, and their second reading averages about
   **160.1**, with no treatment. Theory: E[second | first] = 140 + ρ(first − 140) with ρ = 225/(225 + 100) = 0.69, which
   gives 140 + 0.69 × 29.1 ≈ **160.2** ✓. A "treatment" given to these patients would appear to lower blood pressure by
   about 9 points.
10. ∂/∂b₀: Σ(yᵢ − b₀ − b₁xᵢ) = 0 ⇒ b₀ = ȳ − b₁x̄. Substitute into ∂/∂b₁: Σxᵢ(yᵢ − ȳ − b₁(xᵢ − x̄)) = 0 ⇒
    Sxy − b₁Sxx = 0 ⇒ b₁ = Sxy/Sxx (using Σxᵢ(yᵢ − ȳ) = Sxy and Σxᵢ(xᵢ − x̄) = Sxx).
11. ŷᵢ − ȳ = b₁(xᵢ − x̄), so SST − SSE = Σ(ŷᵢ − ȳ)² = b₁²Sxx = Sxy²/Sxx. Hence R² = Sxy²/(Sxx Syy) = r². ∎
12. All four fits give intercept ≈ 3.00 (SE 1.12), slope 0.500 (SE 0.118) and R² ≈ 0.67. Residual plots: set I shows
    random scatter (the model fits); II shows a clear curve (non-linearity); III is nearly perfect except for one big
    outlier, which distorts the line; IV has one point with huge leverage that alone determines the slope. Identical
    summary output, four different stories: always plot the residuals (Lesson 06.03).
