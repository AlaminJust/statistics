# 01.04 Solutions

1. [4x]₀³ = **12**; [x⁴/4]₀¹ = **1/4**; [ln x]₁ᵉ = 1 − 0 = **1**; [eˣ]₀¹ = **e − 1 ≈ 1.718**.
2. Δx = 0.5. Left: (0 + 0.125 + 1 + 3.375) × 0.5 = **2.25**. Right: (0.125 + 1 + 3.375 + 8) × 0.5 = **6.25**.
   Trapezoid: (2.25 + 6.25)/2 = **4.25**. Exact: [x⁴/4]₀² = **4**.
3. [x² + x]₁³ = (9 + 3) − (1 + 1) = **10**.
4. ∫₀⁴ kx dx = k · 16/2 = 8k = 1 ⇒ **k = 1/8**.
   P(1 ≤ X ≤ 3) = (1/8)[x²/2]₁³ = (1/8)(4.5 − 0.5) = **0.5**.
   E[X] = ∫₀⁴ x · x/8 dx = (1/8)[x³/3]₀⁴ = (1/8)(64/3) = **8/3 ≈ 2.667**.
5. 3x² ≥ 0 on [0, 1] and [x³]₀¹ = 1 ✓. P(X ≤ 0.5) = 0.5³ = **0.125**. E[X] = ∫₀¹ 3x³ dx = **3/4**.
6. ∫₀^∞ λe^(−λx) dx = [−e^(−λx)]₀^∞ = 0 − (−1) = 1 ✓. P(X > t) = [−e^(−λx)]ₜ^∞ = **e^(−λt)**.
   With λ = 0.5, t = 2: e⁻¹ ≈ **0.368**.
7. ```python
   def integrate(f, a, b, n=1000):
       dx = (b - a) / n
       total = 0.5 * (f(a) + f(b))
       for i in range(1, n):
           total += f(a + i * dx)
       return total * dx
   ```
8. Errors: 0.01333, 0.00333, 0.00083, 0.00021. Each doubling of n shrinks the error by **4**: the trapezoid error
   is proportional to 1/n².
9. **0.9500**. This is why "±1.96 standard deviations" gives a 95% interval (Module 05).
10. d/dx[−(x + 1)e⁻ˣ] = −e⁻ˣ + (x + 1)e⁻ˣ = xe⁻ˣ ✓. So E[X] = [−(x + 1)e⁻ˣ]₀^∞ = 0 − (−1) = **1**.
11. True value ≈ 0.746824. Typical estimates (seed 0): 100 draws → 0.7095; 10,000 → 0.7472; 1,000,000 → 0.7467.
    The error shrinks roughly like 1/√n: 100 times more draws → about 10 times smaller error. Slower than the
    trapezoid rule in 1-D, but Monte Carlo works the same in 1,000 dimensions, where grids are impossible. That is why
    Bayesian computation (Module 07) relies on it.
12. P(X ≤ m) = m² = 0.5 ⇒ **m = √0.5 ≈ 0.707**. The median (0.707) is above the mean (0.667): this density has a longer
    tail to the left, which pulls the mean down. (Module 02: mean vs median and skew.)
