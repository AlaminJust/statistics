# 01.03 Solutions

1. 12x² − 2; −5x⁻²; (1/2)x^(−1/2) = 1/(2√x); 3eˣ; 1/x.
2. 3(2x − 5)² · 2 = **6(2x − 5)²**; **4e^(4x)**; **2x / (x² + 1)**; **−2(x − 2) e^(−(x − 2)²)**.
   At x = 1 these are 54, 218.39, 1 and 0.7358 — your `numerical_derivative` should match (exercise 9).
3. Going **down**. Change ≈ f′(x) · Δx = −2 × 0.01 = **−0.02**.
4. f′(x) = 2x − 10 = 0 ⇒ **x = 5**; f(5) = 25 − 50 + 4 = −21. f″ = 2 > 0 ⇒ minimum.
5. f(p) = p − p², f′(p) = 1 − 2p = 0 ⇒ **p = 0.5**, f(0.5) = 0.25. f″ = −2 < 0 ⇒ maximum. A fair coin is the most
   unpredictable coin.
6. S′(m) = −2Σ(xᵢ − m) = 0 ⇒ m = 15/3 = 5. S(5) = 9 + 1 + 16 = 26; S(4) = 4 + 0 + 25 = 29; S(6) = 16 + 4 + 9 = 29.
   Both neighbours are larger. ✓
7. ∂g/∂x = 4x + y; ∂g/∂y = x − 3. At (1, 1): **∇g = (5, −2)**.
8. f′(x) = 2(x + 1). Steps: 2 → 2 − 0.25·6 = **0.5** → 0.5 − 0.25·3 = **−0.25** → −0.25 − 0.25·1.5 = **−0.625**.
   The gap to −1 goes 3 → 1.5 → 0.75 → 0.375: multiplied by **1 − 2η = 0.5** each step.
9. ```python
   def numerical_derivative(f, x, h=1e-5):
       return (f(x + h) - f(x - h)) / (2 * h)
   ```
10. ```python
    def gradient_descent(grad, x0, learning_rate, steps):
        x = x0
        for _ in range(steps):
            x = x - learning_rate * grad(x)
        return x
    gradient_descent(lambda x: 2 * x - 10, 0.0, 0.4, 20)   # ≈ 5.0 (factor 1 − 2·0.4 = 0.2)
    gradient_descent(lambda x: 2 * x - 10, 0.0, 1.2, 20)   # diverges (factor −1.4)
    ```
    Any η with |1 − 2η| < 1, i.e. 0 < η < 1, converges; η > 1 diverges.
11. σ = (1 + e⁻ᶻ)⁻¹. Chain rule: σ′ = −(1 + e⁻ᶻ)⁻² · (−e⁻ᶻ) = e⁻ᶻ / (1 + e⁻ᶻ)².
    Split it: [1 / (1 + e⁻ᶻ)] · [e⁻ᶻ / (1 + e⁻ᶻ)] = σ(z) · (1 − σ(z)), since 1 − σ = e⁻ᶻ / (1 + e⁻ᶻ). ✓
12. dL/dw = −(2/n) Σ xᵢ(yᵢ − w xᵢ) = 0 ⇒ Σxᵢyᵢ = w Σxᵢ² ⇒ **w = Σxᵢyᵢ / Σxᵢ²**.
    Σxᵢyᵢ = 2 + 8 + 21 = 31, Σxᵢ² = 14 ⇒ **w = 31/14 ≈ 2.214**.
13. `gradient_descent(lambda w: -(2/3) * sum(x*(y - w*x) for x, y in zip(xs, ys)), 0.0, 0.05, 200)` ≈ 2.2143.
14. The minimum of Σ|xᵢ − m| is at **m = 4** (the median), value 18; the mean is 6. Changing 12 to 1200 moves the mean
    to **243.6** but leaves the median at **4**. The median is *robust* to extreme values (Module 02 explores this).
