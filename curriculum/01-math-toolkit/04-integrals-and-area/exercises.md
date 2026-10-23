# 01.04 Exercises — Integrals and area

`make lesson L=01-math-toolkit/04-integrals-and-area`

## ★ Warm-up

1. **(M)** Compute exactly: ∫₀³ 4 dx; ∫₀¹ x³ dx; ∫₁ᵉ (1/x) dx; ∫₀¹ eˣ dx.
2. **(M)** For f(x) = x³ on [0, 2] with n = 4 strips, compute the left sum, right sum and trapezoid estimate. Compare
   with the exact value.
3. **(M)** Compute ∫₁³ (2x + 1) dx exactly.

## ★★ Core

4. **(M)** Find the constant k that makes f(x) = k·x a valid density on [0, 4]. Then compute P(1 ≤ X ≤ 3) and E[X].
5. **(M)** Show that f(x) = 3x² on [0, 1] is a valid density. Compute P(X ≤ 0.5) and E[X].
6. **(M)** The exponential density with rate λ > 0 is f(x) = λe^(−λx) on [0, ∞). Show it integrates to 1 and that
   P(X > t) = e^(−λt). If buses arrive at rate λ = 0.5 per minute, what is the probability of waiting more than
   2 minutes?
7. **(P)** Implement `integrate(f, a, b, n=1000)` (trapezoid rule) in `statkit/mathkit.py`. Check exercises 1, 4 and 5.
8. **(P)** For ∫₀² x² dx, compute the trapezoid error for n = 10, 20, 40, 80. By what factor does the error shrink each
   time n doubles?
9. **(P)** Using the normal density from the lesson, find P(−1.96 ≤ Z ≤ 1.96) numerically.

## ★★★ Challenge

10. **(M)** Show that E[X] = 1 for the density e⁻ˣ on [0, ∞). (Hint: check that the derivative of −(x + 1)e⁻ˣ is xe⁻ˣ.)
11. **(P)** **Monte Carlo integration.** If U is uniform on [0, 1], the average of f(U) over many draws estimates
    ∫₀¹ f(x) dx. Estimate ∫₀¹ e^(−x²) dx with 100, 10,000 and 1,000,000 draws, and compare with `integrate`. How does
    the error change with the number of draws?
12. **(M)** The **median** of a density is the m with P(X ≤ m) = 0.5. Find the median of f(x) = 2x on [0, 1] and
    compare it with the mean 2/3.

Solutions: [solutions/solutions.md](solutions/solutions.md)
