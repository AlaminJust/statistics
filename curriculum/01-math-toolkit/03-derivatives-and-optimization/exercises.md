# 01.03 Exercises — Derivatives and optimisation

`make lesson L=01-math-toolkit/03-derivatives-and-optimization`

## ★ Warm-up

1. **(M)** Differentiate: 4x³ − 2x + 7; 5x⁻¹; √x; 3eˣ; ln(x).
2. **(M)** Using the chain rule: (2x − 5)³; e^(4x); ln(x² + 1); e^(−(x − 2)²).
3. **(C)** At a point where f′(x) = −2, is f going up or down? By about how much does f change if x increases by 0.01?

## ★★ Core

4. **(M)** Find the minimum of f(x) = x² − 10x + 4. Confirm with the second derivative.
5. **(M)** Find the maximum of f(p) = p(1 − p) for p between 0 and 1. (This is the variance of a coin flip; you'll meet
   it in Module 04.)
6. **(M)** Repeat the "mean minimises squared error" proof for the data (2, 4, 9), checking S(m) at the answer and at
   m ± 1.
7. **(M)** For g(x, y) = 2x² + xy − 3y, compute ∂g/∂x, ∂g/∂y and the gradient at (1, 1).
8. **(M)** Do 3 steps of gradient descent by hand on f(x) = (x + 1)² from x₀ = 2 with η = 0.25. What is the factor that
   multiplies the gap to the minimum each step?
9. **(P)** Implement `numerical_derivative(f, x, h=1e-5)` (central difference) in `statkit/mathkit.py`. Check it on
   every function from exercise 2 at x = 1 against your hand derivatives.
10. **(P)** Implement `gradient_descent(grad, x0, learning_rate, steps)` returning the final x. Use it on
    f(x) = x² − 10x + 4. Find a learning rate that converges in under 20 steps and one that diverges.

## ★★★ Challenge

11. **(M)** Show that σ′(z) = σ(z)(1 − σ(z)) for σ(z) = 1/(1 + e⁻ᶻ). (Hint: write σ = (1 + e⁻ᶻ)⁻¹ and use the chain rule.)
12. **(M)** For L(w) = (1/n) Σ (yᵢ − w xᵢ)² (a line through the origin), set dL/dw = 0 and show that
    w = Σxᵢyᵢ / Σxᵢ². Compute w for x = (1, 2, 3), y = (2, 4, 7).
13. **(P)** Verify exercise 12 with your `gradient_descent`, starting from w = 0.
14. **(P)** Show numerically that the median minimises Σ|xᵢ − m|: for data (1, 3, 4, 10, 12), evaluate the sum on a fine
    grid of m and find where it is smallest. Compare with the mean. Which is more affected if you change 12 to 1200?

Solutions: [solutions/solutions.md](solutions/solutions.md)
