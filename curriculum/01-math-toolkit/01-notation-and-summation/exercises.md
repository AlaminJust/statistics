# 01.01 Exercises — Notation and summation

`make lesson L=01-math-toolkit/01-notation-and-summation`

## ★ Warm-up

1. **(M)** Compute Σ from i = 1 to 5 of i, and Σ from i = 1 to 5 of i².
2. **(M)** Data: x = (2, 4, 9). Find n, x₂, Σxᵢ, Σxᵢ², (Σxᵢ)² and x̄.
3. **(M)** Write in Σ notation: 3 + 6 + 9 + 12 + 15.
4. **(C)** Translate into words: (1/n) Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)².

## ★★ Core

5. **(M)** Using only the rules of summation, simplify Σᵢ₌₁ⁿ (3xᵢ + 2) in terms of Σxᵢ and n. Check with x = (2, 4, 9).
6. **(M)** Show that Σᵢ₌₁ⁿ (xᵢ − c) = Σxᵢ − n·c for any constant c. For which value of c is the result 0?
7. **(M)** For x = (2, 4, 9), compute Σ(xᵢ − x̄)² two ways: directly, and with Σxᵢ² − n x̄². Do they match?
8. **(M)** For x = (1, 2) and y = (3, 4), compute Σxᵢyᵢ and (Σxᵢ)(Σyᵢ). Explain why they differ.
9. **(M)** Compute Σᵢ₌₁³ Σⱼ₌₁² (i + j) using a table.
10. **(P)** Write a function `summation(f, start, stop)` that returns Σ from i = start to stop of f(i), using a loop.
    Test it on exercise 1 and on worked example 1 (answer 24). Then add it to `statkit/mathkit.py`.
11. **(P)** Write `product(values)` (Π) using a loop. Compute 6! and the probability that 10 independent events, each
    with probability 0.9, all happen.

## ★★★ Challenge

12. **(M)** Prove that Σᵢ₌₁ⁿ i = n(n + 1)/2. (Hint: write the sum forwards and backwards and add the two lines.)
    Check with `summation` for n = 100.
13. **(M)** The *weighted mean* is Σwᵢxᵢ / Σwᵢ. Show that if all weights are equal (wᵢ = w), it becomes the ordinary
    mean x̄.
14. **(P)** In ML, a mean-squared-error loss is L = (1/n) Σ(yᵢ − ŷᵢ)². For y = (3, 5, 7) and predictions
    ŷ = (2.5, 5.5, 8), compute L by hand, with a loop, and with NumPy in one line.

Solutions: [solutions/solutions.md](solutions/solutions.md)
