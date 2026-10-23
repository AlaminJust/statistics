# 01.04 — Integrals and area

**Time:** about 3 hours · **Prerequisites:** 01.01–01.03

## Why this matters

For a *continuous* quantity such as height or waiting time, the probability of any exact value (say 170.000… cm)
is zero. Probability lives in **areas under a curve** instead: the chance that a height falls between 165 and 175 cm
is the area under the height curve between those two points. An **integral** is how we compute that area.

You will not need to integrate hard functions by hand. You need to know what an integral *means*, a few basic results,
and how to compute one numerically.

## Learning objectives

- Explain an integral as the area under a curve and as the limit of a sum of rectangles.
- Compute Riemann sums and trapezoid approximations by hand.
- Integrate simple functions using antiderivatives.
- Check whether a function is a valid probability density and compute probabilities as areas.
- Compute an expected value as an integral.

## Before you start

Guess: a function is drawn from x = 0 to x = 1 as a straight line from height 0 up to height 2. What is the area
under it?

---

## 1. Area as a sum of rectangles

To find the area under f(x) from a to b, cut the interval into n strips of width Δx = (b − a)/n, draw a rectangle in
each, and add the rectangle areas:

$$\text{area} \approx \sum_{i=1}^{n} f(x_i)\,\Delta x.$$

This is a **Riemann sum**. With more, thinner strips the approximation gets better; the exact area is the limit as
n → ∞, written

$$\int_a^b f(x)\,dx.$$

The ∫ sign is a stretched S for "sum", and dx plays the role of Δx. **An integral is a Σ with infinitely thin strips.**

### Worked example 1: the area under x² from 0 to 2, by rectangles

Use n = 4 strips, so Δx = 0.5. The heights depend on where in each strip you measure:

| Strip | Left end x | f = x² | Right end x | f = x² | Midpoint x | f = x² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0.5 | 0.25 | 0.25 | 0.0625 |
| 2 | 0.5 | 0.25 | 1 | 1 | 0.75 | 0.5625 |
| 3 | 1 | 1 | 1.5 | 2.25 | 1.25 | 1.5625 |
| 4 | 1.5 | 2.25 | 2 | 4 | 1.75 | 3.0625 |
| **sum × 0.5** | | **1.75** | | **3.75** | | **2.625** |

The left sum underestimates (the curve rises inside each strip), the right sum overestimates. The exact answer, found
in section 3, is 8/3 ≈ **2.667**. The midpoint sum is already close.

## 2. The trapezoid rule

Replace each rectangle with a trapezoid joining the curve at both ends of the strip. Its area is
Δx · (f(left) + f(right)) / 2, so the trapezoid estimate is the average of the left and right sums:

(1.75 + 3.75) / 2 = **2.75** with 4 strips. With 100 strips it is **2.6668**, within 0.0002 of the truth.
This is what `np.trapezoid` and your `statkit.integrate` compute.

## 3. Exact integrals with antiderivatives

The **fundamental theorem of calculus** connects integrals and derivatives: if F′(x) = f(x), then

$$\int_a^b f(x)\,dx = F(b) - F(a).$$

F is an **antiderivative**: a function whose derivative is f. Reverse the derivative rules:

| f(x) | Antiderivative F(x) | Check: F′(x) |
| --- | --- | --- |
| c | c·x | c |
| xⁿ (n ≠ −1) | xⁿ⁺¹ / (n + 1) | xⁿ |
| eˣ | eˣ | eˣ |
| e⁻ˣ | −e⁻ˣ | e⁻ˣ |
| 1/x | ln x | 1/x |

**Worked example 2.** ∫₀² x² dx. F(x) = x³/3. F(2) − F(0) = 8/3 − 0 = **8/3 ≈ 2.667**. ✓ matches section 1.

**Worked example 3.** ∫₀¹ 2x dx. F(x) = x². F(1) − F(0) = **1**. This answers "Before you start": the shape is a
triangle with base 1 and height 2, area ½ × 1 × 2 = 1.

**Worked example 4.** ∫₀^∞ e⁻ˣ dx. F(x) = −e⁻ˣ. As x → ∞, −e⁻ˣ → 0, and F(0) = −1. Area = 0 − (−1) = **1**.
An infinitely long region can have a finite area.

## 4. Probability densities: area = probability

A **probability density function** (pdf) f(x) describes a continuous random quantity. It must satisfy two rules:

1. f(x) ≥ 0 everywhere;
2. the total area is 1: ∫ f(x) dx = 1 over all possible values.

Then **P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx**.

From worked examples 3 and 4, both f(x) = 2x on [0, 1] and f(x) = e⁻ˣ on [0, ∞) are valid densities. (The second is
the *exponential distribution*, a model for waiting times, Module 04.)

**Worked example 5.** X has density f(x) = 2x on [0, 1]. Probability X ≤ 0.5?

$$P(X \le 0.5) = \int_0^{0.5} 2x\,dx = [x^2]_0^{0.5} = 0.25.$$

Only 25%, even though 0.5 is halfway: the density is larger near 1, so values near 1 are more likely.

**Worked example 6.** Waiting time X ~ density e⁻ˣ (in minutes). P(wait longer than 2 minutes)?

$$P(X > 2) = \int_2^\infty e^{-x}\,dx = [-e^{-x}]_2^\infty = 0 + e^{-2} \approx 0.135.$$

**A density value is not a probability.** f(x) = 2x gives f(1) = 2, which is fine. Only *areas* must be ≤ 1.

## 5. Expected value as an integral

For a list of numbers, the mean is Σ value × weight. For a continuous variable, the **expected value** replaces Σ by ∫
and the weight by the density:

$$E[X] = \int x\,f(x)\,dx.$$

**Worked example 7.** For f(x) = 2x on [0, 1]: E[X] = ∫₀¹ x · 2x dx = ∫₀¹ 2x² dx = [2x³/3]₀¹ = **2/3**.
The average is above 0.5 because the density leans right. This formula will be used constantly from Module 03 onwards.

## 6. Numerically, the same way you'd code a Σ

```python
def integrate(f, a, b, n=1000):
    dx = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * dx)
    return total * dx              # trapezoid rule
```

`examples.py` uses it to check that the standard normal density, e^(−x²/2)/√(2π), integrates to 1 (over −10 to 10,
because the tails beyond are negligible).

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| "A density can't exceed 1" | It can; only areas (probabilities) are capped at 1 |
| P(X = 0.5) = f(0.5) | For continuous X, P(X = any single value) = 0 |
| Forgetting F(a) in F(b) − F(a) | Always subtract the lower limit |
| ∫ xⁿ dx = xⁿ⁺¹ | Divide by n + 1 |

## Summary

- ∫ₐᵇ f(x) dx = area under f from a to b = limit of Σ f(xᵢ)Δx.
- Trapezoid rule = average of left and right Riemann sums; accurate for many strips.
- Exact: ∫ₐᵇ f = F(b) − F(a) where F′ = f.
- Density: f ≥ 0, total area 1, probability = area. Expected value = ∫ x f(x) dx.

## Self-test

1. Compute ∫₁³ (2x + 1) dx exactly.
2. Is f(x) = 3x² on [0, 1] a valid density? Find P(X ≤ 0.5).
3. Why is the probability of an exact value 0 for a continuous variable?
4. Write the integral for the expected value of X with density e⁻ˣ on [0, ∞).

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
