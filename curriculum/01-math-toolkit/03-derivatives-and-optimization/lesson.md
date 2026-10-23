# 01.03 — Derivatives and optimisation

**Time:** about 4 hours · **Prerequisites:** 01.01, 01.02

## Why this matters

Almost every method in this course **chooses the best value of something**:

- the mean is the number closest (in squared distance) to all the data;
- maximum likelihood picks the parameter that makes the data most probable;
- regression picks the line with the smallest squared error;
- training a neural network picks millions of weights that minimise a loss.

"Best" means "minimum (or maximum) of a function", and derivatives are the tool that finds minima. Gradient descent,
the algorithm behind all of deep learning, is just "follow the derivative downhill".

## Learning objectives

- Explain a derivative as a slope and estimate it numerically.
- Differentiate using the power, sum, constant, exponential, log and chain rules.
- Find a minimum by setting the derivative to zero, and check it with the second derivative.
- Prove that the mean minimises the sum of squared deviations.
- Compute partial derivatives and a gradient.
- Run gradient descent by hand and explain the role of the learning rate.

## Before you start

Guess: for data 4, 7, 1, 8, which single number m makes (4 − m)² + (7 − m)² + (1 − m)² + (8 − m)² as small as
possible?

---

## 1. A derivative is a slope

For a straight line the slope is rise / run. For a curve, the slope changes from point to point. The **derivative**
f′(x) is the slope of the curve exactly at x: the slope of a tiny chord from x to x + h as h shrinks to 0.

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.$$

**Worked example 1 (numerically).** f(x) = x² at x = 3, with h = 0.001:

(f(3.001) − f(3)) / 0.001 = (9.006001 − 9) / 0.001 = **6.001**.

As h shrinks the answer approaches **6**. The **central difference** (f(x + h) − f(x − h)) / (2h) is more accurate:
(9.006001 − 8.994001) / 0.002 = **6.000**. You'll code this as `numerical_derivative`.

What the number means: near x = 3, increasing x by a small amount δ increases x² by about 6δ.

| f′(x) | The function is… |
| --- | --- |
| positive | going up (increasing) |
| negative | going down (decreasing) |
| zero | flat — possibly a minimum or maximum |

## 2. Rules for derivatives

You rarely use the limit definition. These rules cover everything in this course:

| Rule | Function | Derivative | Example |
| --- | --- | --- | --- |
| Constant | c | 0 | d/dx 7 = 0 |
| Power | xⁿ | n·xⁿ⁻¹ | d/dx x³ = 3x² |
| Constant multiple | c·f(x) | c·f′(x) | d/dx 5x² = 10x |
| Sum | f(x) + g(x) | f′(x) + g′(x) | d/dx (x² + 3x) = 2x + 3 |
| Exponential | eˣ | eˣ | |
| Logarithm | ln x | 1/x | |
| **Chain** | f(g(x)) | f′(g(x)) · g′(x) | see below |

**The chain rule** handles a function inside a function: differentiate the outside (leaving the inside alone), then
multiply by the derivative of the inside.

**Worked example 2.** h(x) = (3x + 1)². Outside: u² → 2u. Inside: u = 3x + 1 → 3.
h′(x) = 2(3x + 1) · 3 = **6(3x + 1)**. At x = 0: h′(0) = 6.

**Worked example 3.** h(x) = e^(−x²) (the bell-curve shape). Outside: eᵘ → eᵘ. Inside: u = −x² → −2x.
h′(x) = **−2x · e^(−x²)**. At x = 0 the slope is 0: the top of the bell.

**Worked example 4 (ML).** The sigmoid σ(z) = 1/(1 + e⁻ᶻ) has the neat derivative **σ′(z) = σ(z)(1 − σ(z))**.
At z = 0: 0.5 × 0.5 = 0.25. This formula is used in every backpropagation through a sigmoid. (Exercise 11 derives it.)

## 3. Finding a minimum

At the bottom of a smooth valley the curve is flat, so **f′(x) = 0**. To find a minimum:

1. Differentiate.
2. Set f′(x) = 0 and solve for x.
3. Check it's a minimum, not a maximum: the **second derivative** f″(x) (the derivative of the derivative) is
   positive at a minimum (the curve bends upwards, like a cup).

**Worked example 5.** f(x) = (x − 3)². f′(x) = 2(x − 3) = 0 ⇒ x = 3. f″(x) = 2 > 0 ⇒ minimum. ✓

### The big one: the mean minimises squared error

Let S(m) = Σᵢ (xᵢ − m)², the total squared distance from a number m to all the data. Which m makes it smallest?

1. Differentiate with respect to m (chain rule on each term: outside u² → 2u, inside xᵢ − m → −1):

   $$S'(m) = \sum_i 2(x_i - m)(-1) = -2\sum_i (x_i - m).$$

2. Set to zero: Σ(xᵢ − m) = 0 ⇒ Σxᵢ − n·m = 0 ⇒ **m = Σxᵢ / n = x̄**.
3. S″(m) = 2n > 0 ⇒ minimum. ✓

That answers "Before you start": for 4, 7, 1, 8 the best m is the mean, 5, giving S(5) = 1 + 4 + 16 + 9 = 30.
Check neighbours: S(4) = 0 + 9 + 9 + 16 = 34 and S(6) = 4 + 1 + 25 + 4 = 34, both larger. ✓

**Why this matters:** the mean is not just "add up and divide". It is the *solution to an optimisation problem*:
the best single-number summary under squared error. Change the error and you change the answer: minimising
Σ|xᵢ − m| (absolute error) gives the **median** (challenge exercise 14). This "choose a loss, minimise it" idea is
exactly how ML models are defined.

## 4. Several variables: partial derivatives and the gradient

Models have many parameters. For a function of several variables, the **partial derivative** ∂f/∂x is the
derivative with respect to x **treating all other variables as constants**.

**Worked example 6.** g(x, y) = x² + 3xy + y².

- ∂g/∂x = 2x + 3y (y is a constant, so 3xy → 3y and y² → 0)
- ∂g/∂y = 3x + 2y

At (1, 2): ∂g/∂x = 2 + 6 = 8 and ∂g/∂y = 3 + 4 = 7.

The **gradient** collects all partial derivatives into a vector: ∇g(1, 2) = (8, 7). It points in the direction of
**steepest increase**, so −∇g points steepest downhill.

**Worked example 7 (fitting a line).** For a line ŷ = w·x + b and the mean-squared-error loss
L(w, b) = (1/n) Σ (yᵢ − (w xᵢ + b))², the chain rule gives

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum_i x_i\,(y_i - \hat{y}_i), \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i (y_i - \hat{y}_i).$$

You'll use exactly these two formulas in the mini-project.

## 5. Gradient descent

Setting derivatives to zero and solving works for simple functions. For complicated ones (neural networks), we
instead **walk downhill in small steps**:

$$x_{\text{new}} = x_{\text{old}} - \eta \cdot f'(x_{\text{old}})$$

η (eta) is the **learning rate**, the step size. If the slope is positive, step left; if negative, step right.

**Worked example 8.** Minimise f(x) = (x − 3)², so f′(x) = 2(x − 3). Start at x₀ = 0 with η = 0.1.

| step | x | f′(x) = 2(x − 3) | new x = x − 0.1·f′(x) |
| --- | --- | --- | --- |
| 0 | 0 | −6 | 0 + 0.6 = **0.6** |
| 1 | 0.6 | −4.8 | 0.6 + 0.48 = **1.08** |
| 2 | 1.08 | −3.84 | 1.08 + 0.384 = **1.464** |
| … | | | |
| 20 | | | **2.965** |

Each step closes 20% of the remaining gap to 3, because x − 3 is multiplied by (1 − 2η) = 0.8 each time. After k steps
x = 3(1 − 0.8ᵏ).

**The learning rate matters:**

| η | Factor (1 − 2η) | Behaviour |
| --- | --- | --- |
| 0.1 | 0.8 | smooth, steady convergence |
| 0.5 | 0 | jumps to 3 in one step (lucky for this f) |
| 1.0 | −1 | bounces forever: 0, 6, 0, 6, … |
| 1.1 | −1.2 | **diverges**: 6.6, −1.32, 8.18, −3.22, … |

Too small: slow. Too large: oscillates or explodes. Choosing η is the first thing you tune when training any model.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| d/dx (3x + 1)² = 2(3x + 1) | Forgot the inner derivative: × 3 |
| d/dx eˣ² = eˣ² | Chain rule: 2x · eˣ² |
| f′(x) = 0 ⇒ minimum | Could be a maximum or flat point; check f″ |
| "Partial derivative" means something new | Same rules; just freeze the other variables |
| `x = x + lr * grad` | Gradient **descent** subtracts the gradient |

## Summary

- f′(x) is the slope at x; numerically ≈ (f(x + h) − f(x − h)) / 2h.
- Rules: power, sum, constant, eˣ, ln x → 1/x, chain.
- Minimum: f′ = 0 and f″ > 0.
- **The mean minimises Σ(xᵢ − m)²**; the median minimises Σ|xᵢ − m|.
- Gradient = vector of partial derivatives; descent steps along −gradient with learning rate η.

## Self-test

1. Differentiate 4x³ − 2x + 7 and ln(x²).
2. Find the minimum of f(x) = x² − 10x + 4.
3. Why does gradient descent subtract the gradient instead of adding it?
4. What happens with a learning rate that is too large?
5. Which summary statistic minimises squared error? Which minimises absolute error?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
