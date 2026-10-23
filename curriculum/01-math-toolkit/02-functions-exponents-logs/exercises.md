# 01.02 Exercises — Functions, exponents and logarithms

`make lesson L=01-math-toolkit/02-functions-exponents-logs`

## ★ Warm-up

1. **(M)** Simplify: 3² · 3³, (5²)³, 4^(1/2), 10⁻², e⁰.
2. **(M)** Evaluate without a calculator: log₂(32), log₁₀(0.01), ln(e⁴), e^(ln 9), ln(1).
3. **(M)** For f(x) = 40 + 2.2x, find f(0), f(5) and the x for which f(x) = 84.

## ★★ Core

4. **(M)** Using log rules, write ln(x³y / z) as a combination of ln x, ln y and ln z.
5. **(M)** Is ln(a + b) = ln a + ln b? Test with a = b = 1, then explain.
6. **(M)** Compute the odds and log-odds for p = 0.2, 0.5 and 0.9. What do you notice about p = 0.2 vs p = 0.8?
7. **(M)** Show that σ(−z) = 1 − σ(z). (Hint: multiply the top and bottom of σ(−z) by eᶻ.)
8. **(P)** Implement `sigmoid(z)` in `statkit/mathkit.py`. It must work on a single number. Check σ(0) = 0.5 and
   σ(ln 4) = 0.8.
9. **(P)** Implement `log_sum_exp(values)` using the max-subtraction trick (use `math.exp` and `math.log`). Check it on
   (1, 2, 3) against the naive formula, and on (1000, 1001, 1002) where the naive formula fails.
10. **(P)** Ten independent events each have probability 0.001. Compute the probability that all happen (a) directly
    and (b) as exp(Σ ln p). Now try 500 events. Which method still works?

## ★★★ Challenge

11. **(M)** Show that softmax(z + c) = softmax(z) for any constant c added to every score. Why does this make the
    log-sum-exp trick legal?
12. **(M)** Show that the inverse of σ is the logit: if p = σ(z) then z = ln(p / (1 − p)).
13. **(P)** Plot `exam_score` against `ln(study_hours)` and against `study_hours`. Which looks more like a straight line?
    What does that suggest about the true relationship (look at `data/make_students.py`)?

Solutions: [solutions/solutions.md](solutions/solutions.md)
