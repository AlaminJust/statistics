# 01.02 Solutions

1. 3⁵ = 243; 5⁶ = 15625; 2; 0.01; 1.
2. 5; −2; 4; 9; 0.
3. f(0) = 40, f(5) = 51; 40 + 2.2x = 84 ⇒ x = 44 / 2.2 = **20**.
4. ln(x³y / z) = **3 ln x + ln y − ln z**.
5. No. With a = b = 1: ln(2) = 0.693, but ln 1 + ln 1 = 0. The rule is ln(ab) = ln a + ln b, for products only.
6. | p | odds | log-odds |
   | --- | --- | --- |
   | 0.2 | 0.25 | −1.386 |
   | 0.5 | 1 | 0 |
   | 0.9 | 9 | 2.197 |

   p = 0.2 and p = 0.8 have log-odds −1.386 and +1.386: opposite events have opposite log-odds (symmetric around 0).
7. σ(−z) = 1/(1 + eᶻ). Multiply top and bottom by e⁻ᶻ: e⁻ᶻ / (e⁻ᶻ + 1). And 1 − σ(z) = 1 − 1/(1 + e⁻ᶻ) =
   e⁻ᶻ / (1 + e⁻ᶻ). Same. ✓
8. ```python
   def sigmoid(z):
       return 1 / (1 + math.exp(-z))
   ```
   (For very negative z, `math.exp(-z)` overflows; the reference version in `statkit_reference.py` handles this by
   using e^z / (1 + e^z) when z < 0.)
9. ```python
   def log_sum_exp(values):
       m = max(values)
       return m + math.log(sum(math.exp(v - m) for v in values))
   ```
   (1, 2, 3) → 3.4076 both ways. (1000, 1001, 1002) → 1002.4076; the naive version raises `OverflowError` or gives `inf`.
10. (a) 0.001¹⁰ = 1e-30 (fine). (b) exp(10 · ln 0.001) = 1e-30. For 500 events the true value is 1e-1500: the direct
    product gives 0.0 (underflow) and exp of the log-sum also gives 0.0 — but the **log** value, 500 · ln(0.001) =
    −3453.9, is still exact. This is why you keep everything on the log scale and never exponentiate back.
11. softmax(z + c)ₖ = e^(zₖ + c) / Σ e^(zⱼ + c) = eᶜ e^(zₖ) / (eᶜ Σ e^(zⱼ)) = softmax(z)ₖ. Choosing c = −max(z)
    changes nothing in the result but makes every exponent ≤ 0, so nothing overflows.
12. p = 1/(1 + e⁻ᶻ) ⇒ 1 + e⁻ᶻ = 1/p ⇒ e⁻ᶻ = (1 − p)/p ⇒ −z = ln((1 − p)/p) ⇒ z = ln(p/(1 − p)). ✓
13. `exam_score` vs `study_hours` is closer to a straight line, because `make_students.py` builds the score as
    40 + **2.2 · study_hours** + …: linear in hours, not in log-hours.
